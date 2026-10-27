# okc: problem=muse_moe_w4a16
"""Triton NVFP4 MoE experts, v1: expert-major, dequant-in-kernel, 4 kernels.

_route_prep: per-local-expert route lists (deterministic scan, no atomics).
_k1:  x @ dequant(w13)^T with SwiGLU fused in the epilogue (reads w13 once per touched expert).
_k2:  inter @ dequant(w2)^T.
_k3:  per-route RMSNorm(1+w) and weighted sum into the token's output row.

Workspaces are shape-dependent torch.empty allocations (captured by the CUDA graph);
no host syncs, no value-dependent control flow."""
import torch
import triton
import triton.language as tl

POST_NORM_EPS = 1e-8


@triton.jit
def _e2m1(c):
    """E2M1 code (0..15) -> float value."""
    m = c & 7
    mag = tl.where(m < 2, m.to(tl.float32) * 0.5,
                   tl.where(m < 4, m.to(tl.float32) - 1.0,
                            tl.exp2((m // 2).to(tl.float32) - 1.0)
                            * (1.0 + (m & 1).to(tl.float32) * 0.5)))
    return tl.where((c & 8) != 0, -mag, mag)


@triton.jit
def _dq(b):
    """Packed byte (low nibble = even element) -> fp32 pair, interleaved to [..., 2K]."""
    lo = _e2m1((b & 0x0F).to(tl.int32))
    hi = _e2m1((b >> 4).to(tl.int32))
    return tl.interleave(lo, hi)


@triton.jit
def _route_prep(ids_ptr, list_ptr, counts_ptr, M8, start, stride_l, BLOCK: tl.constexpr):
    """For each local expert e: sorted list of route slots (t*8+k) routed to it, plus count."""
    e = tl.program_id(0)
    gid = start + e
    run = 0
    for f0 in tl.range(0, M8, BLOCK):
        f = f0 + tl.arange(0, BLOCK)
        m = f < M8
        idv = tl.load(ids_ptr + f, mask=m, other=-1)
        mi = ((idv == gid) & m).to(tl.int32)
        pos = run + tl.cumsum(mi, 0) - mi
        tl.store(list_ptr + e * stride_l + pos, f, mask=mi != 0)
        run += tl.sum(mi, 0)
    tl.store(counts_ptr + e, run)


@triton.jit
def _k1(x_ptr, list_ptr, counts_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr,
        M8, start, stride_l,
        H: tl.constexpr, I: tl.constexpr, BR: tl.constexpr, KC: tl.constexpr, CP: tl.constexpr):
    """h = x @ dequant(w13[e])^T for the expert's routes; store silu(gate)*up to inter[ent, I]."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * (2 * I)
    j = tl.program_id(1) * BR + tl.arange(0, BR)
    jm = j < I
    for c0 in tl.range(0, c, CP):
        pe = c0 + tl.arange(0, CP)
        pv = pe < c
        ent = tl.load(list_ptr + e * stride_l + pe, mask=pv, other=0)
        tok = ent // 8
        accg = tl.zeros((CP, BR), tl.float32)
        accu = tl.zeros((CP, BR), tl.float32)
        for k0 in tl.range(0, H, KC):
            kk = k0 + tl.arange(0, KC)
            kkh = k0 // 2 + tl.arange(0, KC // 2)
            a = tl.load(x_ptr + tok[:, None] * H + kk[None, :], mask=pv[:, None], other=0.0)
            bg = tl.load(w_ptr + wbase + j[:, None] * (H // 2) + kkh[None, :],
                         mask=jm[:, None], other=0)
            sg = tl.load(sf_ptr + wbase + j[:, None] * (H // 16) + (kk // 16)[None, :],
                         mask=jm[:, None], other=0.0)
            wg = (_dq(bg) * (sg.to(tl.float32) * gs)).to(tl.bfloat16)
            accg = tl.dot(a, tl.trans(wg), accg)
            bu = tl.load(w_ptr + wbase + (I + j)[:, None] * (H // 2) + kkh[None, :],
                         mask=jm[:, None], other=0)
            su = tl.load(sf_ptr + wbase + (I + j)[:, None] * (H // 16) + (kk // 16)[None, :],
                         mask=jm[:, None], other=0.0)
            wu = (_dq(bu) * (su.to(tl.float32) * gs)).to(tl.bfloat16)
            accu = tl.dot(a, tl.trans(wu), accu)
        inter = accg * tl.sigmoid(accg) * accu
        tl.store(out_ptr + ent[:, None] * I + j[None, :], inter.to(tl.bfloat16),
                 mask=pv[:, None] & jm[None, :])


@triton.jit
def _k2(in_ptr, list_ptr, counts_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr,
        start, stride_l,
        H: tl.constexpr, I: tl.constexpr, BR: tl.constexpr, KC: tl.constexpr, CP: tl.constexpr):
    """y = inter @ dequant(w2[e])^T for the expert's routes, into y[ent, H]."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * H
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    rm = r < H
    for c0 in tl.range(0, c, CP):
        pe = c0 + tl.arange(0, CP)
        pv = pe < c
        ent = tl.load(list_ptr + e * stride_l + pe, mask=pv, other=0)
        acc = tl.zeros((CP, BR), tl.float32)
        for k0 in tl.range(0, I, KC):
            kk = k0 + tl.arange(0, KC)
            kkh = k0 // 2 + tl.arange(0, KC // 2)
            a = tl.load(in_ptr + ent[:, None] * I + kk[None, :], mask=pv[:, None], other=0.0)
            b = tl.load(w_ptr + wbase + r[:, None] * (I // 2) + kkh[None, :],
                        mask=rm[:, None], other=0)
            s = tl.load(sf_ptr + wbase + r[:, None] * (I // 16) + (kk // 16)[None, :],
                        mask=rm[:, None], other=0.0)
            w = (_dq(b) * (s.to(tl.float32) * gs)).to(tl.bfloat16)
            acc = tl.dot(a, tl.trans(w), acc)
        tl.store(out_ptr + ent[:, None] * H + r[None, :], acc.to(tl.bfloat16),
                 mask=pv[:, None] & rm[None, :])


@triton.jit
def _k3(y_ptr, ids_ptr, gw_ptr, nw_ptr, out_ptr, start, count,
        H: tl.constexpr, EPS: tl.constexpr):
    """out[t] = sum over local routes of topk_weight * rmsnorm(y * (1 + post_norm_w))."""
    t = tl.program_id(0)
    h = tl.arange(0, H)
    nw = tl.load(nw_ptr + h).to(tl.float32)
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(8):
        f = t * 8 + k
        idv = tl.load(ids_ptr + f)
        lok = (idv >= start) & (idv < start + count)
        yv = tl.load(y_ptr + f * H + h).to(tl.float32)
        s = yv * (1.0 + nw)
        ms = tl.sum(s * s, 0) / H
        gwk = tl.load(gw_ptr + f)
        acc += tl.where((h < H) & lok, s * tl.rsqrt(ms + EPS) * gwk, 0.0)
    tl.store(out_ptr + t * H + h, acc.to(tl.bfloat16))


def prepare(weights):
    return weights


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    ids = data["topk_ids"]
    gwt = data["topk_weights"]
    M, H = x.shape
    I = cfg["d_expert"]
    start = cfg["local_start"]
    count = cfg["local_count"]
    dev = x.device
    M8 = M * 8

    list_ws = torch.empty((count, M8), dtype=torch.int32, device=dev)
    counts = torch.empty((count,), dtype=torch.int32, device=dev)
    BLOCK_P = max(16, triton.next_power_of_2(M8))
    _route_prep[(count,)](ids, list_ws, counts, M8, start, M8, BLOCK=BLOCK_P, num_warps=8)

    CP = min(max(triton.next_power_of_2(M), 16), 128)
    inter = torch.empty((M8, I), dtype=torch.bfloat16, device=dev)
    y = torch.empty((M8, H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    _k1[(count, triton.cdiv(I, 64))](x, list_ws, counts, state["w13"], state["w13_sf"],
                                     state["w13_gs"], inter, M8, start, M8,
                                     H=H, I=I, BR=64, KC=128, CP=CP, num_warps=8, num_stages=2)
    _k2[(count, triton.cdiv(H, 64))](inter, list_ws, counts, state["w2"], state["w2_sf"],
                                     state["w2_gs"], y, start, M8,
                                     H=H, I=I, BR=64, KC=128, CP=CP, num_warps=8, num_stages=2)
    _k3[(M,)](y, ids, gwt, state["post_norm_w"], out, start, count,
              H=H, EPS=POST_NORM_EPS, num_warps=8)
    return out
