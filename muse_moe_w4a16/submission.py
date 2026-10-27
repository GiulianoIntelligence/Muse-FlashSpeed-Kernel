# okc: problem=muse_moe_w4a16
"""Triton NVFP4 MoE experts, v2: expert-major, dequant-in-kernel, 4 kernels.

prepare() repacks each weight tensor transposed ([rows, K/2] -> [K/2, rows], same NVFP4
bytes) so the GEMM B-operand loads directly in [K, N] layout: no in-register transpose,
no nibble interleave (even/odd x split instead: dot(x_even, lo) + dot(x_odd, hi)).
Kernels are dtype-generic (element_ty casts) so TRITON_INTERPRET can validate them.

_route_prep: per-local-expert route lists (deterministic scan, no atomics).
_k1:  h = x @ dequant(w13[e])^T with SwiGLU fused in the epilogue.
_k2:  y = inter @ dequant(w2[e])^T.
_k3:  per-route RMSNorm(1+w) and weighted sum into the token's output row."""
import torch
import triton
import triton.language as tl

POST_NORM_EPS = 1e-8


@triton.jit
def _e2m1(c):
    """E2M1 code (0..15) -> float value; magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6)."""
    m = c & 7
    mag = tl.where(m < 4, m.to(tl.float32) * 0.5,
                   tl.exp2((m // 2).to(tl.float32) - 1.0)
                   * (1.0 + (m & 1).to(tl.float32) * 0.5))
    return tl.where((c & 8) != 0, -mag, mag)


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
def _k1(x_ptr, list_ptr, counts_ptr, wt_ptr, sft_ptr, gs_ptr, out_ptr,
        M8, start, stride_l,
        H: tl.constexpr, I: tl.constexpr, BR: tl.constexpr, KC: tl.constexpr, CP: tl.constexpr):
    """inter[ent, I] = silu(x @ gate^T) * (x @ up^T), dequant in-kernel.

    wt: w13 repacked [E, H/2, 2I] — byte kb of column j holds elements (2kb, 2kb+1) of
    weight row j; columns [0, I) are gate rows, [I, 2I) up rows. sft: [E, H/16, 2I]."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * (H // 2) * (2 * I)
    sbase = gid64 * (H // 16) * (2 * I)
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
            kb = k0 // 2 + tl.arange(0, KC // 2)
            aev = tl.load(x_ptr + tok[:, None] * H + (2 * kb)[None, :],
                          mask=pv[:, None], other=0.0)
            aod = tl.load(x_ptr + tok[:, None] * H + (2 * kb + 1)[None, :],
                          mask=pv[:, None], other=0.0)
            bg = tl.load(wt_ptr + wbase + kb[:, None] * (2 * I) + j[None, :],
                         mask=jm[None, :], other=0)
            sg = tl.load(sft_ptr + sbase + (kb // 8)[:, None] * (2 * I) + j[None, :],
                         mask=jm[None, :], other=0.0)
            sc = sg.to(tl.float32) * gs
            wg_lo = (_e2m1(bg & 0xF) * sc).to(aev.dtype)
            wg_hi = (_e2m1(bg >> 4) * sc).to(aev.dtype)
            bu = tl.load(wt_ptr + wbase + kb[:, None] * (2 * I) + (I + j)[None, :],
                         mask=jm[None, :], other=0)
            su = tl.load(sft_ptr + sbase + (kb // 8)[:, None] * (2 * I) + (I + j)[None, :],
                         mask=jm[None, :], other=0.0)
            su = su.to(tl.float32) * gs
            wu_lo = (_e2m1(bu & 0xF) * su).to(aev.dtype)
            wu_hi = (_e2m1(bu >> 4) * su).to(aev.dtype)
            accg = tl.dot(aev, wg_lo, accg)
            accg = tl.dot(aod, wg_hi, accg)
            accu = tl.dot(aev, wu_lo, accu)
            accu = tl.dot(aod, wu_hi, accu)
        inter = accg * tl.sigmoid(accg) * accu
        tl.store(out_ptr + ent[:, None] * I + j[None, :],
                 inter.to(out_ptr.dtype.element_ty), mask=pv[:, None] & jm[None, :])


@triton.jit
def _k2(in_ptr, list_ptr, counts_ptr, wt_ptr, sft_ptr, gs_ptr, out_ptr,
        start, stride_l,
        H: tl.constexpr, I: tl.constexpr, BR: tl.constexpr, KC: tl.constexpr, CP: tl.constexpr):
    """y[ent, H] = inter @ dequant(w2[e])^T. wt: w2 repacked [E, I/2, H]; sft: [E, I/16, H]."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * (I // 2) * H
    sbase = gid64 * (I // 16) * H
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    rm = r < H
    for c0 in tl.range(0, c, CP):
        pe = c0 + tl.arange(0, CP)
        pv = pe < c
        ent = tl.load(list_ptr + e * stride_l + pe, mask=pv, other=0)
        acc = tl.zeros((CP, BR), tl.float32)
        for k0 in tl.range(0, I, KC):
            kb = k0 // 2 + tl.arange(0, KC // 2)
            aev = tl.load(in_ptr + ent[:, None] * I + (2 * kb)[None, :],
                          mask=pv[:, None], other=0.0)
            aod = tl.load(in_ptr + ent[:, None] * I + (2 * kb + 1)[None, :],
                          mask=pv[:, None], other=0.0)
            b = tl.load(wt_ptr + wbase + kb[:, None] * H + r[None, :],
                        mask=rm[None, :], other=0)
            s = tl.load(sft_ptr + sbase + (kb // 8)[:, None] * H + r[None, :],
                        mask=rm[None, :], other=0.0)
            sc = s.to(tl.float32) * gs
            w_lo = (_e2m1(b & 0xF) * sc).to(aev.dtype)
            w_hi = (_e2m1(b >> 4) * sc).to(aev.dtype)
            acc = tl.dot(aev, w_lo, acc)
            acc = tl.dot(aod, w_hi, acc)
        tl.store(out_ptr + ent[:, None] * H + r[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=pv[:, None] & rm[None, :])


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
    tl.store(out_ptr + t * H + h, acc.to(out_ptr.dtype.element_ty))


def prepare(weights):
    """Transpose the packed NVFP4 tensors (same bytes, K-major layout) for coalesced
    B-operand loads; precision and values unchanged."""
    return {
        "w13T": weights["w13"].transpose(1, 2).contiguous(),        # [E, H/2, 2I] uint8
        "w13_sfT": weights["w13_sf"].transpose(1, 2).contiguous(),  # [E, H/16, 2I] fp8
        "w13_gs": weights["w13_gs"],
        "w2T": weights["w2"].transpose(1, 2).contiguous(),          # [E, I/2, H] uint8
        "w2_sfT": weights["w2_sf"].transpose(1, 2).contiguous(),    # [E, I/16, H] fp8
        "w2_gs": weights["w2_gs"],
        "post_norm_w": weights["post_norm_w"],
    }


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
    inter = torch.empty((M8, I), dtype=x.dtype, device=dev)
    y = torch.empty((M8, H), dtype=x.dtype, device=dev)
    out = torch.empty((M, H), dtype=x.dtype, device=dev)

    _k1[(count, triton.cdiv(I, 64))](x, list_ws, counts, state["w13T"], state["w13_sfT"],
                                     state["w13_gs"], inter, M8, start, M8,
                                     H=H, I=I, BR=64, KC=128, CP=CP, num_warps=8, num_stages=1)
    _k2[(count, triton.cdiv(H, 64))](inter, list_ws, counts, state["w2T"], state["w2_sfT"],
                                     state["w2_gs"], y, start, M8,
                                     H=H, I=I, BR=64, KC=128, CP=CP, num_warps=8, num_stages=1)
    _k3[(M,)](y, ids, gwt, state["post_norm_w"], out, start, count,
              H=H, EPS=POST_NORM_EPS, num_warps=8)
    return out
