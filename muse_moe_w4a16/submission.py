# okc: problem=muse_moe_w4a16
"""Triton NVFP4 MoE experts, v3: expert-major, dequant-in-kernel, 4 kernels.

prepare() repacks weights (same NVFP4 bytes): transposed to [K/2, rows] and word-packed
4 bytes/int32, so weight tiles load as u32 words (vectorizable) along the row axis and
are unpacked to bytes in registers. Scales load as true [KC/16, rows] fp8 tiles (8x
smaller than v2's gather) and broadcast per 16-element block. No tl.trans, no interleave.
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
def _word_bytes(w):
    """int32 word -> 4 byte values stacked along two new trailing axes (flatten outside)."""
    b0 = w & 0xFF
    b1 = (w >> 8) & 0xFF
    b2 = (w >> 16) & 0xFF
    b3 = (w >> 24) & 0xFF
    return tl.join(tl.join(b0, b2), tl.join(b1, b3))


@triton.jit
def _k1(x_ptr, list_ptr, counts_ptr, wt_ptr, sft_ptr, gs_ptr, out_ptr,
        M8, start, stride_l,
        H: tl.constexpr, I: tl.constexpr, BR: tl.constexpr, KC: tl.constexpr, CP: tl.constexpr):
    """inter[ent, I] = silu(x @ gate^T) * (x @ up^T), dequant in-kernel, row-major words.

    wt: w13 viewed [E, 2I, H/8] int32 — word (j, kw) packs bytes 4kw..4kw+3 of weight
    row j (2048B contiguous rows); low nibble = element 2kb, high = 2kb+1. Rows [0, I)
    are gate, [I, 2I) up. Weights are the dot A-operand (k-major)."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * (2 * I) * (H // 8)
    sbase = gid64 * (2 * I) * (H // 16)
    j = tl.program_id(1) * BR + tl.arange(0, BR)
    jm = j < I
    for c0 in tl.range(0, c, CP):
        pe = c0 + tl.arange(0, CP)
        pv = pe < c
        ent = tl.load(list_ptr + e * stride_l + pe, mask=pv, other=0)
        tok = ent // 8
        accg = tl.zeros((BR, CP), tl.float32)
        accu = tl.zeros((BR, CP), tl.float32)
        for k0 in tl.range(0, H, KC):
            kw = k0 // 8 + tl.arange(0, KC // 8)
            kb = k0 // 2 + tl.arange(0, KC // 2)
            kb8 = k0 // 16 + tl.arange(0, KC // 16)
            aev = tl.load(x_ptr + tok[:, None] * H + (2 * kb)[None, :],
                          mask=pv[:, None], other=0.0)
            aod = tl.load(x_ptr + tok[:, None] * H + (2 * kb + 1)[None, :],
                          mask=pv[:, None], other=0.0)
            aevT = tl.trans(aev)
            aodT = tl.trans(aod)
            bw = tl.load(wt_ptr + wbase + j[:, None] * (H // 8) + kw[None, :],
                         mask=jm[:, None], other=0)
            bg = tl.reshape(_word_bytes(bw), (BR, KC // 2))
            sw = tl.load(sft_ptr + sbase + j[:, None] * (H // 16) + kb8[None, :],
                         mask=jm[:, None], other=0.0)
            sc = tl.reshape(tl.broadcast_to(sw[:, :, None], (BR, KC // 16, 8)),
                            (BR, KC // 2)).to(tl.float32) * gs
            wg_lo = (_e2m1(bg & 0xF) * sc).to(aev.dtype)
            wg_hi = (_e2m1(bg >> 4) * sc).to(aev.dtype)
            bwu = tl.load(wt_ptr + wbase + (I + j)[:, None] * (H // 8) + kw[None, :],
                          mask=jm[:, None], other=0)
            bu = tl.reshape(_word_bytes(bwu), (BR, KC // 2))
            swu = tl.load(sft_ptr + sbase + (I + j)[:, None] * (H // 16) + kb8[None, :],
                          mask=jm[:, None], other=0.0)
            scu = tl.reshape(tl.broadcast_to(swu[:, :, None], (BR, KC // 16, 8)),
                             (BR, KC // 2)).to(tl.float32) * gs
            wu_lo = (_e2m1(bu & 0xF) * scu).to(aev.dtype)
            wu_hi = (_e2m1(bu >> 4) * scu).to(aev.dtype)
            accg = tl.dot(wg_lo, aevT, accg)
            accg = tl.dot(wg_hi, aodT, accg)
            accu = tl.dot(wu_lo, aevT, accu)
            accu = tl.dot(wu_hi, aodT, accu)
        interT = accg * tl.sigmoid(accg) * accu
        tl.store(out_ptr + ent[None, :] * I + j[:, None],
                 interT.to(out_ptr.dtype.element_ty), mask=jm[:, None] & pv[None, :])


@triton.jit
def _k2(in_ptr, list_ptr, counts_ptr, wt_ptr, sft_ptr, gs_ptr, out_ptr,
        start, stride_l,
        H: tl.constexpr, I: tl.constexpr, BR: tl.constexpr, KC: tl.constexpr, CP: tl.constexpr):
    """y[ent, H] = inter @ dequant(w2[e])^T, row-major words. wt: w2 viewed [E, H, I/8];
    sft: [E, H, I/16] fp8."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * H * (I // 8)
    sbase = gid64 * H * (I // 16)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    rm = r < H
    for c0 in tl.range(0, c, CP):
        pe = c0 + tl.arange(0, CP)
        pv = pe < c
        ent = tl.load(list_ptr + e * stride_l + pe, mask=pv, other=0)
        acc = tl.zeros((BR, CP), tl.float32)
        for k0 in tl.range(0, I, KC):
            kw = k0 // 8 + tl.arange(0, KC // 8)
            kb = k0 // 2 + tl.arange(0, KC // 2)
            kb8 = k0 // 16 + tl.arange(0, KC // 16)
            aev = tl.load(in_ptr + ent[:, None] * I + (2 * kb)[None, :],
                          mask=pv[:, None], other=0.0)
            aod = tl.load(in_ptr + ent[:, None] * I + (2 * kb + 1)[None, :],
                          mask=pv[:, None], other=0.0)
            aevT = tl.trans(aev)
            aodT = tl.trans(aod)
            bw = tl.load(wt_ptr + wbase + r[:, None] * (I // 8) + kw[None, :],
                         mask=rm[:, None], other=0)
            b = tl.reshape(_word_bytes(bw), (BR, KC // 2))
            sw = tl.load(sft_ptr + sbase + r[:, None] * (I // 16) + kb8[None, :],
                         mask=rm[:, None], other=0.0)
            sc = tl.reshape(tl.broadcast_to(sw[:, :, None], (BR, KC // 16, 8)),
                            (BR, KC // 2)).to(tl.float32) * gs
            w_lo = (_e2m1(b & 0xF) * sc).to(aev.dtype)
            w_hi = (_e2m1(b >> 4) * sc).to(aev.dtype)
            acc = tl.dot(w_lo, aevT, acc)
            acc = tl.dot(w_hi, aodT, acc)
        tl.store(out_ptr + ent[None, :] * H + r[:, None],
                 acc.to(out_ptr.dtype.element_ty), mask=rm[:, None] & pv[None, :])


@triton.jit
def _gv1(x_ptr, list_ptr, counts_ptr, wt_ptr, sft_ptr, gs_ptr, out_ptr,
         M8, start, stride_l,
         H: tl.constexpr, I: tl.constexpr, RP: tl.constexpr, KC: tl.constexpr,
         RPAD: tl.constexpr):
    """Dot-free GEMV, row-major words: inter[ent, j] = silu(x@gate[j]) * (x@up[j]).

    wt: w13 viewed [E, 2I, H/8] int32 — word (j, kw) packs bytes 4kw..4kw+3 of weight
    row j (contiguous 2048B rows -> DRAM-friendly). sft: [E, 2I, H/16] fp8."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * (2 * I) * (H // 8)
    sbase = gid64 * (2 * I) * (H // 16)
    j = tl.program_id(1) * RP + tl.arange(0, RP)
    jm = j < I
    r8 = tl.arange(0, RPAD)
    ents = tl.load(list_ptr + e * stride_l + r8, mask=r8 < c, other=0)
    toks = ents // 8
    accg = tl.zeros((RPAD, RP), tl.float32)
    accu = tl.zeros((RPAD, RP), tl.float32)
    for k0 in tl.range(0, H, KC):
        kw = k0 // 8 + tl.arange(0, KC // 8)
        kb = k0 // 2 + tl.arange(0, KC // 2)
        kb8 = k0 // 16 + tl.arange(0, KC // 16)
        xev = tl.load(x_ptr + toks[:, None] * H + (2 * kb)[None, :],
                      mask=(r8 < c)[:, None], other=0.0).to(tl.float32)
        xod = tl.load(x_ptr + toks[:, None] * H + (2 * kb + 1)[None, :],
                      mask=(r8 < c)[:, None], other=0.0).to(tl.float32)
        bw = tl.load(wt_ptr + wbase + j[:, None] * (H // 8) + kw[None, :],
                     mask=jm[:, None], other=0)
        bg = tl.reshape(_word_bytes(bw), (RP, KC // 2))
        sw = tl.load(sft_ptr + sbase + j[:, None] * (H // 16) + kb8[None, :],
                     mask=jm[:, None], other=0.0)
        sc = tl.reshape(tl.broadcast_to(sw[:, :, None], (RP, KC // 16, 8)),
                        (RP, KC // 2)).to(tl.float32) * gs
        wg_lo = (_e2m1(bg & 0xF) * sc).to(tl.float32)
        wg_hi = (_e2m1(bg >> 4) * sc).to(tl.float32)
        accg += tl.sum(xev[:, None, :] * wg_lo[None, :, :], 2)
        accg += tl.sum(xod[:, None, :] * wg_hi[None, :, :], 2)
        bwu = tl.load(wt_ptr + wbase + (I + j)[:, None] * (H // 8) + kw[None, :],
                      mask=jm[:, None], other=0)
        bu = tl.reshape(_word_bytes(bwu), (RP, KC // 2))
        swu = tl.load(sft_ptr + sbase + (I + j)[:, None] * (H // 16) + kb8[None, :],
                      mask=jm[:, None], other=0.0)
        scu = tl.reshape(tl.broadcast_to(swu[:, :, None], (RP, KC // 16, 8)),
                         (RP, KC // 2)).to(tl.float32) * gs
        wu_lo = (_e2m1(bu & 0xF) * scu).to(tl.float32)
        wu_hi = (_e2m1(bu >> 4) * scu).to(tl.float32)
        accu += tl.sum(xev[:, None, :] * wu_lo[None, :, :], 2)
        accu += tl.sum(xod[:, None, :] * wu_hi[None, :, :], 2)
    for r in tl.range(0, c):
        dg = tl.sum(tl.where((r8 == r)[:, None], accg, 0.0), 0)
        du = tl.sum(tl.where((r8 == r)[:, None], accu, 0.0), 0)
        inter = dg * tl.sigmoid(dg) * du
        ent = tl.sum(tl.where(r8 == r, ents, 0))
        tl.store(out_ptr + ent * I + j, inter.to(out_ptr.dtype.element_ty), mask=jm)


@triton.jit
def _gv2(in_ptr, list_ptr, counts_ptr, wt_ptr, sft_ptr, gs_ptr, out_ptr,
         start, stride_l,
         H: tl.constexpr, I: tl.constexpr, RP: tl.constexpr, KC: tl.constexpr,
         RPAD: tl.constexpr):
    """Dot-free GEMV, row-major words: y[ent, r] = inter @ dequant(w2[e])^T.

    wt: w2 viewed [E, H, I/8] int32; sft: [E, H, I/16] fp8."""
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e)
    if c == 0:
        return
    gid64 = (start + e).to(tl.int64)
    gs = tl.load(gs_ptr + gid64)
    wbase = gid64 * H * (I // 8)
    sbase = gid64 * H * (I // 16)
    r = tl.program_id(1) * RP + tl.arange(0, RP)
    rm = r < H
    r8 = tl.arange(0, RPAD)
    ents = tl.load(list_ptr + e * stride_l + r8, mask=r8 < c, other=0)
    acc = tl.zeros((RPAD, RP), tl.float32)
    for k0 in tl.range(0, I, KC):
        kw = k0 // 8 + tl.arange(0, KC // 8)
        kb = k0 // 2 + tl.arange(0, KC // 2)
        kb8 = k0 // 16 + tl.arange(0, KC // 16)
        aev = tl.load(in_ptr + ents[:, None] * I + (2 * kb)[None, :],
                      mask=(r8 < c)[:, None], other=0.0).to(tl.float32)
        aod = tl.load(in_ptr + ents[:, None] * I + (2 * kb + 1)[None, :],
                      mask=(r8 < c)[:, None], other=0.0).to(tl.float32)
        bw = tl.load(wt_ptr + wbase + r[:, None] * (I // 8) + kw[None, :],
                     mask=rm[:, None], other=0)
        b = tl.reshape(_word_bytes(bw), (RP, KC // 2))
        sw = tl.load(sft_ptr + sbase + r[:, None] * (I // 16) + kb8[None, :],
                     mask=rm[:, None], other=0.0)
        sc = tl.reshape(tl.broadcast_to(sw[:, :, None], (RP, KC // 16, 8)),
                        (RP, KC // 2)).to(tl.float32) * gs
        w_lo = (_e2m1(b & 0xF) * sc).to(tl.float32)
        w_hi = (_e2m1(b >> 4) * sc).to(tl.float32)
        acc += tl.sum(aev[:, None, :] * w_lo[None, :, :], 2)
        acc += tl.sum(aod[:, None, :] * w_hi[None, :, :], 2)
    for ri in tl.range(0, c):
        a = tl.sum(tl.where((r8 == ri)[:, None], acc, 0.0), 0)
        ent = tl.sum(tl.where(r8 == ri, ents, 0))
        tl.store(out_ptr + ent * H + r, a.to(out_ptr.dtype.element_ty), mask=rm)


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
    """Repack (same bytes): view packed rows as int32 words (4 bytes each) so tile loads
    vectorize; the row-major layout keeps 2048B rows contiguous for DRAM. Scales keep
    their original [rows, K/16] fp8 layout."""
    return {
        "w13T": weights["w13"].view(torch.int32),             # [E, 2I, H/8]
        "w13_sfT": weights["w13_sf"],                          # [E, 2I, H/16] fp8
        "w13_gs": weights["w13_gs"],
        "w2T": weights["w2"].view(torch.int32),               # [E, H, I/8]
        "w2_sfT": weights["w2_sf"],                            # [E, H, I/16] fp8
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

    if M <= 8:
        RPAD = min(max(triton.next_power_of_2(M), 1), 8)
        _gv1[(count, I // 16)](x, list_ws, counts, state["w13T"], state["w13_sfT"],
                               state["w13_gs"], inter, M8, start, M8,
                               H=H, I=I, RP=16, KC=128, RPAD=RPAD, num_warps=4, num_stages=2)
        _gv2[(count, H // 16)](inter, list_ws, counts, state["w2T"], state["w2_sfT"],
                               state["w2_gs"], y, start, M8,
                               H=H, I=I, RP=16, KC=128, RPAD=RPAD, num_warps=4, num_stages=2)
    else:
        _k1[(count, triton.cdiv(I, 32))](x, list_ws, counts, state["w13T"], state["w13_sfT"],
                                         state["w13_gs"], inter, M8, start, M8,
                                         H=H, I=I, BR=32, KC=64, CP=CP, num_warps=8, num_stages=2)
        _k2[(count, triton.cdiv(H, 32))](inter, list_ws, counts, state["w2T"], state["w2_sfT"],
                                         state["w2_gs"], y, start, M8,
                                         H=H, I=I, BR=32, KC=64, CP=CP, num_warps=8, num_stages=2)
    _k3[(M,)](y, ids, gwt, state["post_norm_w"], out, start, count,
              H=H, EPS=POST_NORM_EPS, num_warps=8)
    return out
