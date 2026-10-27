# RESULTS.md — experiment log

Read this before every new attempt. One row per idea. Never repeat a failed idea;
revisit kept ones for further gains.

Status: kept / reverted (with reason).

Limits per problem per day (reset 00:00 UTC): 10 scored `okc submit`, 20 `okc check`.
Submit only when a check clearly beats our current best.

## Speed-limit estimate (muse_moe_w4a16, done before coding)

- Per (token, expert) pair: w13 16.78 MB + w13_sf 2.10 MB + w2 8.39 MB + w2_sf 1.05 MB
  ≈ 28.3 MB of NVFP4+scale bytes. ~100 MFLOP/pair.
- CORRECTED after starter check job 2c94279cfae14e17 (the "13.45 ms" in okc show is a
  weighted score, not a per-step time). Per-shape reality (µs):

  | shape      | SGLang | SOL floor | SGLang/floor |
  |---|---|---|---|
  | decode 1   | 77.4   | 11.4      | 6.8× |
  | decode 16  | 191.1  | 52.5      | 3.6× |
  | decode 32  | 202.6  | 70.8      | 2.9× |
  | verify 1   | 142.4  | 37.3      | 3.8× |
  | verify 16  | 468.4  | 101.3     | 4.6× |
  | verify 32  | 615.2  | 108.9     | 5.6× |

  Geomean SGLang 222.8 µs, geomean floor 50.9 µs → **max speedup ≈ 4.38×**.
  Leaderboard: leader 2.62× (≈85 µs geomean) of 919 entries. Score shown = 60×mean(ours);
  speedup = geomean(SGLang/ours). User's "0.5 ms" target is below the physical floor in
  these units — real goal: geomean ≤ 85 µs (beat leader), stretch 3-4×.
- Key insight: hot experts are few (floors imply ~2–19 distinct experts touched per shape);
  read only touched experts, expert-major, one pass, dequant in-kernel. Avoid SGLang's
  moe_align + 2× Marlin gemm + separate activation/norm passes.

## Log

| Problem | Idea | Check result | Speedup | Status |
|---|---|---|---|---|
| moe_w4a16 | starter (naive PyTorch, dequant per call) | passed 5/5 tests | 0.006× | baseline sanity ✓ |
| moe_w4a16 | v1: 4-kernel Triton (route prep, GEMM1+SwiGLU, GEMM2, norm+weighted-sum), expert-major, dequant in-kernel, only touched experts | FAILED tests bs=4/33: rel L2 1.413 ≈ √2 → output decorrelated with ref (job 58a260bf05454c17) | — | reverted; root cause: E2M1 code-3 decoded as 2.0 instead of 1.5 (1/16 of weights wrong) + likely tl.trans/interleave lowering; v2 removed trans/interleave entirely |
| moe_w4a16 | v2: prepare() transposed NVFP4 repack → B-operand loads [K,N] native; even/odd x split (no interleave); dtype-generic kernels validated exact in TRITON_INTERPRET | PASSED 5/5 (rel L2 4.8e-3, job bda09d374198440f). Per shape µs ours vs sol: 396/11.4, 997/52.5, 1304/70.8, 776/37.3, 2492/101, 2908/109 → 0.188× (k1+k2 ≈ 5% of peak BW; num_stages=1 latency-bound) | 0.188× | kept — SUBMITTED as first board entry (job 2cf4389f24844cd2, 0.188×) |
| moe_w4a16 | num_stages 1→2 (pipelined k-chunk loads) | passed; AOT showed pipeliner NOT engaging on v2 (identical PTX) | 0.185× | reverted (noise, slightly worse) |
| moe_w4a16 | v3: word-packed u32 weight loads (4× fewer load inst), sf tile broadcast, no trans/interleave; pipeliner engaged (AOT: shared 24KB, PTX changes) | passed; decode1 396→255µs (0.30×) BUT all M≥16 shapes regressed ~1.5× (broadcast_to+reshape goes through smem+barriers ×4/chunk) | 0.154× | reverted (keep ideas) |
| moe_w4a16 | v4: v3 + dot-free GEMV path (mul+sum, no mma layout conv) for M≤8 | passed; decode1 238µs (0.325×, best) but verify1 (M=8) 700→1350µs (RPAD=8 3-D products too heavy) | 0.140× | reverted; gv only viable for M≤2 |
| moe_w4a16 | v5: row-major word layout (DRAM-friendly 2048B rows), weights as dot A-operand | passed; ALL shapes worse (decode1 406µs, verify1 4067µs) | 0.100× | reverted — transposed layout was fine; DRAM-locality hypothesis DEAD |
| moe_w4a16 | v6: v3 + 256-entry LUT dequant (byte→2×bf16 bitcast) + per-element sf gather (no smem reshape) | passed; decode1 289µs, verify32 3712→4229?? no: 3087(v2)→4229 — LUT ≈ neutral overall | 0.155× | reverted (LUT idea kept for later) |

| moe_w4a16 | split-K (S=4) for M<=8 + fused norm in k2 merge kernel | FAILED on judge (bs=4 rel 1.000 = out exactly 0; bs=1 ~1.3-1.5) despite exact interpreter validation AND on-GPU debug showing split-out ≈ full-out (maxdiff 0.004) | — | reverted & PARKED; needs GPU-level debugging (ncu); suspicion: harness capture/replay interaction — out matched the full path yet the check failed; stages=1/warps=4 bisect ruled out the pipeliner |
| moe_w4a16 | sweep 1-4: per-M config table (BR/KC/warps/stages, one combo per scored shape per check) | sweep2 0.213x, sweep3 0.228x, sweep4 pinned winners → SUBMITTED 0.2376x (job ae3ea7bcafd8407e) | 0.238× | KEPT — board best. Winners: M=1 (32,128,8,4); M≤8 (32,128,4,3); M≤16 (64,128,8,3); M≤40 (64,64,8,4); M≤128 (64,128,8,4); M>128 (64,128,8,3) |

## Insights (read before next attempt)

1. All 6 kernel structures land at ~250-700GB/s effective (5-15% of H200 peak). NOT load-width, NOT DRAM layout, NOT dot-vs-mulsum alone.
2. v2→v3 regression isolated the sf `broadcast_to+reshape` (smem round trip + barriers ×4/chunk) as a real cost; v6's per-element L1 gather avoids it.
3. e2m1 where-chain ≈ 12-24 ALU ops/byte is significant; LUT gather ≈ 9 — helps decode, neutral elsewhere.
4. Per-chunk constant cost ~2-4µs/block regardless of size — suspect dot smem-staging barriers (num_warps=8) OR pipelining not covering the register-computed operand path. UNRESOLVED — needs targeted probe.
5. Interpreter (TRITON_INTERPRET=1, fp32 buffers, dtype-generic kernels) + AOT sm90 PTX inspection (triton.compile with GPUTarget, no GPU needed) = free local validation; caught E2M1 code-3 bug and join-order semantics.
6. tl.join(join(a,b),join(c,d)) flattens (a,c,b,d) — nesting must be join(join(a,c),join(b,d)) for (a,b,c,d).
7. Configs: BR=32/KC=128/warps=8/stages=2 current best for dot path; large-M needs the x-re-read + mma count addressed (fused lo+hi dot, CP=128 padding waste).

## Next ideas (priority order)

- Config sweep per shape via shape-branch (BR, KC, warps, stages) — cheap, several combos per check.
- Fuse _k3 into _k2 epilogue (saves ~6µs/call + y_ws round trip — matters at decode1's 11µs floor).
- Split-K across blocks for decode1 (only ~333 active blocks; need 1000+ for latency hiding).
- Env-gated cuda-event timing inside eager test phase to attribute the per-chunk constant cost on real GPU.
- Bigger redesign (day-scale): CUDA load_inline Marlin-style kernel (cp.async + ldmatrix + hand dequant) for the hot path.
