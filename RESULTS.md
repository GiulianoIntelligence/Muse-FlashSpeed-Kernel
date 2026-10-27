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
| moe_w4a16 | v2: prepare() transposed NVFP4 repack → B-operand loads [K,N] native; even/odd x split (no interleave); dtype-generic kernels validated exact in TRITON_INTERPRET | PASSED 5/5 (rel L2 4.8e-3, job bda09d374198440f). Per shape µs ours vs sol: 396/11.4, 997/52.5, 1304/70.8, 776/37.3, 2492/101, 2908/109 → 0.188× (k1+k2 ≈ 5% of peak BW; num_stages=1 latency-bound) | 0.188× | kept (first correct kernel); next: num_stages=2 |
