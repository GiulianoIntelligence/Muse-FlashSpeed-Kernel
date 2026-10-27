# Muse-FlashSpeed-Kernel

Kernels for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/) —
beat SGLang's kernels on H200 for Muse Spark Open (256-expert MoE, NVFP4 weights).

- One folder per problem (`muse_*`), produced by `okc init <problem> <folder>`.
- `RESULTS.md` is the experiment log — read before each new attempt.
- Workflow: edit `submission.py` → `okc check` (never submit a failing kernel) →
  `okc submit` only if the check clearly beats our best → commit + push.
- Rules that matter: CUDA-graph capturable (no `.item()`, `.cpu()`, sync, value-dependent
  control flow), precision preserved (NVFP4 stays NVFP4, BF16 stays BF16, FP8 KV cache),
  no timer gaming, output caching, or lower precision.
