# M3 Readiness

> Status: M2.8 gate complete; M3 headline per-kernel optimization may be planned from this baseline.
> Scope: official M3 entry evidence package for qwen3.6-ultraspeed.

## Official Baseline

- Raw report path: `profiles/e2e/m2.8-m3-gate-block-scoped.json`
- Raw report SHA256: `c0222d0b409fb976c0adeb4f2a1b88948fbf6955ca8921cb37bb72ad85cb5a31`
- Generation command: `./build/bench/qus_e2e_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus --output-json profiles/e2e/m2.8-m3-gate-block-scoped.json --case cn_short:bench/fixtures/prompts/cn_short.ids:128 --case long_2k:bench/fixtures/prompts/long_2k.ids:1 --warmup-repeats 1 --repeats 3 --max-ctx 8192`
- Committed summary path: `docs/bench/baselines/m2.8-m3-gate-block-scoped-summary.json`
- Git commit: `d64fada1ac5c6d5cbcb616144e4e612a2283baad`
- Worktree dirty: `False`
- q5090 path: `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus`
- q5090 file size bytes: `17104116736`
- q5090 SHA256: `5a4cf3485a12d307586c2dc34e44d244079fb28a9b0331ee97c85ed9fc9c36e7`
- Workspace lifetime policy: `block_scoped_mixer_mlp_rewind`
- Memory accounting scope: `engine_owned_device_arenas_complete`
- Hidden device allocations: `False`
- Decode metric: `decode_eager_tok_s`
- Token readback: `per_step_sync_d2h`

## Required Cases

### `cn_short`

- Prompt tokens: `49`
- Requested max new tokens: `128`
- Warmup repeats: `1`
- Measured repeats: `3`
- Deterministic token ids: `True`
- Prefill median seconds: `1.29595`
- Decode median seconds: `28.6874`
- Decode eager tok/s median: `4.42704`
- Max workspace peak bytes: `9261568`

### `long_2k`

- Prompt tokens: `7539`
- Requested max new tokens: `1`
- Warmup repeats: `1`
- Measured repeats: `3`
- Deterministic token ids: `True`
- Prefill median seconds: `198.714`
- Decode median seconds: `0`
- Decode eager tok/s median: `None`
- Max workspace peak bytes: `2346435072`

## First-Wave M3 Target Order

This ordering is an implementation starting point, not a performance claim. Any future claim must cite
both local profiler evidence and before/after e2e report artifacts.

1. **Linear backend: quantized Q4/Q5/Q6 GEMV/GEMM plus BF16 dense projections**
   Dominates attention, GDN, and MLP projection work across every layer.
2. **Gated DeltaNet recurrent/chunked path**
   Covers the high-frequency GDN mixer path and its stateful prefill/decode behavior.
3. **GQA attention prefill/decode kernels**
   Full-attention layers are less frequent but expensive, especially at long context.
4. **MLP elementwise and normalization tail ops**
   RMSNorm, L2Norm, SiLU/gate, sigmoid gate, and residual ops are shared across blocks.
5. **Embedding, argmax, and host-visible token readback plumbing**
   Keep eager decode semantics measurable after larger kernel work is underway.

## Gate Interpretation

The M2.8 gate can be evaluated from this file, the committed summary, and the local raw report SHA.
Raw reports and decoded sidecars remain local under `profiles/e2e/` and are intentionally not committed.
