# Prefill Tensor-Core GEMM (P2) — implementation plan

> Status: partially executed 2026-07-01. Design: [`../2026-07-01-prefill-linear-foundation-design.md`](../2026-07-01-prefill-linear-foundation-design.md) §7.

## Goal

Add a tensor-core GEMM for the LargeT regime so prefill linear (nsys: ~97% of prefill, ~4-6% of the
~220 TFLOP/s ceiling) can approach the tensor-core roofline. `out[N,T] = W[N,K] . x[K,T]`, W is
Q4/Q5/Q6 row-split, x/out bf16.

## Non-goals

New weight layout / repack / second copy; CUTLASS; FP8/FP4 activations; epilogue fusion; changes to
decode or the SmallT multi-step path.

## Execution mode

Single-agent, direct sequential; strict review for this high-risk CUDA/numerical kernel.

## Status

- **Done (correct + validated).**
  - `src/kernels/linear/gemm/linear_rowsplit_gemm_mma.{cuh,cu}`: generic (Q4/Q5/Q6) bf16
    `mma.sync.aligned.m16n8k16` GEMM with on-chip low-bit dequant (identical `Codec::load_pair`
    math), fp32 accumulate, N/T/K tails.
  - Seam: `RowsplitLowbitGemmMma` policy (`uses_tensor_cores=true`); `resolve_plan` routes low-bit
    LargeT (`T>tau`, `tau=64`) to it; SmallT stays multi-step; decode/dense unchanged.
  - Numerics methodology: `Tolerance::linear_tc` normwise (`rel_l2<=4e-3`) parity preset for the TC
    path, documented in [`../l1-op-test-standard.md`](../l1-op-test-standard.md) §1.3. Full
    `qus_linear_test` prefill matrix green; `compute-sanitizer memcheck` clean.
- **Not yet met: the perf bar.** The current kernel is correctness-first (per-element global loads,
  no shared staging / ldmatrix / cp.async), so it is latency-bound (~3.5-5.6% of the TC ceiling) and
  perf-neutral vs the multi-step kernel; e2e 705-tok prefill 3.72 -> 3.74 s (no regression, no gain).

## Remaining work (roofline pushdown)

1. Stage quantized weight planes + x into shared with coalesced/cp.async loads; dequant once per
   K-tile into a shared bf16 tile reused across a large token tile; feed mma via `ldmatrix`
   (`gdn_common.cuh` helpers).
2. ncu-tune tile shapes / pipeline depth / swizzle toward `>=50%` of the measured TC ceiling on the
   dominant shapes (Q4 gate/up 17408x5120, Q5 proj/out/down); recalibrate `tau` from the T-swept
   bench; refresh `profiles/prefill-linear-foundation/baseline_p2.csv`; re-measure e2e prefill.
3. Strict review of fragment/ldmatrix correctness, dequant-into-shared, tails, shared/cp.async
   lifetime.

## Verification commands

```
cmake --build build -j
./build/tests/qus_linear_test
compute-sanitizer --tool memcheck ./build/tests/qus_linear_test
./build/bench/qus_linear_op_bench --t-sweep 1,8,64,128,512,2048 --csv-out profiles/prefill-linear-foundation/baseline_p2.csv
```
