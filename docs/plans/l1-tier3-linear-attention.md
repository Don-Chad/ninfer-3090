# L1 Tier-3 — `linear` (W4A16 GEMM/GEMV) + GQA attention — Plan (Codex-self-contained)

> Self-contained for Codex. Reuses the execution machinery in
> [`docs/plans/l1-tier1-simple-ops.md`](l1-tier1-simple-ops.md) (read its "How to execute", "Subagent
> prompt templates", "ncu procedure") and the frozen framework
> ([`docs/l1-op-test-standard.md`](../l1-op-test-standard.md),
> [`tests/kernels/op_check.h`](../../tests/kernels/op_check.h),
> [`tests/kernels/op_tester.h`](../../tests/kernels/op_tester.h),
> [`bench/qus_bench_common.h`](../../bench/qus_bench_common.h)). Op contracts:
> [`docs/l1-operator-catalog.md`](../l1-operator-catalog.md) §3.1 (`linear`) + §3.6 (`gqa_attention`).

**Goal:** Implement the two remaining L1 ops — `linear` (all text qtypes + dense, GEMV decode + GEMM
prefill) and `gqa_attention` (prefill + decode) — **correct and API-complete**, with the test/bench
framework and the fixed model dimensions, so the full L2 pipeline can be assembled. **Performance is
not tuned here** (see Performance policy).

**Architecture:** Four-layer L1 layout per `l1-kernel-layering.md`. `linear` consumes the `Weight`
seam (catalog §3.1); bf16 activations, fp32 accumulate. `gqa_attention` consumes `Tensor` + `KVCache`.

**Tech Stack:** C++20, CUDA 13.1 (sm_120), CMake ≥ 3.28, gcc 13.3, build dir `build/`.

---

## Performance policy (READ FIRST — this differs from Tier-1/2)
- **Optimization is NOT a task and NOT gated.** Each kernel must be **correct** and implement the
  **full API**; write it as efficiently as is *reasonable without deep tuning* (sensible vectorization
  / coalescing / grid-stride — do not write deliberately wasteful code, but do not chase the roofline).
- A bench binary IS built per op (real model shapes) so we have a baseline number and the framework is
  in place — but there is **no perf acceptance %** and no profiling gate in these tasks.
- **Where heavy optimization will later be needed (flagged, not done now):** the **W4A16 decode GEMV**
  is the decode weight-bandwidth roofline (`design.md` §1, the headline metric) and **flash-attention
  prefill** — both get a dedicated Tier-3-perf milestone after the pipeline runs end-to-end. Note this
  in the kernel files with a `// PERF:` comment pointing here.

## Hard rules
- Frozen framework read-only (you MAY *add* the presets below; never weaken existing ones). No math
  approximation. **FORMAT before every commit:** `clang-format -i <new/changed .h/.cuh/.cpp/.cu>` then
  `clang-format --dry-run --Werror <…>` (must exit 0; the repo `.clang-format` is the style). Work on
  `master`, one commit per task.

---

## Design & conventions

### `linear` (the Weight seam)
`void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);`
Wrapper validates, then `switch(w.qtype)`; decode `T==1` → GEMV path, prefill `T>1` → GEMM path
(infer `T` from `x`). bf16 `x`/`out`, **fp32 accumulate**. v1 text qtypes: `Q4G64_F16S`,
`Q5G64_F16S`, `Q6G64_F16S` (all `TILE_N64_K64`), and dense `BF16_CTRL`/`FP32_CTRL` (`as_dense`).
`W8G128` is MTP/vision-only → leave a `default:` that throws "unsupported qtype" (deferred).

### q5090 `TILE_N64_K64` layout (authoritative source — mirror it, do not invent)
`tools/q5090_convert/{qtypes.py, packing.py, layouts.py}` define the ABI; `layouts.py::decode_tile_lowbit`
is the exact inverse the GEMV + the CPU ref must implement. Summary for an `[N,K]` weight (here all
text `N,K` are already multiples of 64, no padding):
- `nt = N/64` tiles, `kg = K/64` groups; per `(tile, group)` the blob is `64` fp16 **scales**
  (one per row of the 64-row tile) followed by `64 × bpr` packed code bytes, where
  `bpr = group_size*bits/8` (Q4→32, Q5→40, Q6→48). Codes are **symmetric signed** two's-complement,
  LSB-first bit-packed (`packing.py`); `w[n,k] = code * scale` (no zero point). `Weight` carries
  `qdata`/`scales`/`n`/`k`/`group`/`qtype`/`layout` (see `include/qus/core/tensor.h`).
- The Tier-1 `embed_gather` already handles the `ROW_GROUPED_G64` layout; `linear` only needs
  `TILE_N64_K64` + `CONTIGUOUS`.

### `gqa_attention` (catalog §3.6)
`gqa_attention_prefill(q,k,v, scale, kv, layer, out, stream)` and
`gqa_attention_decode(q,k,v, pos, scale, kv, layer, out, stream)`. `q=[256,24,T]`, `k/v=[256,4,T]`,
`out=[256,24,T]`, bf16; `scale = 1/√256`; **causal**; GQA `kv_head = q_head / (n_q/n_kv) = q_head/6`.
fp32 softmax/accumulate. RoPE is already applied (Tier-1 `rope`) and the output gate is applied after
(Tier-1 `sigmoid_gate_mul`) — attention does NOT do them. The **KV write is folded in**: prefill
writes all `T` k/v into `kv[layer]`; decode appends the new k/v at `pos` then attends `[0..pos]`.
`KVCache` layout: `include/qus/core/kv_cache.h`.

---

## Fixed dimensions

**`linear` — every text use `(N, K, qtype)`** (from `tools/q5090_convert/tensor_plan.py`; `K∈{5120,6144,17408}`):
- Q4G64: `q_proj (6144,5120)`, `k_proj (1024,5120)`, `in_q (2048,5120)`, `in_k (2048,5120)`,
  `mlp_gate (17408,5120)`, `mlp_up (17408,5120)`
- Q5G64: `gate_proj (6144,5120)`, `v_proj (1024,5120)`, `o_proj (5120,6144)`, `in_v (6144,5120)`,
  `in_z (6144,5120)`, `out_proj (5120,6144)`, `mlp_down (5120,17408)`
- Q6G64: `lm_head (248320,5120)`
- dense BF16: `in_a (48,5120)`, `in_b (48,5120)`
- `T`: decode `1` (GEMV), prefill `>1` (GEMM).

**`gqa_attention`** (the 16 full-attn layers): `head_dim=256`, `n_q=24`, `n_kv=4`, `scale=1/√256`,
causal, bf16 KV, `max_context=128K`. Bench positions: decode `pos∈{2048, 32768}`; prefill `T∈{128, 2048}`.

---

## Test & bench framework

- **`linear` correctness:** CPU ref = **fp64** `dequant(Weight) @ x` (mirror `decode_tile_lowbit` /
  `as_dense`). Provide a host helper `tests/kernels/q5090_pack.h` that builds a `Weight` in
  `TILE_N64_K64` from an fp32 matrix (port `quantize_core` + `pack_lowbit_groups` + `encode_tile_lowbit`
  from `tools/q5090_convert/`), so tests can synthesize Q4/Q5/Q6 weights. Round `x` to bf16 first.
- **`gqa_attention` correctness:** CPU ref = **fp64** naive causal GQA: `s_ij = (q_i·k_j)*scale` for
  `j≤i`, softmax over `j`, `o_i = Σ_j p_ij v_j`, with `kv_head=q_head/6`. Drive `KVCache` directly
  (decode: pre-fill `pos` slots + the new token; prefill: empty cache, fill `T`).
- **New presets (add to `op_check.h`; additive):**
  ```cpp
  static constexpr Tolerance linear_bf16()    { return {2e-3, 1.6e-2, 2e-3, 5.0, 8e-3}; } // GEMV/GEMM reduce over K
  static constexpr Tolerance attention_bf16() { return {2e-3, 1.6e-2, 2e-3, 5.0, 8e-3}; } // softmax-weighted sum
  ```
- **Bench (built, not gated):** `bench/linear_bench.cu` (decode GEMV + prefill GEMM at the real
  `(N,K)`), `bench/gqa_attention_bench.cu` (decode at the bench positions + prefill `T`). Report GB/s
  via `qus_bench_common.h` for reference only.

---

## Tasks (correctness-only; each ends with build + test + sanitizer + bench-runs + clang-format + commit)

### Task L-1 — `linear` scaffolding + dense path
- **Reading:** catalog §3.1, design above, `weight-handle-design.md`, `silu_and_mul.*` template, `op_tester.h`.
- **Files:** `include/qus/kernels/linear.h`; `wrapper/linear.cpp` (validate; `switch(qtype)` with dense
  arm via `as_dense`; GEMV/GEMM split by `T`; `default:` throw for W8/unknown); `launcher/linear.h` +
  `linear_dense.cu`; `kernel/linear_dense.cuh` (naive fp32-accumulate dense GEMV+GEMM). Test
  `tests/kernels/test_linear.cpp` (dense BF16/FP32, e.g. `(48,5120)`, `(5120,6144)`; `T∈{1,7}`) using
  `linear_bf16`. `bench/linear_bench.cu` scaffolding (+ `qus_add_bench`). Add the two presets.
- **DoD:** dense test PASS, sanitizer clean, bench builds/runs, clang-format clean. Commit
  `feat(kernels): linear scaffolding + dense path (correctness)`.

### Task L-2 — `linear` Q4G64 (TILE_N64_K64, GEMV + GEMM)
- **Reading:** the q5090 layout section; `tools/q5090_convert/{qtypes.py,packing.py,layouts.py}`; L-1 files.
- **Files:** `kernel/linear_q4.cuh` (decode `TILE_N64_K64` Q4 + fp32-accumulate GEMV(`T=1`)/GEMM(`T>1`));
  `launcher/linear_q4.cu`; wire `case QType::Q4G64_F16S` in the wrapper. `tests/kernels/q5090_pack.h`
  helper. Extend `test_linear.cpp` with Q4 shapes (`(6144,5120)`,`(1024,5120)`,`(2048,5120)`,
  `(17408,5120)`; `T∈{1,2,7,64}`) vs the fp64 dequant ref. Extend `linear_bench.cu` (Q4 decode `(17408,5120)` GEMV).
- **DoD:** Q4 tests PASS (`linear_bf16`), sanitizer clean, bench runs, format clean. Add a `// PERF:` note
  in `linear_q4.cuh` (decode GEMV = the roofline op, tuned later). Commit `feat(kernels): linear Q4G64 GEMV+GEMM (correctness)`.

### Task L-3 — `linear` Q5G64 + Q6G64 (incl. lm_head)
- **Reading:** L-2; the layout section (Q5=40B, Q6=48B per group).
- **Files:** `kernel/linear_q5.cuh`, `kernel/linear_q6.cuh` (+ launchers; wire the two `case`s). Extend the
  packer helper for Q5/Q6. Extend `test_linear.cpp`: Q5 (`(6144,5120)`,`(5120,6144)`,`(5120,17408)`),
  Q6 lm_head at a reduced `(4096,5120)` for the golden + `T∈{1,7}`. Extend `linear_bench.cu` with the
  **full** lm_head `(248320,5120)` decode GEMV (bench only, no golden).
- **DoD:** Q5/Q6 tests PASS, sanitizer clean, bench runs (incl. full lm_head), format clean. Commit
  `feat(kernels): linear Q5G64 + Q6G64 (correctness)`.

### Task A-1 — `gqa_attention_decode` (+ KV append)
- **Reading:** catalog §3.6, architecture §6.4/§10.3, `kv_cache.h`, design above, `op_tester.h`.
- **Files:** `include/qus/kernels/gqa_attention.h` (both decls); `wrapper/gqa_attention.cpp` (validate +
  phase dispatch); `launcher/gqa_attention.h` + `gqa_attention_decode.cu`; `kernel/gqa_attention_decode.cuh`
  (append new k/v at `pos`, single-query attend `[0..pos]`, fp32 softmax, GQA `q/6`). Test
  `tests/kernels/test_gqa_attention.cpp` (decode: pre-fill `pos∈{1,17,2048}` KV slots + new token; vs
  fp64 naive ref; `attention_bf16`). `bench/gqa_attention_bench.cu` (decode `pos∈{2048,32768}`).
- **DoD:** decode test PASS, sanitizer clean, bench runs, format clean. Commit
  `feat(kernels): gqa_attention_decode (correctness)`.

### Task A-2 — `gqa_attention_prefill` (causal, fills KV)
- **Reading:** A-1; architecture §6.4 (flash/causal).
- **Files:** `launcher/gqa_attention_prefill.cu`; `kernel/gqa_attention_prefill.cuh` (causal GQA over the
  prompt; write all `T` k/v into `kv[layer]`; fp32 online or naive softmax — correctness first). Extend
  `test_gqa_attention.cpp`: prefill `T∈{1,7,128,512}` vs the fp64 ref; **prefill→decode consistency**
  (prefill `T`, then one decode step at `pos=T` must match a naive `T+1` reference). Extend the bench
  (prefill `T∈{128,2048}`).
- **DoD:** prefill + consistency tests PASS, sanitizer clean, bench runs, format clean. Add a `// PERF:`
  note (flash optimization later). Commit `feat(kernels): gqa_attention_prefill (correctness)`.

---

## Coverage map (proves the pipeline API is complete)
Every L2 call is now an implemented op: embed (`embed_gather`, T1) · norms (`rmsnorm`, T1) · **all
projections + lm_head (`linear`, this plan)** · rope (`rope`, T1) · **full attention
(`gqa_attention_*`, this plan)** · attn gate (`sigmoid_gate_mul`, T1) · residual (`residual_add`, T1) ·
SwiGLU (`silu_and_mul`, T1) · GDN conv/gate/l2norm (T1) · GDN recurrence (`gated_delta_rule_*`, T2) ·
final argmax (`argmax`, T1). After Tier-3, the L2 card can be wired end-to-end (M2).

## Done criteria
`linear` (Q4/Q5/Q6 + dense, GEMV + GEMM) and `gqa_attention` (prefill + decode) pass the frozen-framework
correctness tests (fp64 goldens + the new presets), `compute-sanitizer` clean, all code clang-format
clean, and each has a building/running bench (numbers recorded, not gated). W8G128 is the only deferred
qtype (MTP/vision). Performance tuning of the decode GEMV + flash prefill is a separate later milestone.

## Self-review notes (author)
- No separate perf task (per the user); a single Performance policy section + `// PERF:` markers instead.
- All model `(N,K,qtype)` uses + the attention dims are fixed above; the coverage map shows the full L2
  API is satisfied after this tier.
- New presets are additive; q5090 decode mirrors the authoritative converter code (no re-derivation).
- clang-format is in the Hard rules and every task DoD.
