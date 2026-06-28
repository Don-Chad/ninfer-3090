# GDN Port — Migration Design (`~/chunked_gdn` → qus L1 `gated_delta_rule`)

> Status: design (brainstorm). Date: 2026-06-26.
> Scope: Tier-2 of [`l1-implementation-roadmap.md`](l1-implementation-roadmap.md) — port the proven
> **chunked** (prefill) and **AR** (decode) Gated-DeltaNet kernels from `~/chunked_gdn` into qus as the
> `gated_delta_rule_chunked` / `gated_delta_rule_recurrent` ops
> ([`l1-operator-catalog.md`](l1-operator-catalog.md) §3.7), localized to the project's conventions
> (`Tensor`/`GdnState`, the api→wrapper→launcher→kernel layout, the frozen test/bench framework).
>
> Source: `~/chunked_gdn` (`chunked/`, `ar/`, `include/gdn_common.h`, `reference/`, `bench/`).
> Targets: [`l1-kernel-layering.md`](l1-kernel-layering.md), [`l1-op-test-standard.md`](l1-op-test-standard.md),
> math in [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) §6.5/§7.
> This is a **port + localization**, not a rewrite: the recurrence math is identical.

---

## 1. Source vs target

**Source (`~/chunked_gdn`, all fp32):**
- `ar_gdn::launch(config)` — AR recurrence (decode); one head/warp, `SxS` state in registers.
- `chunked_gdn::launch(config)` — chunked (prefill); 3 stages (`prepare_wy_wu` → `state_passing` →
  `chunk_output`) + a `workspace`. Full `kChunkSize=64` chunks only; tail goes through AR.
- `gdn::compute_sizes` / `gdn::head_map` (shared layout + head mapping).
- `cpu_ref::gdn_forward` (AR, **fp64 internal** — the numerical truth for both paths) and
  `cpu_chunked_ref::gdn_forward_chunked` (chunked, fp64) — the golden references.
- `bench/bench_common.h` — cudaEvent/median bench harness.

**Target (catalog §3.7):**
```cpp
void gated_delta_rule_recurrent(const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& g, const Tensor& beta, float scale,
                                Tensor& ssm_state, Tensor& out, cudaStream_t stream);   // decode
void gated_delta_rule_chunked  (const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& g, const Tensor& beta, float scale, int chunk_size,
                                WorkspaceArena& ws, Tensor& ssm_state, Tensor& out, cudaStream_t stream); // prefill
```

**Not part of this port** (already Tier-1): `causal_conv1d`, `gdn_gating`, `l2norm`. The op receives
already-conv'd, already-gated, already-L2-normed `q/k/v/g/beta`; it applies only the recurrence and
the `scale = 1/√dk` on `q`.

Fixed dims (Qwen3.6-27B): `S = dk = dv = 128`, `H_qk = 16`, `H_v = 48` (so `G = H_v/H_qk = 3`),
`B = 1`, `kda = false` always.

---

## 2. The good news: layouts already match

qus `Tensor`s are `ne`-order (`ne[0]` fastest). The source buffers and the qus tensors are the **same
byte layout**, differing only in element dtype:

| Tensor | qus `ne`-order | source layout | match |
|---|---|---|---|
| `q`,`k` | `[128, 16, T]` (S, H_qk, T) | `[B,L,H_qk,S]` (S fastest) | yes |
| `v` | `[128, 48, T]` | `[B,L,H_v,S]` | yes |
| `g`,`beta` | `[48, T]` (H_v, T) | `[B,L,H_v]` | yes |
| `ssm_state` | `[128, 128, 48]` (dk, dv, H_v) | `[B,H_v,S,S]` AR-transposed `buf[v*S+k]` | yes (k fastest, then v, then head) |
| `out` | `[128, 48, T]` | `attn_out [B,L,H_v,S]` | yes |

So no transpose/relayout is needed; the only interface gap is dtype (§3.1). `g`/`beta`/`ssm_state`
are fp32 on both sides → passed through untouched.

---

## 3. Localization gaps (the actual work)

### 3.1 dtype — DECISION: boundary-cast for the port, native bf16 later
Source is fp32 throughout; qus uses **bf16** `q/k/v/out` with **fp32** `g/beta/state`.

- **(A) Boundary-cast [chosen for the port].** Keep the chunked/AR kernels operating in fp32
  internally; in the wrapper/launcher, cast bf16 `q/k/v` → fp32 scratch (from `WorkspaceArena`), run
  the ported fp32 kernels unchanged, cast the fp32 `attn_out` → bf16 `out`. `g/beta/state` need no
  cast. Two tiny cast kernels (`bf16→f32`, `f32→bf16`).
- **(B) Native bf16 I/O [deferred to a perf task].** Modify the kernels' load/store paths to read
  bf16 / write bf16 directly (fp32 compute). Removes the cast scratch + traffic.

Rationale for (A): the chunked kernels are intricate (TF32 MMA, WY/UT solve, fastdiv) — rewriting
their I/O dtype is error-prone, and the whole point of porting `chunked_gdn` is to reuse its proven,
FLA-competitive code. Decode (AR) cast overhead is negligible (one token: ~10K elems). Prefill
(chunked) is compute-bound (the matmuls dominate), so the cast cost is small. This keeps "port
correctly" (Tier-2) separate from "optimize" (later), per the roadmap. The fp32 scratch for prefill
(`q/k/v/out` ≈ a few hundred MB at `T=4096`) lives in the workspace arena. Native bf16 (B) also
changes the kernels' internal **smem tile sizes** (bf16 vs fp32 staging), which forces a careful
re-tune — confirmed reason to keep it a separate perf task, not part of the port.

### 3.2 head mapping — GROUPED (`h_qk = h_v // G`), confirmed via vLLM
The source's `gdn::head_map::qk_head` uses **modulo** (`h_qk = h_v % H_qk`, llama.cpp's GDN op). That
is **WRONG for Qwen3.6**, which uses **grouped** mapping. Confirmed in vLLM
`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py::fix_query_key_value_ordering`: `mixed_qkvz`
is reshaped to `[…, num_k_heads, head_k + head_k + (head_v+head_v)*num_v_heads//num_k_heads]`, i.e.
each k-head's slot holds its `G = num_v_heads/num_k_heads = 3` v-heads **contiguously**, so v-head
`h_v` belongs to k-head `h_v // G`. This matches `qwen3.6-27b-architecture.md` §14.2.

Localize `head_map` accordingly (centralized — kernels + CPU refs all route through it):
- `qk_head(h_v) = h_v / G` (integer divide; `G = H_v / H_qk = 3`). Use `init_fastdiv_values(G)` (the
  source's fastdiv divisor changes from `H_qk` to `G`), and a fastdivide instead of fastmodulo.
- `cta_h_v(cta_h) = cta_h` (**identity**): the source's CTA permutation existed only to recover L2
  reuse under modulo (v-heads sharing a k-head are strided by `H_qk`). Under grouped mapping the
  v-heads sharing a k-head are already contiguous (`h_v = qk*G + j`), so the identity already gives
  adjacent CTAs the same K row — drop the permutation.

The M2 per-layer parity check vs HF/vLLM remains the final arbiter.

### 3.3 mechanical localization
- Namespaces: `gdn` / `ar_gdn` / `chunked_gdn` → `qus::kernels` (+ a `detail` for stage launchers).
- Error macro: `GDN_HOST_CHECK` → `CUDA_CHECK` (`qus/core/device.h`).
- `config` structs → the catalog op signatures; `B` is hardcoded `1`.
- `scale`: the source hardcodes `1/√S` inside `chunk_output`/AR; replace with the op's `scale` param
  (assert `scale == 1/√dk` for v1).
- `workspace`: `chunked_gdn::workspace_bytes(...)` → bump-allocate from the passed `WorkspaceArena`.
- Device helpers: port the needed bits of `~/chunked_gdn/include/cuda_utils.cuh` (fastdiv/`fastmodulo`,
  warp-shuffle, TF32 MMA helpers) into a private `src/kernels/kernel/gdn_common.cuh`.
- KDA: `kda` is always `false` for Qwen3.6 — drop the KDA code paths (or keep them dead behind
  `kda=false`); do not port KDA stages.

### 3.4 chunked tail handling
`chunked_gdn::launch` processes only full 64-token chunks. The qus `gated_delta_rule_chunked` wrapper
must, for a prefill of `T` tokens: run the chunked pipeline on the first `floor(T/64)*64` tokens, then
run the **ported AR kernel** on the `T % 64` tail using the chunked end-state — a single in-place
state hand-off (the source's documented split-dispatch). So the AR kernel is shared by both ops.

---

## 4. Target file layout (4-layer mapping)

```
include/qus/kernels/gated_delta_rule.h     # api: ..._recurrent + ..._chunked (catalog signatures)
src/kernels/wrapper/gated_delta_rule.cpp   # validate; build cast scratch + chunked workspace from ws;
                                           #   map Tensor/GdnState -> ported launchers; chunked tail split
src/kernels/launcher/gated_delta_rule.h    # detail:: prototypes (recurrent, chunked-3-stages, casts)
src/kernels/launcher/gated_delta_rule_recurrent.cu   # ported ar_gdn launch
src/kernels/launcher/gated_delta_rule_chunked.cu     # ported chunked pipeline (3 stage launches)
src/kernels/kernel/gdn_common.cuh          # ported head_map + cuda_utils bits (fastdiv, shfl, MMA)
src/kernels/kernel/gated_delta_rule_recurrent.cuh    # ported AR kernel
src/kernels/kernel/gdn_prepare_wy_wu.cuh             # ported stage 1
src/kernels/kernel/gdn_state_passing.cuh             # ported stage 2
src/kernels/kernel/gdn_chunk_output.cuh              # ported stage 3
src/kernels/kernel/gdn_cast.cuh            # bf16<->f32 cast kernels (boundary-cast)
```

The wrapper validates (`q/k/v` BF16 with the fixed dims; `g/beta` FP32 `[48,T]`; `ssm_state` FP32
`[128,128,48]`; `out` BF16; `scale>0`; chunked: `chunk_size==64`) then dispatches by entry
(recurrent vs chunked). `ssm_state` is the per-GDN-layer `GdnState::ssm[idx]` (AR-transposed);
**confirm `GdnState` constructs ssm as `[dk,dv,H_v]`** to match §2 (a small constructor check, not a
redesign).

---

## 5. Test infrastructure (frozen standard + GDN presets)

- **Golden:** port `cpu_ref::gdn_forward` into `tests/kernels/gdn_ref.h` as an **fp64** AR recurrence
  that writes `double` `attn_out` + `state_out` (the truth for BOTH ops). Port
  `cpu_chunked_ref::gdn_forward_chunked` as a second fp64 path for the cross-check.
- **Inputs:** port `~/chunked_gdn/tests/test_utils.h::make_inputs` honest distributions (q/k
  L2-normalized per head, `g∈[-4,0]`, `beta∈[0.05,0.95]`, `state∈[-0.1,0.1]`) + the stress variant
  (`g∈[-1,-0.05]`); `kda=false`. Round q/k/v to bf16 before the golden (per `op_tester.h`).
- **`tests/kernels/test_gated_delta_rule.cpp`:**
  - recurrent: kernel (bf16 `out`, fp32 `state`) vs the fp64 AR golden.
  - chunked: kernel vs the fp64 AR golden, AND vs the recurrent kernel (cross-codepath), AND the
    **state-passing chain-equivalence** check (chunked over `T` == AR stepped `T` times).
  - shapes: decode `T=1`; prefill `T ∈ {64,128,256,4096}` (multiples of 64) + a non-multiple (e.g.
    `T=200`) to exercise the tail split; always `S=128,H_qk=16,H_v=48`. `compute-sanitizer` clean.
- **New presets (framework EXTENSION, not a weakening of existing ones):** add to `op_check.h`
  `gdn_output_bf16` (the bf16 `out`) and `gdn_state_fp32` (the fp32 `state`), grounded in
  `chunked_gdn`'s measured `cross_codepath` noise (`atol 5e-4, rtol 5e-3, tail_frac 2e-2,
  worst_ratio_max 5, rel_l2 5e-3`) — adding a named preset for a new op class is allowed; modifying an
  existing preset is not.

---

## 6. Bench infrastructure

- `bench/gated_delta_rule_bench.cu` on the **real** shapes: decode AR `[S128,Hqk16,Hv48,L1,B1]`;
  prefill chunked `[…,L4096]` (and the `--sweep` L curve, ported from `bench_chunked`).
- **Perf criterion (different from Tier-1):** GDN is **not** purely memory-bound — the chunked path is
  matmul-heavy (TF32 MMA). So the gate is **no regression vs the source**: the ported kernel's
  median µs must be within a small margin of `~/chunked_gdn`'s `bench_chunked`/`bench_ar` at the same
  shape, plus an `ncu` profile recorded. (Do **not** apply the 85%-DRAM gate here; that is for the
  memory-bound Tier-1 ops.) Profile per `l1-op-test-standard.md` §2.4 +
  `~/.cursor/skills/profile-cuda/SKILL.md`.

---

## 7. Migration steps (ordered, for Codex)

1. **Apply grouped head mapping** (§3.2): set `qk_head(h_v) = h_v / G` (fastdiv by `G`) and
   `cta_h_v = identity`. (Grouped is confirmed via vLLM; M2 parity is the final check.)
2. Port `gdn_common.cuh` (head_map + needed `cuda_utils` device helpers) into `src/kernels/kernel/`.
3. Port the **AR** kernel + `gated_delta_rule_recurrent` (launcher + wrapper entry) with boundary casts.
   Port `cpu_ref` → fp64 golden + `make_inputs`; add the GDN presets; write the recurrent test. Build,
   test, sanitizer. Commit.
4. Port the **chunked** 3 stages + `gated_delta_rule_chunked` (3-stage launcher + workspace + tail-AR
   split). Port `cpu_chunked_ref` for the cross-check; extend the test (chunked vs golden + vs
   recurrent + chain-equivalence + tail). Build, test, sanitizer. Commit.
5. **Perf:** `bench/gated_delta_rule_bench.cu`; profile with ncu; confirm no regression vs the source
   bench; record numbers. Commit.

Each step = a Codex task; use the subagent prompt templates + workflow in
[`docs/plans/l1-tier1-simple-ops.md`](plans/l1-tier1-simple-ops.md) (correctness and perf as
separate tasks; frozen framework read-only; commit per task).

---

## 8. Verifications & risks (flagged)
- **Head mapping (§3.2)** — RESOLVED: grouped (`h_v / G`), confirmed via vLLM `fix_query_key_value_ordering`; the source's modulo must be replaced. M2 per-layer parity is the final check.
- **`GdnState::ssm` layout** = `[dk,dv,H_v]` AR-transposed — confirm the constructor matches §2.
- **TF32 in the chunked KKT/MMA** — bf16 inputs cast to fp32 then TF32-rounded in the matmuls; the
  fp64 golden + the `gdn_*` presets bound this. If chunked drifts past the preset, that is a real
  precision finding (escalate), not a tolerance to loosen.
- **`chunk_size`** — v1 supports `64` only (assert); the source pins `kChunkSize=64`.

## 9. Out of scope
- KDA (Qwen3.6 is GDN scalar-gate; `kda=false`).
- `causal_conv1d` / `gdn_gating` / `l2norm` (Tier-1, done).
- Native bf16 kernel I/O (§3.1 option B) — a later performance task.
- Batch > 1 (`B=1`).

## 10. Sources
- `~/chunked_gdn`: `include/{gdn_common.h,cuda_utils.cuh}`, `chunked/{chunked_gdn.cuh,*.cu}`,
  `ar/ar_gdn.{cuh,cu}`, `reference/{cpu_ref,cpu_chunked_ref}.{h,cpp}`, `tests/test_utils.h`,
  `bench/bench_common.h`, `ALGORITHM.md`.
- vLLM (head-mapping confirmation): `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
  (`fix_query_key_value_ordering`, the `[num_k_heads, … v-heads-per-k-head]` reshape).
- qus: `l1-operator-catalog.md` §3.7, `l1-kernel-layering.md`, `l1-op-test-standard.md`,
  `include/qus/core/state_store.h`, `qwen3.6-27b-architecture.md` §6.5/§7/§14.
