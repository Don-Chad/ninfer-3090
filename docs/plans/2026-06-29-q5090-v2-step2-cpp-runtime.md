# q5090 v2 — Step 2: CPP Runtime on v2 (correct inference, then per-kernel tuning)

> Phase 2 of [docs/q5090_v2_roadmap.md](../q5090_v2_roadmap.md). Spec:
> [q5090_packed_file_format_v2.md](../q5090_packed_file_format_v2.md);
> assignment: [qwen3_6_27b_q5090_v2_tensor_plan.md](../qwen3_6_27b_q5090_v2_tensor_plan.md);
> gates: [q5090_v2_verification_contract.md](../q5090_v2_verification_contract.md). Prereq: Phase 1
> (v2 file + Python ref) is green.

## Goal

Make the cpp runtime **load the v2 layout, infer correctly, then run the decode-critical kernels at
high memory throughput**:

1. **Load** the v2 layout (parser + `Weight`/segment binding).
2. **Adapt the kernels** — GEMV (decode `T=1`) and GEMM (prefill `T>1`) — to `ROW_SPLIT`, correct first.
3. **Prove correct inference** end to end: reproduce the recorded quantized greedy snapshot **exactly**
   (HF is diagnostic only, never a gate — see the verification contract).
4. **Per-kernel tuning** of the `ROW_SPLIT` GEMV/GEMM, profile-driven, to approach the weight-bandwidth
   roofline the v2 layout was designed for.

Stages C→D have a hard gate: **no tuning before correctness is green.**

## Non-goals / hard constraints

- **Framework unchanged.** Keep the linear backend structure (`LinearFormat` / `ShapeFamily` /
  `LinearRegime` / `LinearBackendKind` / `LinearPlan` registry, the `linear()` wrapper dispatch, the
  `WeightStore` API, the model-card per-projection call sites). Stage D adds **tuned plans into the
  existing registry** (as the framework intends); it does not restructure the framework.
- **No fused-projection group GEMV.** Per-segment GEMV is used throughout. Fusing a group into one
  large-`N` GEMV (framework §21.5) is a **bigger lever that changes call structure**; it is explicitly
  **out of scope** here and noted as the next phase.
- **No backward compatibility.** Remove all cpp TILE/v1 layout code; no fallback. This includes the
  `QuantLayout::{TileN64K64, W4A16KernelPackedV1, TileN64K128, RowGroupedG64}` enumerators, the v1
  magic `Q5090MIXEDV1`, the tuned-TILE GEMV, and the inline RowGrouped embed-gather payload path.
- **Do not delete the v1 weight file** (`out/…mixed_v1.qus`) — Phase 3 cutover. cpp now loads **v2**.
- **Correctness is invariant under tuning.** The fp64 oracle and greedy parity must re-pass after every
  Stage-D change; performance claims must be ncu/nsys-backed (contract §6).
- No MTP/Vision runtime, no CUDA Graph, no KV/attention changes.

## Execution mode

Subagent-driven, **sequential** (shared core files). One implementer subagent per task; after each code
task, a spec-compliance reviewer then a code-quality reviewer; fix Critical/Important before
proceeding. Stage D uses an implementer + a profiler subagent per round (like the attention perf plan).
Final independent review after T8 (correctness) and again after T11 (performance).

**Prerequisite — capture the current `ctest` baseline first (coordinator action, before Task 1).**
Phase 1 rewrote the shared `tools/q5090_convert/*` modules to v2 (the layout enum is now
`{ROW_SPLIT=0, CONTIGUOUS=1}`; v1 enums deleted). The cpp fixture-dependent tests regenerate their
fixture at runtime by shelling out to `tests/fixtures/make_q5090_fixture.py`, which still references the
now-deleted v1 layout enums — so those tests are **already red** before Phase 2 begins. Run
`cmake --build build -j && ctest --test-dir build` once and record which targets are red and why.
Expected **already-red at the initial baseline** (fixture shells the now-v2 Python): `qus_q5090_parser_test`,
`qus_model_bind_test`, `qus_model_blocks_test`, `qus_weight_store_test`. Expected **still-green at the
initial baseline** but red transiently during the Stage-B boundary (they use the TILE test packer that
still matches the TILE kernel until Stage B flips both): `qus_linear_test`, `qus_q5090_pack_golden_test`,
`qus_embed_gather_test`, `qus_weight_store_real_file_test` (reads the v1 file, green until the parser
becomes v2-only). This baseline makes every subsequent "green" attributable; do not treat a pre-existing
red as introduced by a Phase-2 task.

**Stage-B is ONE functional integration boundary (Tasks 3–5).** The codec interface, the `LinearFormat`
enum, the `linear()` wrapper validation/dispatch, the embed-gather kernel, and the single
`qus_linear_test`/`qus_embed_gather_test` all share one compile-and-dispatch boundary; none of T3/T4/T5
can build-and-verify in isolation (e.g. `qus_linear_test` exercises the public `linear()` wrapper, whose
validation+dispatch only accept `ROW_SPLIT` after T5). Therefore, following the attention-plan
precedent: implement T3→T4→T5 **in order, in the same worktree, with no bisectable commit and no
per-task verification or review until T5 lands**. The G-KERNEL parity + `compute-sanitizer` gate and the
spec/quality reviews run **once at the boundary close** (after T5). T3/T4 DoDs are checkpoints, not
standalone gates.

**Stage-B coordination points (shared edits inside the boundary):** the codec `load_group` signature
change (single `payload` ptr → separate `code_ptr` + `scale_ptr`, segment-relative `(row,group)`
addressing) is shared by the GEMV and GEMM kernels in `linear_generic_lowbit.cuh`; the `LinearFormat`
`Q*G64_N64K64` → `Q*G64_RowSplit` rename touches its definition (`linear_plan.h`) and every `switch`
that uses it at once; the `LinearPolicyId::TunedLowbitGemv` enumerator + `linear_tuned_lowbit_gemv_launch`
+ its dispatch case + its include are **removed** at the boundary (the registry routes `T1` quantized to
`GenericLowbitGemv`), and a tuned plan is reintroduced only in Stage D (Task 10).

Performance agents must read and follow before profiling:
- `/home/neroued/.cursor/skills/profile-cuda/SKILL.md`
- `/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`, `nsys-inference-analysis/SKILL.md`

## Scope and ownership (files)

**Stage A — Load:** `include/qus/core/tensor.h` (`QuantLayout`+`Weight`), `weight_store_parser.{h,cpp}`,
`weight_store.{h,cpp}`, `include/qus/core/weight.h`, `tools/parity/block_dump.cpp`,
`tools/parity/layer_dump.cpp`. Tests/fixtures owned by Stage A:
`tests/fixtures/make_q5090_fixture.py` (emit v2), `tests/test_q5090_parser.cpp`,
`tests/test_weight_store.cpp`, `tests/test_weight_store_real.cpp` (repoint to the v2 file).

**Stage B — Kernels (correctness), single boundary:** `src/kernels/linear/codec/linear_codec.cuh`,
`src/kernels/linear/reference/linear_generic_lowbit.cuh`, `linear_generic_lowbit_gemv.cu`,
`linear_generic_lowbit_gemm.cu`, `linear_generic.h`,
`src/kernels/linear/plan/linear_plan.{h,cpp}`, `src/kernels/linear/linear.cpp`;
embed-gather full path: `src/kernels/wrapper/embed_gather.cpp`,
`src/kernels/kernel/embed_gather.cuh`, `src/kernels/launcher/embed_gather.{h,cu}`,
`include/qus/kernels/embed_gather.h`; **remove**
`src/kernels/linear/gemv/linear_lowbit_gemv.{h,cuh,cu}`; tests/benches:
`tests/kernels/q5090_pack.h`, `tests/kernels/test_linear.cpp`,
`tests/test_q5090_pack_golden.cpp`, `tests/kernels/test_embed_gather.cpp`,
`bench/linear_bench.cu`, `bench/embed_gather_bench.cu`, `CMakeLists` as needed.

**Stage C — Integration:** `src/model/qwen3_6_27b.cpp` (verify), `src/runtime/engine.cpp` (verify),
`tests/test_model_bind.cpp`, `tests/test_model_blocks.cpp`, `tests/CMakeLists.txt`.

**Stage D — Tuning:** new tuned kernels under `src/kernels/linear/gemv/` (e.g.
`linear_rowsplit_gemv.{h,cuh,cu}`) and, if needed, `src/kernels/linear/gemm/`; `linear_plan.{h,cpp}`
(register tuned plans); `bench/` linear bench (if a per-op bench is needed); profiles under
`profiles/ncu-linear-v2/` and `profiles/nsys/` (local artifacts only).

**Coordination points:** `tensor.h` (T1, consumed by all) → `weight_store*` (T1/T2) → linear backend
+ embed-gather (T3–T5 boundary) → model/tests (T6) → `linear_plan.{h,cpp}` again in Stage D (register
tuned plans). The **test/fixture seam** (`make_q5090_fixture.py` consumed by the parser/weight-store
/bind tests) and the **codec interface** are explicit shared edits; edit shared files only in the
owning task/boundary.

## Reference paths

- v2 weights: `out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus`; preserved v1: `out/…mixed_v1.qus` (do not write).
- HF bf16 oracle (diagnostic only): `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16`
- Phase-1 dumps: `out/conv_dump.v2.json` (converter), `out/ref_dump.v2.json` (Python ref).
- Greedy snapshot (G-SNAPSHOT authority): `profiles/e2e/m3-output-gate.json`; prompt fixtures under
  `bench/fixtures/prompts/` (canonical case `cn_short` = `bench/fixtures/prompts/cn_short.ids`).

---

# Stage A — Load the v2 layout

## Task 1 — v2 parser & `QuantLayout`, + v2 fixture & parser test

**Files:** `tensor.h`, `weight_store_parser.{h,cpp}`, `tests/fixtures/make_q5090_fixture.py`,
`tests/test_q5090_parser.cpp`.
**Reading:** binary spec §1–§9; current `weight_store_parser.{h,cpp}`, `tensor.h`, the v2 converter
writer in `tools/q5090_convert/format.py`/`convert.py` (the fixture should mirror it), current
`make_q5090_fixture.py` and `test_q5090_parser.cpp`.
**Requirements:**
- Parse the v2 header (magic `Q5090MIXEDV2`, version 2, the segment/fusion offset+count fields,
  `format_minor`), `ModuleRecord`, `TensorEntry` (block: `segment_count`, `segment_begin`,
  `fusion_group_id`, `fusion_index`, `code_plane_bytes`, `scale_plane_bytes`), and the new
  `SegmentRecord` + `FusionGroupRecord` tables into `ParsedQ5090File` (add `segments` and
  `fusion_groups` vectors and the new `ParsedQ5090Tensor`/`ParsedQ5090Header` fields).
- `QuantLayout` becomes exactly `{ RowSplit = 0, Contiguous = 1 }`; drop `TileN64K64`,
  `W4A16KernelPackedV1`, `TileN64K128`, `RowGroupedG64`. Repoint the default `Weight.layout` (currently
  `W4A16KernelPackedV1`) and `ParsedQ5090Tensor.layout` default to `RowSplit`.
- Validate contract **G-STRUCT** structural items (incl. plan-conformance: offsets in range/ordered,
  payloads 256-aligned/non-overlapping, per-block `crc32`, segments partition `[0,N)`, fusion adjacency
  + `source_kind` rule, plane-byte formulas).
- **Migrate the fixture to v2:** `make_q5090_fixture.py` must emit a small **v2** file (ROW_SPLIT
  standalone + a multi-segment/fused block + CONTIGUOUS control + at least one fusion group), reusing
  the converter's v2 writer/qtypes where possible; it must not reference any deleted v1 layout enum.
  Update `test_q5090_parser.cpp` assertions to the v2 structure.
**DoD:** `qus_q5090_parser_test` builds and passes against the regenerated v2 fixture; L0/G-STRUCT
structural checks pass; `rg -n "TileN64K64|TileN64K128|RowGroupedG64|W4A16KernelPackedV1|Q5090MIXEDV1" include/ src/core/weight_store_parser.* tests/fixtures/make_q5090_fixture.py tests/test_q5090_parser.cpp`
is empty.

## Task 2 — `WeightStore` segment binding + cpp dumps

**Files:** `weight_store.{h,cpp}`, `weight.h`, `tools/parity/block_dump.cpp`,
`tools/parity/layer_dump.cpp`, `tests/test_weight_store.cpp`, `tests/test_weight_store_real.cpp`.
**Reading:** tensor-plan doc; contract G-STRUCT/G-DUMP; current `weight_store.cpp`, `block_dump.cpp`,
`layer_dump.cpp`, the two weight-store tests.
**Requirements:**
- Build the `Weight` view table from **segments**, not per-tensor: for each segment, build a `Weight`
  with `qdata = code_plane + row_begin·G·bpr`, `scales = scale_plane + row_begin·G·2`, `n = row_count`,
  `k = K`, `layout = RowSplit`, identity `(source_kind, source_layer)` + canonical name; dense `Tensor`
  from `CONTIGUOUS` blocks. The `qweight(module, source_kind, source_layer)` and `qweight(name)` keys
  are unchanged (a fused block now yields one `Weight` per segment, each keyed by its own identity).
- Emit cpp dumps (`block_dump`, and `layer_dump` if it consumes the parser/store) matching the
  converter/Python schema. Repoint `test_weight_store_real.cpp` from `…mixed_v1.qus` to
  `…mixed_v2.qus`; update `test_weight_store.cpp` to the v2 fixture + segment-binding assertions.
**DoD (G-DUMP):** every projection resolves to a segment view; `qus_weight_store_test` and
`qus_weight_store_real_file_test` pass; `compare_dumps` shows cpp == converter == Python on
block/segment/fusion metadata and recovered `(scale16, q_i)` **bit-exact** (sampled dequant within a
tiny tolerance — compare recovered codes/scales, not dtype-dependent dequant floats).

# Stage B — Adapt the kernels (correctness, untuned) — ONE integration boundary (T3→T4→T5)

> Implement T3, T4, T5 sequentially in the same worktree. **Do not** run `qus_linear_test` /
> `qus_embed_gather_test` / `compute-sanitizer`, make a bisectable commit, or dispatch the spec/quality
> reviewers until T5 is complete. The boundary verification (below) runs once. T3/T4 DoDs are internal
> checkpoints (the tree is expected to be non-building between T3 and T5).

## Task 3 — `ROW_SPLIT` codec + generic decode GEMV (`T=1`)

**Files:** `linear_codec.cuh`, `linear_generic_lowbit.cuh`, `linear_generic_lowbit_gemv.cu`,
`linear_generic.h`, `q5090_pack.h`, `test_linear.cpp`, `tests/test_q5090_pack_golden.cpp`.
**Reading:** binary spec §9; current codec + generic GEMV; `q5090_pack.h`, `test_q5090_pack_golden.cpp`.
**Requirements:** change the codec to take **two** plane pointers and address segment-relative
`(row,group)` → code `code_ptr+(row·G+group)·bpr`, scale `scale_ptr+(row·G+group)·2`; reuse §9.1
unpack. Implement a **correct, untuned** generic GEMV over `ROW_SPLIT`. Update the launch to pass
`w.qdata` (code plane) + `w.scales` (scale plane). Update the test packer (`q5090_pack.h`) and the
golden test (`test_q5090_pack_golden.cpp`) to `ROW_SPLIT`.
**Checkpoint (verified at boundary close, not here):** generic GEMV math implemented for Q4/Q5/Q6 incl.
non-64 tails; the GEMM kernel in the shared `.cuh` is left in a consistent state for Task 4.

## Task 4 — generic prefill GEMM (`T>1`)

**Files:** `linear_generic_lowbit_gemm.cu`, `linear_generic_lowbit.cuh` (shared codec/kernels),
`test_linear.cpp`.
**Reading:** binary spec §9; current generic GEMM.
**Requirements:** correct, untuned `ROW_SPLIT` GEMM (`T>1`) on the same two-plane codec; SMEM-staged
per-row segment loads (decode-first layout, prefill compute-bound — correctness only here). The shared
`.cuh` must compile with both GEMV (T3) and GEMM kernels on the new codec signature.
**Checkpoint (verified at boundary close, not here):** GEMM math implemented across qtypes/shapes.

## Task 5 — Dispatch rename, remove TILE, embed gather + boundary verification

**Files:** `linear_plan.{h,cpp}`, `linear.cpp`, **remove** `linear/gemv/linear_lowbit_gemv.*`,
`embed_gather.cpp`, `src/kernels/kernel/embed_gather.cuh`, `src/kernels/launcher/embed_gather.{h,cu}`,
`include/qus/kernels/embed_gather.h`, `tests/kernels/test_embed_gather.cpp`, `bench/linear_bench.cu`,
`bench/embed_gather_bench.cu`, `CMakeLists`.
**Reading:** framework doc §5–§14; current `linear_plan.*`, `linear.cpp`, the embed-gather
kernel/launcher/wrapper, the embed test/bench.
**Requirements:**
- `LinearFormat` `Q*G64_N64K64` → `Q*G64_RowSplit`; `classify_format` maps v2 (`RowSplit`) weights;
  rewrite `linear.cpp` validation (replace `require_tile_lowbit_metadata` with a row-split validator
  that checks the §9.2 code/scale plane sizes, unpadded `N`, `K_pad`) and dispatch so the registry
  resolves all quantized keys to the **generic** `ROW_SPLIT` GEMV/GEMM. **Repoint `resolve_plan` `T1`
  quantized from `TunedLowbitGemv` to `GenericLowbitGemv`; delete the `LinearPolicyId::TunedLowbitGemv`
  enumerator, the `linear_tuned_lowbit_gemv_launch` decl/use, its dispatch `case`, and the
  `kernels/linear/gemv/linear_lowbit_gemv.h` include** (the tuned plan returns in Stage D).
- **Embed gather on `ROW_SPLIT`:** the wrapper validates `layout == RowSplit` (drop the RowGrouped
  branch); the kernel + launcher read one row from the **two** Q6 planes (code plane row run +
  separate scale plane), not the inline 50-byte RowGrouped rows. Update `test_embed_gather.cpp` and
  `embed_gather_bench.cu` to ROW_SPLIT.
- Benches build: `linear_bench.cu` / `embed_gather_bench.cu` use the ROW_SPLIT packer/weights.
  Framework types/flow otherwise unchanged.
**Boundary verification (run once, after T5):**
- `cmake --build build -j` (incl. benches + tools) is clean.
- `qus_linear_test` fp64-oracle parity for Q4/Q5/Q6 **GEMV and GEMM** (incl. non-64 tails) — G-KERNEL.
- `qus_embed_gather_test` passes (Q6 ROW_SPLIT gather).
- `compute-sanitizer --tool memcheck ./build/tests/qus_linear_test` and
  `… ./build/tests/qus_embed_gather_test` clean.
- `rg -n "TileN64K64|TileN64K128|RowGroupedG64|W4A16KernelPackedV1|N64K64|RowGrouped|linear_lowbit_gemv|TunedLowbitGemv|Q5090MIXEDV1" src/ include/`
  is empty.
Then dispatch the spec-compliance reviewer and the code-quality reviewer for the whole boundary.

# Stage C — Integrate & prove correct inference

## Task 6 — Model integration & block parity

**Files:** `qwen3_6_27b.cpp` (verify), `engine.cpp` (verify), `test_model_bind.cpp`,
`test_model_blocks.cpp`, `tests/CMakeLists.txt`.
**Requirements:** confirm `bind()` resolves every projection from segment weights with no model-card
structural change; confirm the engine loads the v2 file and that `Engine::default_weight_bytes` (already
file-size-derived) covers the v2 payload (verify, not change). Adapt the bind / block tests to v2 (they
also depend on the Task-1 v2 fixture).
**DoD:** `qus_model_bind_test` passes (every projection binds from a segment; structural).
`qus_model_blocks_test` passes against its in-tree cpp/recorded reference within a bf16 tolerance —
**no HF gate and no per-op cpp-vs-Python comparison** (op correctness is the CUDA↔cpp fp64 oracle,
G-KERNEL).

## Task 7 — End-to-end correctness + sanitizer

**Requirements:** run the cpp engine on the v2 file and reproduce the recorded quantized greedy snapshot
**exactly for the snapshot's own length** (contract **G-SNAPSHOT**). The cpp authority is
`qus_e2e_bench` (the binary that produced `profiles/e2e/m3-output-gate.json`); run it on the v2 file and
compare the report's `generated_token_ids` to the snapshot, per case, for the snapshot's length:

```bash
cmake --build build --target qus_e2e_bench -j
./build/bench/qus_e2e_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --output-json out/e2e.v2.json \
  --fixture-manifest bench/fixtures/prompts/m2.8-v1.manifest.json \
  --case cn_short:bench/fixtures/prompts/cn_short.ids:96 \
  --warmup-repeats 1 --repeats 1 --max-ctx 8192 --device 0 \
  --stop-token-id 248046 --stop-token-id 248044
# G-SNAPSHOT: out/e2e.v2.json case cn_short generated_token_ids must equal
# profiles/e2e/m3-output-gate.json case cn_short generated_token_ids, exactly,
# for the snapshot's length (use tools/bench/compare_e2e_reports.py token compare,
# or assert the generated_token_ids prefix directly). cn_short is the canonical case
# (Phase-1 used it); optionally repeat for en_short/code_short/math_short.
```

HF divergence is reported as a diagnostic only, never a gate. `compute-sanitizer --tool memcheck` clean
over a short decode on the load + linear path. There is **no** per-op cpp-vs-Python comparison — cpp op
correctness is the CUDA↔cpp fp64 oracle (**G-KERNEL**, Tasks 3/4); end-to-end correctness is the
snapshot.
**DoD (G-SNAPSHOT):** greedy matches the snapshot exactly for its length; memcheck clean. (Throughput
irrelevant here.)

## Task 8 — Remove residual cpp v1 layout code (audit)

**Requirements:** no `TileN64K64`/`TileN64K128`/`RowGroupedG64`/`W4A16KernelPackedV1`/v1 magic/tuned-TILE
references in cpp, tests, benches, or parity tools. v1 *weight file* and v1 *docs* untouched (Phase 3).
**DoD:**
```bash
rg -n "TileN64K64|TileN64K128|RowGroupedG64|W4A16KernelPackedV1|TILE_N64|RowGrouped|linear_lowbit_gemv|TunedLowbitGemv|Q5090MIXEDV1" \
   src/ include/ tests/ tools/ bench/   # empty
cmake --build build -j && ctest --test-dir build   # green (all targets build; no pre-existing red remains)
```

> **CORRECTNESS GATE.** Stage D does not begin until T1–T8 are green and the post-T8 review passes.

# Stage D — Per-kernel tuning (performance)

Decode is **weight-bandwidth bound** (~15.4 GB of weights streamed per token); the v2 `ROW_SPLIT` layout
was designed for GEMV-friendly streaming (row-contiguous, coalesced, vectorizable, `N`-parallel). The v1
tile kernels were layout-capped at ~64% weighted DRAM; the goal here is to let the tuned `ROW_SPLIT`
kernels approach the cold-cache DRAM ceiling per shape.

**Methodology (every round):**
- One identified limiter per round; before/after `ncu` artifacts under `profiles/ncu-linear-v2/`.
- Cold-cache (L2-flushed) `dram__throughput.avg.pct_of_peak_sustained_elapsed` is the gate; hot-cache
  numbers are diagnostic only. Re-calibrate the achievable ceiling for `ROW_SPLIT`.
- **Correctness re-verified every commit**: `qus_linear_test` (fp64 oracle) + the Task-7 greedy parity
  must stay green. A tuning change that breaks either is reverted.
- Register tuned plans in the existing `LinearPlan` registry, dispatched by `ShapeFamily`; the generic
  `ROW_SPLIT` plan remains the correctness fallback. (Reintroduce a `LinearPolicyId` for the tuned
  decode GEMV here, removed in Stage B.)

## Task 9 — Baseline profile & roofline

**Files:** none (profile only); artifacts under `profiles/ncu-linear-v2/`.
**Requirements:** ncu cold-cache the generic `ROW_SPLIT` GEMV on the dominant shapes
(`MlpGateUp17408x5120` Q4, `MlpDown5120x17408` Q5, `Out5120x6144`/`Proj6144x5120` Q5,
`LmHead248320x5120` Q6); report duration, achieved occupancy, DRAM/L2 throughput, sectors, top
limiter; compute the per-shape weight-bandwidth roofline. Identify the first limiter to attack per shape.
**DoD:** a metric-backed baseline + per-shape limiter list; no source edits.

## Task 10 — Tuned `ROW_SPLIT` decode GEMV (per shape, profile-driven)

**Files:** new `src/kernels/linear/gemv/linear_rowsplit_gemv.{h,cuh,cu}`, `linear_plan.{h,cpp}`,
optional `bench/` linear per-op bench.
**Requirements:** implement a tuned `ROW_SPLIT` decode GEMV and tune it per shape family, one limiter
per round (candidate levers: warp-per-row vs warps-per-row; vectorized 16-byte code loads; scale-plane
prefetch; loop unroll; `N`-parallel CTA count for occupancy; minimizing the `Q5/Q6` unpack cost). Wire
hot shapes to the tuned plan via the registry (new `LinearPolicyId` + `resolve_plan` entry). Re-verify
correctness each round.
**DoD:** for each dominant shape, cold-cache DRAM throughput materially exceeds the Task-9 baseline and
approaches the calibrated ceiling, or `ncu` shows a non-bandwidth limiter requiring a larger change
(documented). Correctness green throughout; before/after `ncu` artifacts saved.

## Task 11 — End-to-end nsys + optional prefill GEMM tuning

**Files:** none for nsys (artifacts under `profiles/nsys/`); prefill GEMM tuning only if pursued
(`src/kernels/linear/gemm/…`, `linear_plan.{h,cpp}`).
**Requirements:** nsys a long decode run on the v2 file; confirm the tuned GEMV moves real decode time
(lower-bit GEMV share drops, tok/s rises) and report the new top decode bottleneck. Prefill GEMM tuning
is **optional and lower priority** (prefill is compute-bound, not the decode bottleneck); pursue only if
TTFT is a concern.
**DoD:** e2e nsys shows the per-op gains move the real decode workload; the remaining top bottleneck is
named with profile evidence.

> **Beyond per-kernel tuning (next phase, not here):** the largest remaining decode lever is the
> **fused-projection group GEMV** (one large-`N` GEMV per fusion block, framework §21.5), which the v2
> layout already stores for. It changes the L1/L2 call structure and is planned separately.

## Definition of done

**Correctness (Stages A–C):** the cpp `ctest` baseline is restored to green (Phase-1-induced red
fixtures fixed); cpp loads v2; every projection binds as a `ROW_SPLIT` segment; cpp/converter/Python
dumps agree bit-exact (G-DUMP); generic GEMV/GEMM and the Q6 embed gather pass the CUDA↔cpp fp64 oracle
+ memcheck (G-KERNEL); model bind + block tests pass against the in-tree reference; the cpp engine
greedy reproduces the recorded snapshot exactly (G-SNAPSHOT) via `qus_e2e_bench`; HF metrics are
diagnostic only; `compute-sanitizer` clean; framework unchanged; no TILE/v1 layout code remains in
src/include/tests/tools/bench; v1 weight file untouched.

**Performance (Stage D):** per dominant shape, the tuned `ROW_SPLIT` GEMV beats the generic baseline and
approaches the cold-cache ceiling (or the limiter is documented); e2e nsys confirms the decode gain;
correctness gates remained green throughout; tuned plans are registered in the existing framework.

## Review phase

Risk: CUDA kernels, numerical behavior, q5090 ABI, GPU memory, benchmark evidence. Reviews:

**After T8 (correctness):**
1. **Numerical-correctness reviewer** — codec addressing, generic GEMV/GEMM math, the Q6 embed-gather
   dequant, the CUDA↔cpp fp64 oracle (G-KERNEL), and the snapshot match (G-SNAPSHOT); confirms every HF
   comparison is diagnostic only, never gating.
2. **CUDA memory/lifetime reviewer** — segment sub-view bounds, weight-load/arena lifetime,
   `embed_gather` plane bounds; `compute-sanitizer` evidence.
3. **Format/ABI reviewer** — parser vs spec, segment binding, dump cross-check, `source_kind` rule, and
   the v2 **fixture/test migration** (fixture emits valid v2; parser/weight-store/bind tests exercise
   segment binding and pass).
4. **Scope reviewer** — framework unchanged, no perf kernels yet, no fused-group GEMV, v1 file
   untouched, all cpp/tests/bench/tools v1 layout code removed, baseline red fully accounted for.

**After T11 (performance):**
5. **Performance-evidence reviewer** — claims are cold-cache `ncu`/`nsys`-backed (not hot-cache), one
   limiter per round with before/after artifacts, and correctness (fp64 oracle + greedy) stayed green
   across every tuning commit.
