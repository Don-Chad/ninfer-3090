# Qwen3.6-27B Decode Projection and Artifact Cutover Plan

> Status: implemented and archived on 2026-07-17
>
> Date: 2026-07-17
>
> Target: `qwen3_6_27b_rtx5090` on RTX 5090, CUDA 13.1
>
> Scope: one atomic GDN artifact-layout cutover plus two decode input-projection
> implementation changes. This plan does not register another target or artifact lane.

## Completion record

The source, converter, verifier, Python reference, C++ binder, tests, active documentation, and
canonical artifact were cut over atomically. The canonical artifact contains 1124 objects (1118
tensors and six resources), including 48 physical `gdn/value_z` parents and no physical
`gdn/value` or `gdn/z`. Before promotion, an exact migration audit compared 1076 unaffected objects
and every value/z code, high-bit, and scale row range; the canonical path was then re-inspected,
source-verified, and loaded through the public Engine. The recipe ID and container version remain
unchanged, and the obsolete artifact and candidate files were removed.

The fixed-shape benchmark showed the two-parent Attention route faster than the former
four-projection control at every `T=1..16` (37.920 vs 48.064 us at T1; 134.336 vs 173.024 us at
T16). GDN direct output retained projection time while removing materialization/copy overhead
(29.984 vs 35.136 us at T1; 123.936 vs 127.904 us at T16). The ordinary `tg32` NSYS trace records
528 Q4-parent and 528 Q5-parent launches, and D2D topology changed exactly as designed: 8192-byte
events 1584 -> 0, 12288-byte events 3168 -> 1584, and all D2D events 8017 -> 4849.

An initial five-repetition end-to-end sample was retained, but its ordinary-decode result did not
support a speedup claim; the user explicitly waived additional matched A/B attribution and asked to
finish the remaining cutover work. Accordingly, this archive claims the operator and topology
results above, not an end-to-end throughput improvement.

## 1. Required outcome

Complete these three changes as one bounded implementation effort:

1. Replace each 27B GDN layer's two physical Q5 objects, `gdn/value` and `gdn/z`, with one
   physical `gdn/value_z [12288,5120]` object and two logical row views. Regenerate the canonical
   artifact, validate the candidate against the checkpoint and both consumers, then replace the
   canonical artifact without retaining the obsolete weight file.
2. Change full-attention input projection so the already-fused Q4 `query_key [7168,5120]` parent
   performs one projection and the already-fused Q5 `gate_value [7168,5120]` parent performs one
   projection. The registered implementation must therefore execute two projection kernels, not
   four, for decode and Small-T verification.
3. Remove the two Small-T GDN input-projection D2D copies. Keep QK and V as two independent
   projections and write them directly through row-sliced views of the final contiguous
   `qkv [10240,T]` tensor.

Completion means the new artifact contract, converter, verifier, Python reference, C++ binding,
runtime implementation, tests, active documentation, canonical artifact, and retained performance
evidence all describe and exercise the same route.

## 2. Locked decisions

These decisions are not implementation alternatives:

- This is an atomic cutover. Do not add old-name aliases, dual binders, optional-object handling,
  artifact probing, fallback recipes, or runtime compatibility branches.
- The rebuilt program intentionally rejects the old artifact and the old program intentionally
  rejects the new artifact even though version identifiers stay unchanged. Source and canonical
  artifact are promoted as one owner-controlled release state.
- Do not change `model_id`, the common `.ninfer` container schema, a numeric-format definition, a
  storage-layout definition, the conversion-report schema, or
  `RECIPE_ID = "qwen3_6_27b_rtx5090-v1"`.
- The only persistent-object change is `gdn/value` + `gdn/z` -> `gdn/value_z`. Attention objects,
  MLP objects, A/B projections, MTP objects, Vision objects, resources, quantization assignments,
  and object semantics remain unchanged.
- `value` and `z` remain separate logical roles and separate runtime projections. Only their
  persistent storage parent changes.
- Attention remains two homogeneous projections: one Q4 Query/Key projection and one Q5
  Gate/Value projection. Do not introduce a Q4+Q5 mixed-format kernel.
- GDN QK and V remain two independent projection kernels. P1.2 changes only output addressing and
  removes temporary tensors and D2D copies.
- Public `ops::linear()` continues to require contiguous output. Strided output support remains an
  implementation detail used by `gdn_input_proj`; do not broaden the public Linear contract.
- The work does not include 35B runtime bring-up, A/B artifact fusion, V/Z compute fusion, MTP
  projection restructuring, convolution layout changes, or unrelated kernel tuning.
- The separate 35B artifact design already stores GDN Q/K/V/Z in one
  `gdn/query_key_value_z` parent, so it has no corresponding split V/Z objects to migrate. Use that
  only as design confirmation; do not modify the 35B converter, binder, reference, or documents in
  this cutover.

## 3. Baseline and success measures

The retained ordinary-decode topology baseline is the Release RTX 5090 `tg32` workload recorded
under:

```text
profiles/nsys/linear-architecture-20260716/release/text_tg32.nsys-rep
profiles/nsys/linear-architecture-20260716/release/text_tg32.sqlite
profiles/nsys/linear-architecture-20260716/release/text_tg32_summary.md
profiles/nsys/linear-architecture-20260716/release/text_tg32.json
```

Its exact command is recorded in the JSON report. Use the same binary mode, artifact role,
`-n 32`, CUDA Graph setting, GPU, and toolchain for the post-change structural comparison. The
matching retained MTP topology evidence is:

```text
profiles/nsys/linear-architecture-20260716/release/mtp_pg32x32_draft.nsys-rep
profiles/nsys/linear-architecture-20260716/release/mtp_pg32x32_draft.sqlite
profiles/nsys/linear-architecture-20260716/release/mtp_pg32x32_draft_summary.md
profiles/nsys/linear-architecture-20260716/release/mtp_pg32x32_draft.json
```

Those NSYS runs are sufficient for old kernel/copy topology but their single measured repetition
does not establish end-to-end no-regression. Before changing source or the canonical artifact,
capture matched multi-repetition old-route results:

```bash
mkdir -p profiles/bench/decode-projection-cutover-20260717

./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  -n 32 --warmup 2 -r 5 --output json \
  --output-file profiles/bench/decode-projection-cutover-20260717/before_text_tg32.json

./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  -pg 32,32 --mtp-draft-tokens 5 --lm-head-draft \
  --warmup 2 -r 5 --output json \
  --output-file profiles/bench/decode-projection-cutover-20260717/before_mtp_tg32_draft.json
```

Run the exact same commands against the explicit candidate with distinct `after_*.json` paths.
Retain the JSON/summary, not an old artifact copy. A new fixed-shape input-projection benchmark in
Sections 5/6 provides same-build production-vs-old-control operator comparisons after cutover, so
the obsolete program/artifact is not needed for later microbenchmark reruns.

Structural performance requirements are:

- each full-attention layer executes one Q4 input-projection kernel and one Q5 input-projection
  kernel for T=1 and admitted Small-T verification shapes;
- each GDN layer retains one Q4 QK kernel and one Q5 V kernel;
- the two GDN input-projection D2D copies disappear;
- no new materialization, split, transpose, synchronization, allocation, or host round trip appears
  in steady-state decode;
- each affected operator is faster than its old composition, and ordinary and MTP end-to-end
  decode do not regress.

## 4. Workstream A: atomic `gdn/value_z` artifact cutover

### 4.1 New physical contract

For every one of the 48 GDN layers, replace:

```text
text/layers/{l}/gdn/value  Q5G64_F16S [6144,5120]
text/layers/{l}/gdn/z      Q5G64_F16S [6144,5120]
```

with:

```text
text/layers/{l}/gdn/value_z Q5G64_F16S [12288,5120]

rows [0,6144)       -> logical value [6144,5120]
rows [6144,12288)   -> logical z     [6144,5120]
```

The source expression is:

```python
Concat(
    (
        Slice(in_proj_qkv, 0, 4096, 10240),
        SourceTensor(in_proj_z, (6144, 5120)),
    ),
    0,
)
```

Materialize the BF16 concatenation first, then run the existing Q5 row-wise quantizer once over the
`[12288,5120]` matrix. Per-row codes and scales must therefore match the corresponding rows of the
old objects.

For K=5120 and G64, the new RowSplit payload geometry is:

| Plane | Bytes per row | Rows | Bytes |
|---|---:|---:|---:|
| base codes | 2560 | 12288 | 31,457,280 |
| high bits | 640 | 12288 | 7,864,320 |
| FP16 scales | 160 | 12288 | 1,966,080 |
| total | | | 41,287,680 |

This equals the sum of the two old payloads, so tensor payload bytes, H2D bytes, and device weight
arena bytes do not grow. The valid parent order is:

```text
base(value), base(z), high(value), high(z), scale(value), scale(z)
```

Do not concatenate the two complete old payload byte strings, which would place each object's
high/scale planes between the two base planes.

Do not hardcode the final `.ninfer` file size before conversion. The shorter JSON directory and
removal of one per-layer object boundary can change 4 KiB-aligned offsets even though the sum of
encoded tensor payload bytes is unchanged.

### 4.2 Converter and verifier changes

Update the registered target-private conversion implementation:

| File | Required change |
|---|---|
| `tools/convert/qwen3_6_27b_rtx5090/inventory.py` | Replace the two physical specs with `value_z`; add the two logical row-view specs; update derived counts. |
| `tools/convert/qwen3_6_27b_rtx5090/recipe.py` | Replace the two recipes with the one `[value,z]` concat recipe. |
| `tools/convert/qwen3_6_27b_rtx5090/convert.py` | Update exact inventory preflight counts only; leave `RECIPE_ID` and report schema unchanged. |
| `tools/convert/qwen3_6_27b_rtx5090/verify.py` | Probe `gdn/value_z`; validate representative rows on both sides of the 6144-row seam. |

The expected contract counts after the cutover are:

| Count | Old | New |
|---|---:|---:|
| Text tensors | 819 | 771 |
| all tensors | 1166 | 1118 |
| all objects including six resources | 1172 | 1124 |
| Q5 tensors | 294 | 246 |
| RowSplit tensors | 487 | 439 |
| logical row-view templates | 14 | 16 |
| logical row-view bindings | 294 | 390 |

The converter's exact preflight tuple becomes:

```text
(6, 771, 2, 12, 333, 1118, 1124)
```

The verifier must compare at least rows `0`, `6143`, `6144`, and `12287` against source-derived
codes and FP16 scale words. Rows 6143/6144 protect the physical seam; rows 0/12287 protect both
outer bounds. The exact source checkpoint tensor count remains 1199.

### 4.3 Python reference binding

Update `tools/reference/qwen3_6_27b_rtx5090/bindings.py` so its independent artifact contract:

- requires one physical `value_z [12288,5120]` object;
- exposes `value_z` as the owning `PhysicalBlock`;
- creates `value` and `z` as logical row views with the exact row ranges above;
- updates exact object, tensor, and row-view counts;
- rejects both old physical names and any artifact containing extra objects.

Do not change the mathematical schedule in
`tools/reference/qwen3_6_27b_rtx5090/text.py`: it continues to project `value` before convolution
and `z` after the recurrence. WeightStore must consume the logical views without copying or
repacking the parent.

### 4.4 C++ binding and schedule

Update:

```text
src/targets/qwen3_6_27b_rtx5090/impl/load/bindings.h
src/targets/qwen3_6_27b_rtx5090/impl/load/bindings.cpp
```

The target-private structures become conceptually:

```cpp
struct GdnPlan {
    // unchanged fields...
    artifact::ObjectHandle query_key;
    artifact::ObjectHandle value_z;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle output;
};

struct GdnWeights {
    // unchanged fields...
    Weight query_key;
    Weight query;
    Weight key;
    Weight value_z;
    Weight value;
    Weight z;
};
```

Binding requires exactly `gdn/value_z` with Q5 `[12288,5120]`. Materialization constructs the
parent and derives:

```cpp
target.value = row_view(target.value_z, 0, 6144);
target.z     = row_view(target.value_z, 6144, 6144);
```

The existing `row_view()` implementation already offsets base, high, and scale planes separately.
Retain the current target schedule bindings to `source.value` and `source.z`; no hot-path name
lookup and no V/Z compute fusion is introduced.

### 4.5 Active documentation and affected tests

Update the active authorities in the same implementation change:

- `docs/qwen3.6-27b-ninfer-artifact.md`: GDN object table, physical order, logical-view table,
  object/format/layout counts, source recipe, binder obligations, and implemented evidence;
- `docs/ninfer-container-format.md`: the target-specific complete-inventory evidence count;
- the Op contract headers changed by Workstreams B/C, where output and workspace contracts change.

Update only tests that protect the changed observable contract:

```text
tests/targets/qwen3_6_27b/test_inventory.py
tests/targets/qwen3_6_27b/test_recipe.py
tests/targets/qwen3_6_27b/test_verify.py
tests/targets/qwen3_6_27b/test_bindings.py
tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp
tests/test_ninfer_bench_support.cpp
```

Recipe testing must materialize a compact real-shape marker tensor and prove exact `[value,z]` row
order. Binding testing must prove that both logical views share the new parent and expose the exact
row ranges. The structural verifier tuple becomes
`(1124, 1118, 6, 16, 390, 4, 51)` for objects, tensors, resources, row-view templates/bindings, and
the unchanged alias templates/bindings. Do not add tests for deleted aliases or source-string
structure.

Before conversion, run a targeted exact-name/count scan over active code, tests, and documentation.
After the cutover, `gdn/value` and `gdn/z` may appear only as logical role names or in this plan's
before/after explanation, never as physical object requirements.

### 4.6 One-time candidate generation and old-artifact removal

Do not overwrite the canonical artifact before the candidate passes all source, structural,
numerical, binding, runtime, and performance gates. Use one explicit candidate path and aim for one
successful large conversion after converter/reference/C++ unit checks pass:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
CANDIDATE=out/qwen3_6_27b_rtx5090.cutover.ninfer

$PYTHON -m tools.convert.qwen3_6_27b_rtx5090.convert \
  --model "$MODEL" --out "$CANDIDATE"

$PYTHON -m tools.artifact.inspect "$CANDIDATE" --objects

$PYTHON -m tools.convert.qwen3_6_27b_rtx5090.verify \
  "$CANDIDATE" --model "$MODEL"
```

Use explicit candidate paths for Python reference, C++ real-artifact integration, parity, and
benchmark commands. Do not select the candidate by glob or modification time.

Before deleting the old canonical artifact, perform a one-time migration comparison while both
files exist:

- for every object unaffected by this cutover, compare name, kind, format, layout, shape, encoded
  byte count, and payload bytes exactly; directory offsets may move and are not compared;
- for all 48 layers, compare the new parent's value base/high/scale row ranges byte-for-byte with
  the old physical value object's corresponding planes;
- compare the new parent's z ranges byte-for-byte with the old physical z object's planes;
- require all 96 logical views to match and record only the summarized result;
- do not retain a permanent dual-artifact compatibility tool after the cutover.

The conversion report records the candidate path in `artifact.path`. Before promotion, stage a
canonical sidecar in the same directory with only that descriptive field changed to
`out/qwen3_6_27b_rtx5090.ninfer`; retain `arguments.out` as the truthful candidate-generation
argument. Do not change the report schema, recipe ID, object statistics, provenance, or measured
file bytes.

```bash
CANDIDATE_REPORT=out/qwen3_6_27b_rtx5090.cutover.ninfer.conversion.json
NEXT_REPORT=out/qwen3_6_27b_rtx5090.ninfer.conversion.json.next

jq '.artifact.path = "out/qwen3_6_27b_rtx5090.ninfer"' \
  "$CANDIDATE_REPORT" > "$NEXT_REPORT"

jq -e --argjson bytes "$(stat -c%s "$CANDIDATE")" \
  '.artifact.path == "out/qwen3_6_27b_rtx5090.ninfer" and
   .artifact.bytes == $bytes and
   .recipe_id == "qwen3_6_27b_rtx5090-v1"' \
  "$NEXT_REPORT" >/dev/null

cmp \
  <(jq -S 'del(.artifact.path)' "$CANDIDATE_REPORT") \
  <(jq -S 'del(.artifact.path)' "$NEXT_REPORT")
```

Once every pre-promotion gate in Section 8 passes and no Engine process holds the old file open,
replace the canonical files on the same filesystem:

```bash
mv -f \
  out/qwen3_6_27b_rtx5090.cutover.ninfer \
  out/qwen3_6_27b_rtx5090.ninfer

mv -f \
  out/qwen3_6_27b_rtx5090.ninfer.conversion.json.next \
  out/qwen3_6_27b_rtx5090.ninfer.conversion.json
```

The first `mv -f` is the atomic replacement of the runtime artifact path and unlinks the obsolete
canonical weight file. The sidecar is not consumed by Engine and follows immediately; there is no
portable multi-file rename transaction to claim. Do not keep a `.bak`, old-name artifact,
alternate versioned artifact, or fallback path. Inspect and smoke-load the canonical path once
after replacement, and remove the original candidate sidecar plus any failed/partial candidate
files.

## 5. Workstream B: two full-attention projections

### 5.1 Semantic and target binding change

The artifact already contains the correct parents:

```text
attention/query_key  Q4G64_F16S [7168,5120] = Q rows [0,6144) + K rows [6144,7168)
attention/gate_value Q5G64_F16S [7168,5120] = Gate rows [0,6144) + V rows [6144,7168)
```

Change `attn_input_proj` to accept those two parents and produce four independent contiguous BF16
outputs. The semantic contract states the four linear formulas and row mapping; it must not promise
a CUDA launch count. The registered implementation plan, benchmark evidence, and profiler topology
establish the required two-kernel implementation.

`FullAttentionWeights` already owns both parents and their four logical row views. Retain those
views because they are part of the active target binding description; only change `FullLayerW` and
the Text schedule to pass `query_key` and `gate_value` to this Op. Workstream B does not change the
artifact inventory, converter, Python binding/reference, or C++ artifact-binding contract.

The new Op call is conceptually:

```cpp
attn_input_proj(x, query_key, gate_value, q, gate, k, v, ws, stream);
```

Primary files:

```text
include/ninfer/ops/attn_input_proj.h
src/ops/wrapper/attn_input_proj.cpp
src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h
src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h
src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
src/ops/linear/q4/q4_rowsplit_plan.cpp
src/ops/linear/q4/q4_rowsplit_gemv.cuh
src/ops/linear/q4/q4_rowsplit_gemm_simt.cuh
src/ops/linear/q5/q5_rowsplit_plan.cpp
src/ops/linear/q5/q5_rowsplit_launch.cpp
src/ops/linear/q5/q5_rowsplit_gemv.cu
src/ops/linear/q5/q5_rowsplit_gemv.cuh
src/ops/linear/q5/q5_rowsplit_gemm_simt.cu
src/ops/linear/q5/q5_rowsplit_gemm_simt.cuh
src/targets/qwen3_6_27b_rtx5090/impl/schedule/text_context.h
src/targets/qwen3_6_27b_rtx5090/impl/schedule/text_context.cpp
src/CMakeLists.txt
bench/ops/input_proj_bench.cu
bench/CMakeLists.txt
```

Add one thin Op-owned Small-T CUDA source, for example
`q4_q5_attn_input_small_t.cu`, which instantiates the existing Q4/Q5 mainloops with a compile-time
two-output store policy. Do not duplicate weight decode or accumulation, add a generic executor,
or add an arbitrary split-list abstraction. Keep exact-shape code under `src/ops`; target code only
composes the Op.

Rename the Small-T implementation-only route so the plan describes what it now does:

```text
IndependentFixed -> ParentSplitFixed
four subplans     -> query_key Q4 plan + gate_value Q5 plan
```

### 5.2 Q4/Q5 `[7168,5120]` routes

For T<=16, add exact Q4 and Q5 `[7168,5120]` route support and fixed instantiations. Q4 candidate
legality already admits this aligned N. Q5 requires extending the current N=6144-only
`GemvR16S2X` and `SimtSplit4Exact` legality plus adding the N=7168 GEMV/SIMT instances. Reuse the
existing Q4/Q5 decode, K-loop, and accumulation implementation. The only attention-specific
epilogue maps the parent row to one of two output tensors:

```text
Q4 rows [0,6144)    -> q    with ld=6144
Q4 rows [6144,7168) -> k    with ld=1024
Q5 rows [0,6144)    -> gate with ld=6144
Q5 rows [6144,7168) -> v    with ld=1024
```

The split row 6144 is divisible by every relevant row tile: Q4 GEMV R1/R4, Q4/Q5 SIMT R8, Q5
GEMV R16, Q5 Split-K4 R1, and grouped MMA R64. Select the output base and local row once per
CTA/tile before the K-loop; do not branch inside weight decode or FMA loops, materialize a parent
output, or copy child ranges afterward.

Benchmark this finite Small-T candidate set at exact N=7168:

| T | Q4 candidates | Q5 candidates |
|---:|---|---|
| 1 | GEMV R1/W8, GEMV R4/W1 | GEMV R16/S2/X, SIMT R8/C4 |
| 2-6 | SIMT R8/C4, SIMT R8/C8 | Split-K4 exact, SIMT R8/C4, SIMT R8/C8 |
| 7-16 | SIMT R8/C4, SIMT R8/C8 | SIMT R8/C4, SIMT R8/C8 |

The expected seed is Q4 `1:R1/W8, 2-7:C4, 8-16:C8` and Q5
`1:R16/S2/X, 2-6:Split-K4, 7-16:C8`, but do not freeze the thresholds without the exact-shape
measurements. Commit the fastest numerically qualified choice as a closed compile-time catalog; do
not add runtime autotuning. Regardless of threshold, Small-T execution is exactly one Q4 kernel
plus one Q5 kernel, with no four-call fallback.

For T>=17, keep the existing two homogeneous-pair MMA launches and mainloop unchanged. Extend job
construction with parent-plane row offsets/counts so Q/K come from one Q4 parent and Gate/V from
one Q5 parent. Do not reintroduce target-schedule child arguments or change the artifact.

### 5.3 Correctness gates

Extend the existing input-projection plan and CUDA numerical tests rather than creating structural
tests. Exercise every Small-T point plus the large-T boundaries:

```text
T = 1..16, 17, 127, 128, 129
```

For each T:

- build real Q4/Q5 `[7168,5120]` parent payloads with the registered codec;
- use the independent naive FP64 row-split decode/dot-product oracle already used by
  `tests/ops/test_linear.cpp` for Q, K, Gate, and V;
- compare every registered route directly with the oracle using its qualified BF16/SIMT or
  tensor-core tolerance;
- allocate the four outputs independently and protect both sides with canaries;
- check the 6143/6144 output seam and the last row explicitly;
- do not require old/new bitwise equality: combining rows can select a different legal reduction
  profile, especially for Q5 V at T=1;
- verify capture and replay on a non-default stream for a representative T=1 and MTP Small-T
  shape without allocation or synchronization inside capture.

Update `tests/ops/test_input_proj_plan.cpp`, `tests/ops/test_q4_linear_plan.cpp`,
`tests/ops/test_q5_linear_plan.cpp`, `tests/ops/test_q4_linear_candidates.cpp`,
`tests/ops/test_q5_linear_candidates.cpp`, `tests/ops/test_q4_linear_dispatch.cpp`, and the
affected sections of `tests/ops/test_linear.cpp`. The dispatch test must replace its current
N=7168 rejection with the new exact support; candidate tests numerically qualify every schedule
eligible for the sweep, including Q5 N=7168 GEMV and Split-K4. Plan tests assert two parent
subplans and exact admission; numerical tests establish the split mapping. Do not add a test whose
only purpose is to count private source calls; NSYS provides the implementation-topology evidence.

### 5.4 Performance qualification

Use `build/bench/ninfer_linear_op_bench` to sweep the new Q4/Q5 N=7168 candidate schedules at
T=1..16. Measure cold-cache and warm behavior, but choose the production route using the workload
that matches its use: T=1 ordinary decode and T=2..6 MTP verification are mandatory. Then measure
the complete production Op and the public Engine route.

Add one fixed-shape `ninfer_input_proj_bench` target in `bench/ops/input_proj_bench.cu`. Its
Attention cases use the same two packed parents and independently allocated outputs for both paths:

- production calls public `ops::attn_input_proj` and therefore launches two parent projections;
- a measurement-only control creates Q/K and Gate/V row views over those parents and invokes the
  former four independent Linear projections for T=1..16 only;
- cover T=1..16 with production/control latency, and T=17/128/129 as production-only regressions
  for the unchanged two-launch grouped regime; report per-case median/min/p95 under the existing
  cold-cache benchmark convention;
- keep the four-call control inside the benchmark only; it must not be callable from production
  dispatch or become a compatibility fallback.

`ninfer_linear_op_bench` selects the mainloop. `ninfer_input_proj_bench` qualifies the real
split-output epilogue and complete launch composition. This separation avoids inferring complete-Op
performance from two isolated Linear measurements.

Acceptance:

- the sum of the new Q4 and Q5 production projection latency is lower than the old four-projection
  composition for T=1 and each supported MTP verification T;
- no T=1..16 admitted point regresses enough to offset the launch reduction at the complete Op;
- post-change NSYS shows exactly one Q4 and one Q5 attention input-projection kernel per
  full-attention layer/pass;
- in the retained 33-forward `tg32` shape, 16 full-attention layers produce exactly 528 Q4 parent
  kernels and 528 Q5 parent kernels, with no independent Q/K/Gate/V projection kernels; relative
  to the four-call Small-T path this removes 1056 kernel events from the trace;
- ordinary and MTP end-to-end decode throughput do not regress.

Collect NCU only if the candidate sweep leaves a material route choice unresolved or a selected
N=7168 kernel regresses unexpectedly. In that case profile one exact kernel launch with basic
occupancy/SOL first, then memory and stall sections only if they can change the route decision.

## 6. Workstream C: GDN view-direct output and D2D removal

### 6.1 Required output mapping

Keep the existing `gdn_input_proj` semantic result:

```text
qkv [10240,T]
rows [0,4096)     = QK
rows [4096,10240) = V
```

Allocate `qkv` exactly as today and form two row views:

```cpp
Tensor qk    = qkv.slice(0, 0, 4096);
Tensor value = qkv.slice(0, 4096, 6144);
```

Because Tensor dimension 0 is contiguous, both views have:

```text
nb[0] = sizeof(BF16)
nb[1] = 10240 * sizeof(BF16)
```

For T=1 the column stride is unused and the offset pointers are sufficient. For T>1 the current
Small-T kernels must honor `out.nb[1]` instead of assuming `out.ne[0]` is the column leading
dimension.

### 6.2 Minimal kernel change

Do not create a new executor hierarchy or change public Linear semantics. Parameterize the existing
internal Small-T Q4/Q5 launch/store path with the output leading dimension in elements:

```cpp
out_ld = out.nb[1] / sizeof(__nv_bfloat16);
out[col * out_ld + row] = result;
```

Generic contiguous Linear calls pass `out_ld == out.ne[0]`. GDN passes the two sliced views above,
so QK writes at `qkv + 0` with ld=10240 and V writes at `qkv + 4096` with ld=10240.

Only the internal fixed launcher may admit this pitched form, and only for the SIMT routes changed
here. Its validation is:

```text
out.dtype == BF16
out.nb[0] == sizeof(BF16)
out.nb[1] % sizeof(BF16) == 0
out.nb[1] / sizeof(BF16) >= out.ne[0]
out.nb[1] / sizeof(BF16) <= INT32_MAX
output base has the route's existing alignment
```

MMA routes continue to require contiguous output. Public `ops::linear()` also continues to reject
non-contiguous output; the broader internal validation is reachable only through the admitted GDN
plan.

The affected registered Small-T routes are:

| Projection | T | Route | Store change |
|---|---:|---|---|
| Q4 QK `[4096,5120]` | 1 | GEMV R1/W8 | base pointer only; no second-column address |
| same | 2-4 | SIMT R8/C4 | use output ld |
| same | 5-16 | SIMT R8/C8 | use output ld |
| Q5 V `[6144,5120]` | 1 | GEMV R16/S2/X | offset base pointer only |
| same | 2-6 | Split-K4 exact | use output ld |
| same | 7-16 | SIMT R8/C8 | use output ld |

Do not change Q5 Split-K2 or MMA implementations that cannot be reached by this Small-T problem.
The existing T>=17 grouped mixed MMA route already carries output ld and row offset and remains
unchanged.

Primary files:

```text
include/ninfer/ops/gdn_input_proj.h
src/ops/wrapper/gdn_input_proj.cpp
src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h
src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
src/ops/linear/q4/q4_rowsplit_launch.cpp
src/ops/linear/q4/q4_rowsplit_gemm_simt.cu
src/ops/linear/q4/q4_rowsplit_gemm_simt.cuh
src/ops/linear/q5/q5_rowsplit_launch.cpp
src/ops/linear/q5/q5_rowsplit_gemm_simt.cu
src/ops/linear/q5/q5_rowsplit_gemm_simt.cuh
bench/ops/input_proj_bench.cu
bench/CMakeLists.txt
```

If a smaller internal overload can pass the view's existing stride without changing every launch
signature, prefer it. Do not add a generic strided tensor abstraction beyond what `Tensor::nb`
already represents.

### 6.3 Plan and workspace cleanup

Rename implementation-only vocabulary so it describes the new route:

```text
MaterializedFixed -> IndependentDirectFixed
plan.materialized -> plan.independent
```

Delete:

- the QK temporary tensor;
- the V temporary tensor;
- the transient workspace scope used only by them;
- both `cudaMemcpy2DAsync` calls;
- the nonzero Small-T workspace calculation.

All admitted `gdn_input_proj` routes then require zero transient bytes. Retain the existing
`WorkspaceArena&` boundary if that produces the smaller coherent change, but make its sizing query
validate the exact problem and return zero. Leave the existing target `alloc_bytes(0)` call in
`program/layouts.cpp`; it is already a no-op, so changing the target planner would only enlarge the
patch. Do not change the outer `qkv` allocation order or the `[10240,T]` layout required by causal
convolution.

### 6.4 Correctness gates

Test at least:

```text
T = 1, 2, 4, 5, 6, 7, 16, 17
```

Retain the existing T=128/129 grouped-MMA regression points; they prove that removing Small-T
workspace and copies did not disturb the unchanged large-T route.

For every point:

- compare QK and V directly with the independent FP64 row-split decode/dot-product oracle at real
  `[4096,5120]` and `[6144,5120]` shapes;
- compare the complete interleaved-per-token `qkv [10240,T]` output with the oracle;
- for T<=16 only, require supplementary bitwise equality with the old
  contiguous-projection-plus-copy result, place canaries around the parent allocation, and assert
  that the two pitched views share `nb[1]` and cover every output element exactly once;
- for T>=17, use the independent oracle/tolerance and existing grouped-route regression only; do
  not require equality with a separate Linear reduction profile;
- verify the plan reports zero workspace and the T=16/17 route boundary remains closed;
- capture and replay the Op on a non-default stream.

Update the GDN cases in `tests/ops/test_input_proj_plan.cpp` and
`tests/ops/test_linear.cpp`. No full-model numerical test substitutes for this direct output-mapping
check.

### 6.5 Performance and NSYS gates

The Q4 and Q5 projection kernel count and arithmetic must remain unchanged. Compare their kernel
durations before/after and reject any statistically meaningful regression; the only expected
topology change is removal of D2D work and its graph nodes.

The shared `ninfer_input_proj_bench` also carries the GDN operator measurement:

- production calls public `ops::gdn_input_proj` with the final `[10240,T]` output;
- a measurement-only control runs the same Q4/Q5 plans into two contiguous temporaries and performs
  the former two `cudaMemcpy2DAsync` operations into `[10240,T]` for T<=16 only;
- cover T=1, 2, 4, 5, 6, 7, and 16 with production/control latency, and T=17/128/129 as
  production-only regressions for the unchanged grouped route; separately report the projection-
  kernel sum;
- keep the materialize/copy control benchmark-only and allocate all buffers before timing/capture.

This benchmark is the operator-level before/after carrier. NSYS remains the authority for actual
Engine graph-node removal and copy attribution.

For the retained 33-forward `tg32` workload, the structural expectation is:

```text
8192-byte GDN QK copies:  1584 -> 0
12288-byte D2D copies:    3168 -> 1584
4096-byte D2D copies:     3168 -> 3168
all D2D events:           8017 -> 4849
Q4 QK kernel count:       unchanged
logical Q5 V launches:    unchanged at 1584
```

The remaining 12288-byte events belong to later V extraction and are outside this task. Confirm
the copy attribution from the NSYS SQLite event table rather than inferring it only from aggregate
memcpy time. This removes exactly `48 * 2 = 96` graph memcpy nodes per model forward. End-to-end
decode must not regress; report the observed improvement without promoting the previous estimate
to an acceptance threshold.

The raw N=6144 Q5 kernel family is shared by GDN V, GDN Z, and the old Attention Gate route. Its
integrated visible count is expected to change from 3696 to 3168 because Workstream B moves 528
Attention Gate launches into the new N=7168 parent kernel. Attribute logical GDN V with schedule/
layer context or the fixed-shape benchmark; do not misinterpret that aggregate-family count as a
GDN regression. Q4 GDN QK has a distinct kernel identity and remains directly countable at 1584.

## 7. Integrated implementation sequence

Execute in this order to avoid repeated 17.5 GB conversions and partially bound routes:

1. Record the existing canonical artifact inspection summary, retain the Release NSYS reports, and
   capture the matched five-repetition ordinary/MTP baselines from Section 3 before changing source.
   Do not copy or rename the old artifact.
2. Implement the new inventory, recipe, verifier, Python binding, C++ binding, active artifact
   documentation, and affected contract tests. Keep all version identifiers unchanged.
3. Implement GDN direct-view output and remove the temporary/copy path. Run synthetic Op tests.
4. Implement the two-parent Attention Op and N=7168 Q4/Q5 routes, plus the shared fixed-shape
   production/control input-projection benchmark. Run plan, oracle, candidate, and complete-Op
   benchmark tests.
5. Build the complete affected C++ targets in Release and finish all Python/unit checks before
   starting conversion.
6. Generate the explicit candidate artifact, inspect it, verify it against the BF16 source, and run
   Python reference and C++ real-artifact tests using the candidate path. If conversion itself
   fails or the product is invalid, remove the partial candidate, fix the cause, and regenerate at
   the same path; never promote a failed artifact.
7. Run ordinary and MTP correctness/parity checks, then the matched operator and end-to-end
   performance measurements. Use NSYS for topology/copy attribution; use NCU only if a live kernel
   route decision remains.
8. While both artifacts still exist, run the one-time 48-layer plane/view equality comparison.
9. After every gate passes, stage the canonical-path sidecar, replace the canonical artifact, then
   replace its sidecar with same-filesystem `mv -f`; do not retain the obsolete file. Inspect and
   smoke-load the canonical path.
10. Reconcile final measured facts into the active artifact/Op documentation, mark this plan
    complete, move it under `docs/archive/`, remove its active-plan link, and update
    `docs/archive/README.md`.

Do not regenerate the artifact merely to test converter syntax or counts. Those failures must be
caught by compact Python tests before attempting the large conversion.

## 8. Verification checklist

### 8.1 Source and compact tests

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
CANDIDATE=out/qwen3_6_27b_rtx5090.cutover.ninfer

$PYTHON -m py_compile \
  tools/convert/qwen3_6_27b_rtx5090/inventory.py \
  tools/convert/qwen3_6_27b_rtx5090/recipe.py \
  tools/convert/qwen3_6_27b_rtx5090/verify.py \
  tools/convert/qwen3_6_27b_rtx5090/convert.py \
  tools/reference/qwen3_6_27b_rtx5090/bindings.py

$PYTHON -m pytest -q \
  tests/targets/qwen3_6_27b/test_inventory.py \
  tests/targets/qwen3_6_27b/test_recipe.py \
  tests/targets/qwen3_6_27b/test_verify.py \
  tests/targets/qwen3_6_27b/test_convert.py \
  tests/targets/qwen3_6_27b/test_reference_ops.py

cmake --build build -j --target \
  ninfer_input_proj_plan_test \
  ninfer_q4_linear_candidate_test \
  ninfer_q4_linear_dispatch_test \
  ninfer_q4_linear_plan_test \
  ninfer_q5_linear_candidate_test \
  ninfer_q5_linear_plan_test \
  ninfer_linear_test \
  ninfer_bench_support_test \
  ninfer_qwen3_6_27b_prefix_real_test \
  ninfer-qwen3_6_27b-dump \
  ninfer_linear_op_bench \
  ninfer_input_proj_bench \
  ninfer_bench

ctest --test-dir build --output-on-failure -R \
  '^(ninfer_input_proj_plan_test|ninfer_q4_linear_candidate_test|ninfer_q4_linear_dispatch_test|ninfer_q4_linear_plan_test|ninfer_q5_linear_candidate_test|ninfer_q5_linear_plan_test|ninfer_linear_test|ninfer_bench_support_test)$'
```

The real-artifact tests run only after candidate generation. They intentionally use different
environment variables:

```bash
NINFER_QWEN3_6_27B_ARTIFACT="$CANDIDATE" \
  $PYTHON -m pytest -q tests/targets/qwen3_6_27b/test_bindings.py

NINFER_QWEN3_6_27B_WEIGHTS="$CANDIDATE" \
  ctest --test-dir build --output-on-failure \
    -R '^ninfer_qwen3_6_27b_prefix_real_test$'
```

The first variable belongs to the Python typed binding; the second belongs to the C++ Engine
fixture. A return-code-77 skip is not acceptable once the candidate exists. Invoke the built test
executable directly if CTest configuration prevents propagating the explicit environment.

### 8.2 Real artifact and reference

Run the independent Python Text/MTP reference and the existing Python-vs-C++ activation gate on
the candidate, using fresh dump directories:

```bash
IDS=248045,846,198,5834,248046,198
REF_DUMP=/tmp/ninfer-cutover-reference
CPP_DUMP=/tmp/ninfer-cutover-cpp

$PYTHON -m tools.reference.qwen3_6_27b_rtx5090 \
  --weights "$CANDIDATE" --ids "$IDS" --decode 2 --greedy \
  --activation-dump "$REF_DUMP" --dump-level layer

./build/tools/ninfer-qwen3_6_27b-dump \
  --weights "$CANDIDATE" --ids "$IDS" --decode 2 --greedy \
  --activation-dump "$CPP_DUMP" --dump-level layer

$PYTHON -m tools.parity.qwen3_6_27b_rtx5090.activations \
  "$REF_DUMP" "$CPP_DUMP"

$PYTHON -m tools.reference.qwen3_6_27b_rtx5090 \
  --weights "$CANDIDATE" --ids "$IDS" --decode 6 --greedy \
  --mtp-draft-tokens 5 --draft-head
```

Required candidate evidence:

- `tools.artifact.inspect --objects` shows 1124 objects, including exactly 48
  `gdn/value_z [12288,5120]` Q5 objects and no physical old names;
- converter verifier passes exact structure, resources, representative direct tensors, Q5 seam
  rows, and draft-head checks against the canonical BF16 checkpoint;
- Python `ArtifactBinding` reports 1118 tensors and 390 row views, exercises the existing Vision
  binding/residency policy test, and completes representative Text and MTP reference execution;
- C++ Engine load reports target `qwen3_6_27b_rtx5090`, 1118 tensors, and six resources;
- prefix reuse and a representative ordinary decode execute with CUDA Graph enabled;
- MTP verification executes at a representative Small-T point such as T=6;
- no artifact-name lookup, conversion, repacking, allocation, or D2D concat occurs in the C++
  Engine decode hot path. The independent Python reference may continue to express its logical
  GDN concatenation with `torch.cat`; it is not the optimized runtime.

### 8.3 Numerical criteria

- Artifact codec and seam verification are exact at the code and FP16-scale-word level.
- The one-time old/new V and Z plane comparison is byte-exact.
- Attention and GDN projection correctness use the independent FP64 decoded-weight oracle with the
  route-appropriate existing tolerance.
- GDN T<=16 direct-view output additionally matches the former materialize/copy output
  bit-for-bit as supplementary migration evidence; T>=17 remains tolerance-checked against the
  independent oracle.
- End-to-end activation parity uses the existing target tap criteria; generated-token identity is
  not a substitute for operator correctness.

### 8.4 Performance criteria

Record concise provenance for every retained result: RTX 5090, CUDA 13.1, Release build, explicit
artifact path, exact command/workload, and summarized measurements.

Required measurements:

1. Q4 and Q5 `[7168,5120]` candidate sweep for T=1..16, with mandatory attention to T=1 and
   T=2..6.
2. Complete `attn_input_proj` latency at T=1 and representative MTP Small-T shapes.
3. GDN input projection kernel latency before/after plus attributed D2D event counts.
4. Public-Engine ordinary `tg32` decode using the same profile-measured boundary as the retained
   baseline.
5. One representative MTP draft/verify workload with the same draft window and prompt/generation
   shape before and after.

The reproducible operator commands are:

```bash
mkdir -p profiles/bench/decode-projection-cutover-20260717

for route in q4-gemv-r1w8-direct q4-gemv-r4w1-direct; do
  ./build/bench/ninfer_linear_op_bench \
    --rows 7168 --k 5120 --qtype Q4 --candidate "$route" --t-sweep 1 \
    --csv-out "profiles/bench/decode-projection-cutover-20260717/${route}.csv"
done

for route in q4-simt-r8c4 q4-simt-r8c8; do
  ./build/bench/ninfer_linear_op_bench \
    --rows 7168 --k 5120 --qtype Q4 --candidate "$route" \
    --t-sweep 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
    --csv-out "profiles/bench/decode-projection-cutover-20260717/${route}.csv"
done

./build/bench/ninfer_linear_op_bench \
  --rows 7168 --k 5120 --qtype Q5 --candidate q5-gemv-r16s2x --t-sweep 1 \
  --csv-out profiles/bench/decode-projection-cutover-20260717/q5-gemv-r16s2x.csv

./build/bench/ninfer_linear_op_bench \
  --rows 7168 --k 5120 --qtype Q5 --candidate q5-simt-split4-exact \
  --t-sweep 2,3,4,5,6 \
  --csv-out profiles/bench/decode-projection-cutover-20260717/q5-split4.csv

for route in q5-simt-r8c4 q5-simt-r8c8; do
  ./build/bench/ninfer_linear_op_bench \
    --rows 7168 --k 5120 --qtype Q5 --candidate "$route" \
    --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
    --csv-out "profiles/bench/decode-projection-cutover-20260717/${route}.csv"
done

./build/bench/ninfer_input_proj_bench \
  --op all --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,128,129 \
  --warmup 5 --repeat 50 \
  --csv-out profiles/bench/decode-projection-cutover-20260717/input_proj.csv
```

`ninfer_input_proj_bench` must implement the CLI shown above as part of the benchmark addition.
The matched candidate Engine commands are the Section 3 commands with `--weights "$CANDIDATE"`
and `after_text_tg32.json` / `after_mtp_tg32_draft.json` output names.

Capture post-change topology with the exact structural workloads used by the retained reports:

```bash
mkdir -p profiles/nsys/decode-projection-cutover-20260717

nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output=profiles/nsys/decode-projection-cutover-20260717/after_text_tg32 \
  ./build/bench/ninfer_bench --weights "$CANDIDATE" \
    -n 32 --warmup 1 -r 1 --profile-measured --output json \
    --output-file profiles/nsys/decode-projection-cutover-20260717/after_text_tg32.json

nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output=profiles/nsys/decode-projection-cutover-20260717/after_mtp_tg32_draft \
  ./build/bench/ninfer_bench --weights "$CANDIDATE" \
    -pg 32,32 --mtp-draft-tokens 5 --lm-head-draft \
    --warmup 1 -r 1 --profile-measured --output json \
    --output-file profiles/nsys/decode-projection-cutover-20260717/after_mtp_tg32_draft.json

nsys export --type=sqlite --force-overwrite=true \
  --output=profiles/nsys/decode-projection-cutover-20260717/after_text_tg32.sqlite \
  profiles/nsys/decode-projection-cutover-20260717/after_text_tg32.nsys-rep

sqlite3 -header -column \
  profiles/nsys/decode-projection-cutover-20260717/after_text_tg32.sqlite \
  'SELECT bytes, COUNT(*) AS events, ROUND(SUM(end-start)/1e6,3) AS ms
   FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=8
   GROUP BY bytes ORDER BY bytes;'
```

Export/query the MTP report the same way when attributing T=2..6 behavior. Retain summarized
kernel topology and D2D histograms; raw NCU data is conditional, not a default deliverable.

The implementation passes when:

- Attention has two projection kernels per full-attention layer and is faster as a complete Op;
- GDN has unchanged projection work, zero input-concat copies, and no kernel regression;
- ordinary and MTP end-to-end decode do not regress;
- any claimed speedup is supported at the level claimed rather than inferred from launch count.

### 8.5 Static and promotion hygiene

Before promotion, run the active-source scan and inspect every remaining old-role match. Matches
may describe logical `value`/`z` views or this plan's before-state, but no converter physical spec,
Python physical contract, C++ `tensor(...)` binding, test expectation, or active artifact table may
require the old objects:

```bash
rg -n 'gdn/(value|z)' \
  tools/convert/qwen3_6_27b_rtx5090 \
  tools/reference/qwen3_6_27b_rtx5090 \
  src/targets/qwen3_6_27b_rtx5090 \
  tests/targets/qwen3_6_27b \
  docs/qwen3.6-27b-ninfer-artifact.md \
  docs/ninfer-container-format.md

$PYTHON -m tools.artifact.inspect "$CANDIDATE" --objects | \
  awk '$NF ~ /\/gdn\/value_z$/ { new_count++ }
       $NF ~ /\/gdn\/(value|z)$/ { old_count++ }
       END { exit !(new_count == 48 && old_count == 0) }'

git diff --check
```

After promotion, repeat inspect/verify/Engine load at the canonical path, remove only the cutover
partials identified below, then rerun the command and require it to print nothing:

```bash
find out -maxdepth 1 -type f \
  \( -name 'qwen3_6_27b_rtx5090.cutover*' \
     -o -name 'qwen3_6_27b_rtx5090*.partial' \
     -o -name 'qwen3_6_27b_rtx5090*.bak' \
     -o -name 'qwen3_6_27b_rtx5090*.old' \
     -o -name 'qwen3_6_27b_rtx5090*.next' \) -print
```

Do not delete unrelated artifacts or profiler outputs from `out/`; cleanup is limited to the
obsolete/candidate files created by this cutover.

## 9. Completion gate

The task is complete only when all statements below are true:

- source, converter, verifier, Python reference, C++ binder, tests, and active documents require
  the new physical artifact contract and no compatibility route exists;
- the candidate was generated from the canonical BF16 checkpoint, verified, and exercised through
  Python and the public C++ Engine;
- the old and new V/Z planes were compared before deletion;
- the canonical artifact and sidecar were replaced, the obsolete artifact was unlinked, and no
  backup/alternate artifact remains;
- Attention executes two qualified projection kernels for decode/Small-T;
- GDN writes through final-buffer views, allocates no concat scratch, and launches no concat D2D;
- direct numerical checks, CUDA Graph capture/replay, ordinary decode, and MTP verification pass;
- matched performance evidence shows the required topology and no end-to-end regression;
- final stable facts are integrated into active authorities and this completed plan is archived.
