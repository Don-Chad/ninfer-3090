# Linear Kernel Architecture Refactor

> Status: accepted design; implementation pending.
>
> Date: 2026-07-16.
>
> Scope: repository-internal implementation architecture for the pure `Linear` operation
> `Y = W X`. This plan is independent of any one future target adaptation. It replaces the current
> incremental organization of Linear kernels with a coherent, extensible, zero-overhead kernel
> system while retaining finite registration and exact-target performance qualification.
>
> Authority boundary: [`../op-development.md`](../op-development.md) and
> [`../../include/ninfer/ops/linear.h`](../../include/ninfer/ops/linear.h) remain authoritative for
> Op ownership and mathematical semantics. This document is the implementation authority for the
> active Linear refactor. It does not change persistent numeric formats, storage layouts, target
> registration, or the public Engine API.

## 1. Decision

NInfer will refactor `src/ops/linear/` into a small compile-time kernel system built from:

1. physical weight-format and layout definitions;
2. path-specific weight decode atoms;
3. a small number of execution mainloops;
4. schedule, tile, pipeline, and epilogue policies;
5. explicitly instantiated and finitely dispatched production policies.

The desired result is not one universal Linear kernel. It is:

```text
few stable mainloops
  x few registered storage/decode families
  x few measured schedule policies
  x explicit production instances
```

The system must accumulate useful implementation knowledge. Adding a numerical shape should
normally add or select a policy instance rather than copy a complete kernel. Adding a weight format
that belongs to an existing packing family should normally add format/decode behavior rather than
duplicate GEMV and GEMM mainloops. Adding a future activation representation or compute mode should
not require redefining the weight format.

The refactor accepts substantial source movement and kernel rewrites. It does not accept a
performance regression merely to make source code appear uniform. An exact specialization remains
valid when its measured dataflow genuinely differs; it must be expressed as a policy or explicit
custom implementation behind the same plan and ownership seams.

## 2. Why the refactor is now justified

The original Linear implementation was built while there was insufficient kernel evidence to know
which abstractions were real. The repository now contains enough independent implementations and
measurements to identify stable similarities and necessary differences:

- Q4G64, Q5G64, and Q6G64 share a low-nibble plane, optional split high-bit plane, FP16 group
  scales, signed reconstruction, and the same logical G64 row-split traversal;
- W8G32 uses a simpler signed-byte code plane but a different group size, scale cadence, vector
  mapping, and resource balance;
- dense BF16 and FP32 weights need no packed-weight decode but still participate in the same
  scheduling and output semantics;
- single-token GEMV, bundled Small-T work, and tiled Large-T GEMM have materially different
  accumulator, launch-coverage, and data-reuse requirements;
- the current Q4/Q5/Q6 Large-T implementation already proves a shared codec-parameterized MMA
  mainloop;
- the current Small-T implementation already proves a shared format-parameterized streaming
  mainloop;
- the current W8 Large-T work proves that a common mathematical pipeline does not imply a common
  optimal warp-role schedule;
- the current T1 kernels expose the maintenance cost of encoding caller roles and exact shapes in
  whole source files rather than composing a smaller set of schedule policies.

This is the point at which further copy-based specialization would create avoidable fragmentation,
while an earlier attempt at the same design would have been speculative.

## 3. Goals

### 3.1 Primary goals

- Give every pure Linear implementation the same internal vocabulary and dispatch boundary.
- Separate physical weight representation from activation representation and compute mode.
- Express Q4/Q5/Q6 as one split-low4 storage family with compile-time bit width.
- Reuse stable T1, Small-T, and Large-T mainloops across compatible formats and shapes.
- Preserve a direct path for format- or shape-specific scheduling where measurements require it.
- Make current A16 execution explicit without hard-wiring A16 into the weight codec.
- Leave a clean architectural extension point for a future A8 compute mode without adding unused
  A8 implementation now.
- Allow semantically different fused Ops to reuse Linear implementation components later without
  changing the base `linear()` contract.
- Replace caller-role naming with operation-structural names and predicates.
- Keep registration finite, allocation-free, graph-capture friendly, and specialized to compiled
  targets.

### 3.2 What successful extension looks like

After the refactor:

| Change | Expected implementation surface |
|---|---|
| New `(N,K)` using an existing format and regime | add/select a named policy instance and plan predicate |
| New bit width compatible with split-low4 packing | add/instantiate format traits and decode atoms; reuse compatible mainloops |
| New row-split byte format | add a format family and its path-specific decode; reuse schedules where measured appropriate |
| New Large-T tile | add a tile/pipeline policy, not a copied mathematical kernel |
| New T1 ownership strategy | add a schedule policy, not a new format implementation |
| New A8 path | add an activation format and compute mode, plus required mainloop/policies; do not redefine W4 storage |
| New fused semantic Op | add its own Op contract/wrapper and an exact epilogue/output policy over reusable Linear internals |

## 4. Non-goals

This refactor will not create:

- a general-purpose BLAS, CUTLASS replacement, or externally consumable GEMM library;
- a runtime-pluggable kernel registry;
- virtual kernel, codec, schedule, or epilogue interfaces;
- a graph IR, generic model graph, target discovery mechanism, or string-driven dispatch;
- an unconstrained template Cartesian product of every format, activation, tile, schedule, and
  epilogue;
- a promise that one kernel performs well for arbitrary dimensions;
- hidden runtime weight repacking or a second persistent weight copy;
- a requirement that all formats use identical warp roles or shared-memory allocation;
- placeholder A8 code, speculative scale semantics, or an unqualified W4A8 product path;
- fused-Op semantic changes disguised as an internal Linear refactor;
- source or API compatibility with obsolete repository-internal Linear implementation names.

Generic correctness is not production support. A domain is supported only when an exact selected
policy has the required numerical and RTX 5090 performance evidence.

## 5. Locked semantic and storage invariants

The base Op remains:

```text
x   : contiguous BF16 [K,T]
w   : registered logical weight [N,K]
out : contiguous BF16 [N,T]

out[n,t] = BF16(sum_k dequantize(w)[n,k] * float(x[k,t]))
```

The following invariants remain locked throughout the refactor:

1. `linear()` owns only the pure matrix projection and final BF16 output boundary.
2. Weight logical semantics come from `ninfer-tensor-formats.md`; byte addressing comes from
   `ninfer-storage-layouts.md` and the bound `Weight` metadata.
3. The kernel consumes device-resident weight planes directly. It does not allocate or repack
   persistent data.
4. Accumulation grouping may differ by selected backend as already allowed by the Op contract.
5. Runtime targets and caller tensor names do not participate in Op dispatch.
6. Shape and layout assumptions are selected explicitly by the wrapper/plan and encoded in the
   policy instance.
7. The caller-owned workspace remains an implementation resource, not semantic state.

## 6. Architectural axes

The internal design treats the following as separate axes:

```text
LinearProblem
├── WeightFormat       logical numeric reconstruction and plane geometry
├── WeightLayout       physical addressing and tile traversal
├── ActivationOperand  BF16 execution operand today; an explicitly scaled A8 operand may exist later
├── ComputeMode        operand conversion, MMA/FMA atom, accumulator, scale composition
├── Mainloop           T1 GEMV / Small-T / Large-T GEMM
├── Schedule           warp/CTA ownership, split-K, tile raster, reduction
├── Pipeline           movement, staging, synchronization, producer/consumer roles
└── Epilogue           exact output mapping and rounding boundary
```

Not every axis becomes a runtime enum. Most are compile-time members of a named production policy.
The host plan key contains only facts needed to choose among the explicitly instantiated policies.

`ActivationOperand` describes the representation consumed by the selected execution mainloop; it
does not silently widen the public Op input contract. The current source tensor remains BF16. A
future A8 policy under the same Linear semantics would have to define where BF16 is quantized and
where its scales live. Accepting an already-quantized input tensor would require an explicit
semantic/internal Op contract rather than reinterpretation by this dispatcher.

### 6.1 Why the separation matters

Q5G64 does not imply a particular GEMV schedule. It describes how Q5 values and scales are stored
and reconstructed. A T1 row-warp kernel, T1 split-K kernel, Small-T bundled kernel, and Large-T MMA
kernel may all consume Q5G64 through different optimized decode atoms.

Likewise, W4A16 and a possible W4A8 path may share weight storage but differ in activation storage,
MMA atom, accumulator dtype, scale application, and tile alignment. Those differences belong to
activation and compute mode, not to the W4 weight-format definition.

## 7. Host-side problem and plan model

### 7.1 `LinearProblem`

The wrapper constructs a small non-owning internal problem view after semantic validation. Its
conceptual fields are:

```cpp
struct LinearProblem {
    Tensor x;
    const Weight* weight;
    Tensor out;
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    std::int32_t padded_k;
};
```

The exact C++ representation may avoid copying `Tensor`, but it must contain only already-validated
execution facts. It does not contain a target key, tensor role, layer index, or source tensor name.

### 7.2 Plan key

The new plan key is structural:

```cpp
struct LinearPlanKey {
    LinearFormat format;   // registered qtype + physical layout class
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
};
```

Plan resolution derives `T1 / SmallT / LargeT` from `t` and the measured candidates for the exact
format and shape. Keeping exact `t` in the key also permits a selected `T=2..6` direct policy or a
different Small-T token tile without hiding a second dispatch inside the launcher. The key does not
use caller-role `ShapeFamily` values such as `MtpFc`, `AttnIn`, or `GdnQK`. Benchmark labels may
retain descriptive call-site context, but it has no behavioral role.

An exact shape may receive a named structural policy such as `W8A16N5120K10240LargeT`; the policy
name states format, compute mode, numerical shape, and regime rather than model use.

### 7.3 Plan resolution

Plan resolution remains direct and finite:

```text
validated problem
  -> classify registered format/layout
  -> classify T regime using measured format/shape crossover
  -> exact structural policy predicates
  -> finite LinearPolicyId
  -> switch to one explicit launcher
```

There is no dynamic registration or device function-pointer table. The plan implementation may use
compact `constexpr` metadata for matching, followed by a `switch` over `LinearPolicyId`.

Regime crossover is a measured property of `(format,N,K,schedule candidates)`, not a universal
statement that GEMM is compute-bound. An MMA policy may win before the mathematical contraction
crosses the compute roofline. Conversely, a small or tail-heavy GEMM may remain memory-, launch-,
or dequantization-bound well into the Large-T range.

## 8. Weight formats and layouts

### 8.1 Responsibilities

A weight-format definition owns only:

- logical group width;
- code/high/scale plane geometry;
- code interpretation and sign reconstruction;
- scale dtype and logical application;
- compile-time constants needed by tile loaders and decode atoms.

A weight-layout definition owns only:

- plane address calculation;
- row/group/tile strides;
- alignment and padding interpretation;
- the mapping from a logical tile to physical byte ranges.

Neither owns warp assignment, token tiling, reduction, MMA fragments, or the output epilogue.

### 8.2 Split-low4 family: Q4G64, Q5G64, Q6G64

Q4G64, Q5G64, and Q6G64 become instantiations of one compile-time format family:

```cpp
template <int Bits, int GroupK>
struct SplitLow4WeightFormat;

using Q4G64 = SplitLow4WeightFormat<4, 64>;
using Q5G64 = SplitLow4WeightFormat<5, 64>;
using Q6G64 = SplitLow4WeightFormat<6, 64>;
```

For one group of 64 values:

| Format | low plane | high plane | FP16 scale |
|---|---:|---:|---:|
| Q4G64 | 32 B | 0 B | 2 B |
| Q5G64 | 32 B | 8 B | 2 B |
| Q6G64 | 32 B | 16 B | 2 B |

For bit width `B`, logical reconstruction is:

```text
u = low4 | (high << 4)
q = (u xor 2^(B-1)) - 2^(B-1)
w = float(q) * float(fp16_scale)
```

The format family derives high bits per value, high bytes per group, sign bit, bias constants, and
plane-presence decisions at compile time. Hot decode atoms may use format-specific instruction
sequences, including the existing FP16 mantissa construction, but those specializations live under
the common format family rather than in complete caller-specific kernels.

### 8.3 Signed-byte family: W8G32

W8G32 is a separate storage family:

```cpp
template <int GroupK>
struct SignedByteWeightFormat;

using W8G32 = SignedByteWeightFormat<32>;
```

Each group stores 32 signed bytes and one FP16 scale. It shares the format/decode vocabulary but
does not pretend to have a split-low4 high plane. Its different group cadence and half-warp decode
mapping remain visible to policy selection.

### 8.4 Dense formats

Dense BF16 and FP32 weight formats use contiguous row-major values and no group scale:

```cpp
struct DenseBf16WeightFormat;
struct DenseFp32WeightFormat;
```

They participate in the same `LinearProblem`, plan, schedule, compute-mode, and epilogue model.
They do not execute no-op quantization branches inside a low-bit codec.

### 8.5 Path-specific decode atoms

There is deliberately no requirement that every format expose one universal `decode()` method.
Each mainloop consumes an interface suited to its dataflow:

```text
SIMT T1       -> decode the lane-owned pair/vector to FP32 accumulable values
SIMT Small-T  -> decode a staged vector once and reuse it across token accumulators
BF16 MMA      -> decode a raw tile to a swizzled BF16 shared-memory operand
future A8     -> unpack/convert into that compute mode's integer operand representation
```

The implementation should express these as narrow compile-time adapters such as
`SimtWeightDecode<Format>` and `Bf16MmaWeightDecode<Format>`. A format can specialize an adapter
when its optimal instruction sequence differs. Mainloops do not duplicate bit reconstruction or
plane geometry.

## 9. Activation formats and compute modes

### 9.1 Current A16 modes

Current production Linear policies use BF16 activations and one of two compute modes:

```text
SimtA16F32:
  packed/dense weight -> FP32 value
  BF16 activation     -> FP32 value
  FP32 FMA/reduction  -> BF16 epilogue

MmaA16F32:
  packed/dense weight -> BF16 MMA operand
  BF16 activation     -> BF16 MMA operand
  BF16 mma.sync       -> FP32 accumulator -> BF16 epilogue
```

The weight format does not mention A16. A named policy combines a weight format with one of these
compute modes.

### 9.2 Future A8 boundary

A future W4A8 implementation is a new activation and compute-mode design, not another W4 codec
method. It may require:

- an explicit INT8 activation representation and scale contract;
- a defined BF16-to-INT8 preparation boundary if the public source tensor remains BF16;
- integer MMA and INT32 accumulation;
- weight-scale and activation-scale composition at a defined granularity;
- a different K tile and operand layout;
- a separately registered persistent weight layout if native MMA cannot efficiently consume the
  current row-split layout.

No such choices are made by this plan. The only locked requirement is that current interfaces do
not bind W4 storage to one execution operand representation. The existing `linear()` input remains
BF16 unless its semantic contract is deliberately revised. A future implementation must add exact
preparation, scale, workspace, and output semantics rather than activate dormant placeholders.

## 10. Mainloop families

The refactor defines three primary execution mainloops for quantized A16 Linear and one dense
route. They share format, layout, policy, and epilogue vocabulary but retain materially different
dataflows.

### 10.1 T1 GEMV mainloop

The T1 mainloop is bandwidth- and launch-coverage oriented:

```text
stream packed/dense weight
  -> decode lane-owned values to FP32
  -> multiply the reused BF16 activation vector
  -> accumulate/reduce in FP32
  -> epilogue one output column
```

Schedule policies include, when instantiated and measured:

- `WarpPerRow`: one warp owns one complete output row;
- `RowsPerWarp`: one warp owns several small-K rows when instruction balance permits;
- `SplitKPerRow<Warps>`: several warps own K partitions of one row and reduce partials;
- `RowStripe<Rows,KSplits>`: a CTA cooperatively loads a physical row stripe and computes several
  rows;
- `DualWeight`: two equal-shape weights reuse the same activation traversal while retaining
  independent accumulators and outputs.

These are schedule policies, not format names. Q4, Q5, Q6, and W8 may select different schedules
for the same `(N,K)` because packed bytes, decode instruction count, launch waves, and scale traffic
differ.

A fully custom T1 policy remains possible for an exceptional domain, but it must reuse common
format/layout semantics where applicable and be selected as an explicit structural policy.

### 10.2 Small-T mainloop

The Small-T mainloop streams one weight tile and updates a compile-time token tile:

```text
load/stage packed weight once
  -> decode once
  -> update Ttile independent FP32 accumulators
  -> reduce and store valid columns
```

It remains separate from T1 because token-vector registers, activation access, column predicates,
and the useful weight-reuse/occupancy tradeoff are different. T1 and Small-T may share decode and
warp-reduction primitives without sharing one branch-heavy kernel.

Named policies choose a small set of token tiles, initially preserving the measured `Ttile=4/8`
design and any exact `T=2..6` direct instances that remain beneficial. Larger token tiles are not
generated unless their register and occupancy behavior is measured to win.

### 10.3 Quantized-to-BF16 Large-T MMA mainloop

The Large-T A16 mainloop has a common logical CTA lifecycle:

```text
raw weight planes --WeightProducer--> swizzled BF16 As
BF16 x            --ActivationProducer--> swizzled BF16 Bs
As, Bs            --Bf16MmaConsumer--> FP32 accumulator tile
accumulator       --Epilogue--> output
```

The common machinery owns:

- CTA/warp tile coordinates;
- stage lifecycle and synchronization contract;
- shared-memory operand descriptors and swizzle vocabulary;
- `ldmatrix` and `mma.sync.m16n8k16` fragment mapping;
- accumulator traversal;
- full-tile versus edge-tile output mapping;
- epilogue invocation.

The weight producer owns:

- raw plane copies for its format/layout;
- scale staging/cache policy;
- dequantization work distribution;
- conversion into the BF16 shared operand;
- format-dependent shared-memory resources.

This structure allows Q4/Q5/Q6 to share one split-low4 producer family, W8 to use its signed-byte
producer, and dense BF16 to use a direct producer. It does not require them to allocate unused
planes or use identical warp ownership.

### 10.4 Dense mainloops

Dense T1 uses a dedicated dense GEMV mainloop because it has no packed decode. Dense Large-T uses
the common BF16 MMA consumer with a direct BF16 producer where beneficial. Dense FP32 may use a
separate SIMT or conversion policy according to its exact supported domains.

The current generic dense CUDA implementation remains a numerical reference until exact dense
production policies replace it. It is not considered high-performance support.

## 11. Schedule, tile, and pipeline policies

### 11.1 Policy bundling

Kernel entry points take one named policy rather than exposing a long arbitrary template argument
list at every call site:

```cpp
template <class Policy>
__global__ void linear_kernel(...);

struct Q5A16N5120K17408T1Policy {
    using WeightFormat = SplitLow4WeightFormat<5, 64>;
    using WeightLayout = RowSplitK128Layout;
    using ComputeMode  = SimtA16F32;
    using Mainloop     = T1GemvMainloop;
    using Schedule     = /* measured structural schedule */;
    using Epilogue     = StoreBf16;
};
```

The exact class names may be shortened in implementation, but production policy identity must make
format, compute mode, numerical domain, and regime discoverable without encoding a model role.

Policy definitions provide compile-time resource and validity assertions:

- tile divisibility and edge behavior;
- format group and BK compatibility;
- vector alignment;
- shared-memory footprint;
- thread/warp count;
- accumulator and fragment mapping;
- pipeline stage count;
- supported output topology.

### 11.2 No combinatorial registry

Reusable templates may technically accept many combinations, but only named, explicitly
instantiated policies enter the production binary. Build files enumerate those instance TUs.
Unsupported combinations do not acquire a runtime entry merely because their templates compile.

### 11.3 Warp specialization

Warp specialization is a pipeline policy, not the Linear architecture itself.

The current SM120 production default remains the measured cooperative pipeline in which the same
warps participate in movement/dequantization and MMA phases. A future policy may assign dedicated
TMA/copy/dequant producers and MMA consumers when a specific Large-T domain shows remaining
producer/consumer overlap headroom.

The W8 Large-T evidence is binding: the attempted `4 producer + 8 consumer` organization lost to
the final eight-warp cooperative implementation. The new architecture must be capable of
representing both, but it must not replace the winning cooperative policy with warp specialization
for stylistic uniformity.

Likewise, T1 and Small-T kernels do not adopt producer/consumer warp specialization by default.
Their primary constraints are weight bandwidth, packed decode, register count, and launch
coverage. A schedule is selected from measured dataflow, not architecture fashion.

### 11.4 Persistent scheduling and TMA

Persistent tile scheduling, TMA tensor maps, ping-pong consumers, and cooperative consumers are
valid future pipeline policies. They are introduced only for an exact domain whose benchmark and
NCU evidence identifies the problem solved by that machinery. They are not prerequisites for this
refactor and do not enter the initial policy set as unused abstractions.

## 12. Epilogue boundary

The base Linear refactor initially implements exactly one epilogue:

```cpp
struct StoreBf16;
```

It maps accumulator fragment elements to contiguous `[N,T]` output and applies the selected
backend's BF16 output boundary.

The epilogue interface exists because the accumulator traversal and output mapping should not be
copied into every mainloop, and because semantically closed Ops may later reuse the same internal
mainloop. That future reuse does not merge Op contracts:

- `LinearAdd` remains a separate Op with its exact residual rounding/order contract;
- `LinearSwiGLU` remains a separate Op with split and activation semantics;
- multi-output projection Ops retain their declared output views;
- expert-grouped Linear remains private to its closed sparse Op schedule.

Those Ops may instantiate a shared mainloop with an exact compile-time epilogue only when the full
semantic operation and numerical boundary are preserved. They do not add an epilogue runtime axis
to `linear()`.

## 13. Source organization

The intended source tree is:

```text
src/ops/linear/
├── linear.cpp                         # validation + finite policy dispatch
├── problem.h                          # small validated internal problem values
├── format/
│   ├── split_low4.cuh                 # Q4/Q5/Q6 storage traits and reconstruction
│   ├── signed_byte.cuh                # W8 storage traits and reconstruction
│   ├── dense.cuh                      # dense BF16/FP32 traits
│   └── rowsplit_layout.cuh            # registered row-split addressing
├── core/
│   ├── epilogue.cuh                   # StoreBf16 and later exact internal epilogues
│   ├── fragments.cuh                  # Linear-local fragment traversal
│   ├── pipeline.cuh                   # Linear-local stage/pipeline policies
│   └── tile.cuh                       # tile/resource configuration values
├── gemv/
│   ├── t1_mainloop.cuh
│   ├── smallt_mainloop.cuh
│   ├── schedules.cuh
│   └── instances/                     # explicit T1/Small-T production instances
├── gemm/
│   ├── bf16_mma_mainloop.cuh
│   ├── producers.cuh                  # split-low4, W8, and dense BF16 producers
│   ├── schedules.cuh
│   └── instances/                     # explicit Large-T production instances
├── plan/
│   ├── linear_plan.h
│   └── linear_plan.cpp
└── reference/
    ├── linear_reference.h
    └── linear_reference_dense.cu
```

Names may be adjusted during implementation to avoid empty or one-use files, but responsibilities
and dependency direction are fixed:

```text
format/layout + core primitives
              ↓
        mainloop/schedule
              ↓
       explicit instances
              ↓
       plan + linear.cpp
```

`src/ops/common/` continues to own only narrow cross-Op CUDA primitives such as basic memory,
warp, and MMA instructions. Linear-specific format, swizzle, tile, stage, and fragment policy stays
inside `src/ops/linear/`; it is not promoted to repository common merely because multiple Linear
formats use it.

## 14. Current implementation mapping

The current code maps into the new design as follows:

| Current implementation | Destination concept |
|---|---|
| `codec/linear_codec.cuh` Q4/Q5/Q6 | split-low4 format plus path-specific decode adapters |
| `linear_rowsplit_gemm_smallt.*` | Small-T mainloop, format adapters, and explicit instances |
| `linear_rowsplit_gemv_q5_core.cuh` | T1 mainloop plus Q5 policies/schedules |
| caller-role Q4 T1 files | Q4 format adapter plus structural T1 schedule instances |
| lm-head Q4/Q6 file | large-N T1 schedule instance(s), not an `lm_head` arithmetic family |
| `linear_rowsplit_gemm_mma.*` | shared A16 BF16-MMA mainloop plus split-low4 producer |
| `linear_rowsplit_w8g32_gemm_mma.*` | shared MMA consumer/lifecycle where zero-cost, W8 producer and policy |
| dense reference kernels | retained reference, then replaced by explicit dense production policies |
| `ShapeFamily` caller-role enum | structural `(N,K)` matching and named numerical policies |

Migration must delete obsolete caller-role launchers and duplicate codec logic once their selected
production routes move. The repository does not keep compatibility wrappers for deleted internal
names.

## 15. Relationship to grouped expert work

Expert-grouped contractions share arithmetic building blocks but not the ordinary Linear schedule.
Their dynamic expert selection, per-expert `M_e`, pointer/stride description, tile scheduling, and
reduction belong to the closed sparse Op implementation.

Grouped expert kernels may reuse:

- registered weight formats and layouts;
- decode adapters;
- MMA atoms and fragment helpers;
- tile/resource policy vocabulary;
- exact epilogue components.

They do not call the ordinary `linear()` wrapper per expert and do not force dynamic grouped
metadata into `LinearPlan`.

## 16. Migration plan

The refactor proceeds by working implementation slices. Each slice replaces and deletes its old
path before the next family is migrated.

### Phase 1: foundation and format consolidation

1. Add the new format/layout/core directories and the minimal policy vocabulary.
2. Implement the split-low4 format family for Bits 4/5/6.
3. Implement W8 and dense format traits.
4. Add path-specific decode adapters required by the first migrated path.
5. Keep existing public Op and plan behavior while the first new instances are introduced.

Exit condition: Q4/Q5/Q6 logical reconstruction has one source of truth per decode target; no new
mainloop duplicates plane geometry or sign reconstruction.

### Phase 2: Small-T as the first complete vertical slice

1. Re-express the existing Small-T kernel as the new Small-T mainloop.
2. Instantiate Q4/Q5/Q6/W8 policies for the currently selected token tiles.
3. Preserve direct exact-T policies only where they remain measured winners.
4. Replace plan dispatch and delete the old Small-T kernel/codec duplication.

Small-T is first because it already demonstrates cross-format reuse and exercises format, layout,
decode, schedule, policy, launch, and epilogue seams without first changing the most fragmented T1
paths.

### Phase 3: T1 GEMV family

1. Build common T1 mainloop and schedule policies.
2. Migrate the existing Q5 core and its shape configurations.
3. Migrate Q4 direct, row-stripe, and split-K implementations into structural policies.
4. Migrate large-N Q4/Q6 output projection policies.
5. Migrate W8 T1 through the same vocabulary and add special schedules only where measured.
6. Delete caller-role files and policy IDs as each domain moves.

Exit condition: a T1 production route is described by format + numerical shape + schedule policy;
no arithmetic family exists solely because the caller calls its rows Q, K, V, gate, MLP, or head.

### Phase 4: Large-T BF16 MMA family

1. Rebuild the Q4/Q5/Q6 GEMM around the shared BF16-MMA mainloop and split-low4 producer.
2. Express existing tile/config choices as named policies.
3. Integrate W8 through the shared lifecycle/consumer/epilogue only where the generated hot path is
   equal to the current dedicated kernel; retain a W8-specific producer and resource policy.
4. Add a direct dense BF16 producer and explicit dense GEMM policies.
5. Delete superseded Large-T source duplication.

Exit condition: Q4/Q5/Q6, W8, and dense BF16 use one architectural vocabulary; any remaining
separate kernel has a measured, documented reason rather than historical source isolation.

### Phase 5: plan and source cleanup

1. Replace caller-role `ShapeFamily` dispatch with structural keys and policies.
2. Consolidate explicit instance declarations and build registration.
3. Remove obsolete launch headers, policy names, comments, and archived-path references from active
   documentation.
4. Update `op-development.md` with the implemented final tree and permanent internal boundaries.
5. Archive this plan after all selected existing production paths use the new architecture.

### Phase 6: fused consumers, separately scoped

After the pure Linear architecture is complete, evaluate `LinearAdd`, `LinearSwiGLU`, paired
projections, and multi-output projections one Op at a time. Reuse a mainloop/epilogue seam only when
it reduces duplication without changing the semantic contract or the measured winning schedule.
This phase is not required to declare the pure Linear refactor complete.

## 17. Correctness and performance gates

### 17.1 Numerical verification

Each migrated mainloop/format combination uses the existing independent Linear oracle at relevant
real shapes and regime boundaries. Verification follows the current Op contract:

- exact format reconstruction where exact comparison is meaningful;
- the established numerical criterion for SIMT FP32 accumulation;
- the established normwise criterion for BF16-MMA weight rounding;
- explicit full-tile and real padded-K edge domains used by registered artifacts.

The refactor does not add broad tests for template shape, class construction, enum values, or source
organization. Tests protect numerical Linear behavior and selected real domains.

### 17.2 Performance preservation

For a previously roofline-qualified production policy, its replacement must retain the same
limiting-resource interpretation and match or improve representative benchmark performance within
normal measurement noise. If a generalized policy loses materially to the existing implementation,
the architecture must express the winning specialization; the regression is not accepted as an
abstraction cost.

For a newly generalized domain:

- a generic correct route is useful implementation infrastructure but is not support;
- support requires an exact selected policy and retained RTX 5090 evidence;
- GEMV is judged against the relevant cold/warm weight-traffic roofline and launch-coverage limits;
- Large-T GEMM is judged against measured useful throughput plus NCU compute/memory evidence;
- Small-T is judged against the best relevant weight-streaming and MMA candidates rather than an
  assumed backend label.

Run the smallest evidence set that proves each migration slice. NCU is used when a replacement
misses its gate or a new roofline claim needs hardware-counter attribution, not as ceremony for
every source movement. Whole-inference NSYS is required only when a migrated high-impact family
could alter end-to-end attribution or launch composition.

### 17.3 Build and integration properties

Every production policy must remain:

- allocation-free during launch;
- free of hidden host/device synchronization;
- compatible with stable CUDA Graph addresses;
- explicit about workspace use;
- directly compiled for the registered architecture;
- free of runtime target-name or tensor-role dispatch.

## 18. Design constraints for maintainability

The implementation must follow these constraints:

1. Prefer policy composition over inheritance.
2. Use static interfaces and `if constexpr`; do not use virtual dispatch.
3. Bundle named policies so launch sites do not repeat long template argument lists.
4. Keep hot-path format decisions compile-time.
5. Do not instantiate unsupported combinations.
6. Keep format math in one place per decode target.
7. Keep mainloop code independent of model caller roles.
8. Keep exceptional specializations visible in the plan registry.
9. Do not promote Linear-specific helpers into `src/ops/common/` without a second non-Linear Op
   consumer and a genuinely format-independent contract.
10. Treat compile time, binary size, SASS inspectability, and profiler kernel-name readability as
    real design constraints.

## 19. Locked decisions and tuning freedoms

### 19.1 Locked by this plan

- one pure Linear semantic Op;
- structural rather than caller-role dispatch;
- separation of weight format, layout, activation, compute mode, mainloop, schedule, pipeline, and
  epilogue;
- one split-low4 family for Q4/Q5/Q6;
- separate signed-byte and dense format families;
- distinct T1, Small-T, and Large-T mainloops;
- named policy bundles and explicit instantiation;
- cooperative and warp-specialized pipelines represented as policies rather than architecture-wide
  choices;
- no hidden repack, dynamic plugin registry, or universal arbitrary-shape promise;
- performance-preserving specialization when a common policy loses.

### 19.2 Left to measurement

- exact T-regime crossovers per format/shape;
- warp/CTA ownership and split-K factors;
- BM/BN/BK and warp tiles;
- cp.async versus TMA for a particular Large-T producer;
- stage count, cache policy, shared swizzle, and fragment double buffering;
- persistent versus static tile scheduling;
- which exact domains justify a custom policy;
- whether a fused Op benefits from reusing the final mainloop seam.

These are tuning freedoms inside the architecture, not reasons to bypass it.

## 20. Completion criteria

The refactor is complete when:

1. all currently selected pure Linear production paths dispatch through structural policy IDs;
2. Q4/Q5/Q6 storage and reconstruction are represented by the split-low4 family;
3. T1, Small-T, and Large-T implementations use the new mainloop/policy organization;
4. W8 and dense formats participate in the same architectural vocabulary without forced hot-path
   uniformity;
5. obsolete caller-role kernel files, policies, and compatibility launchers are deleted;
6. current exact 27B Linear domains retain numerical behavior and performance qualification;
7. active `op-development.md` reflects the implemented stable source boundary;
8. this plan is moved to `docs/archive/` and removed from the active-plan index.

## 21. Evidence references

The following archived documents are evidence, not current authority:

- [`../archive/optimization-era/m3-linear-backend-framework.md`](../archive/optimization-era/m3-linear-backend-framework.md) — original format/shape/regime plan seam and GEMV/GEMM split;
- [`../archive/optimization-era/2026-07-01-prefill-linear-foundation-design.md`](../archive/optimization-era/2026-07-01-prefill-linear-foundation-design.md) — measured T1/Small-T/Large-T model and first shared Small-T/Large-T paths;
- [`../archive/optimization-era/2026-07-10-w8g32-tensor-core-gemm-design.md`](../archive/optimization-era/2026-07-10-w8g32-tensor-core-gemm-design.md) — proposed W8 producer/consumer design;
- [`../archive/optimization-era/2026-07-10-w8g32-tensor-core-gemm-implementation-report.md`](../archive/optimization-era/2026-07-10-w8g32-tensor-core-gemm-implementation-report.md) — measured rejection of that warp-specialized design and final cooperative W8 policy;
- [`../archive/optimization-era/bench/q5090-v3-prefill-kernel-sweep-after-gemm-tuning.md`](../archive/optimization-era/bench/q5090-v3-prefill-kernel-sweep-after-gemm-tuning.md) — retained Q4/Q5/Q6 Large-T roofline evidence.
