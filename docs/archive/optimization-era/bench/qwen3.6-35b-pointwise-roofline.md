# Qwen3.6-35B Pointwise Op Qualification

Date: 2026-07-16

Target: Section 5 E1-E8 on NVIDIA GeForce RTX 5090 (`sm_120a`, CUDA 13.1, driver
591.86, NCU 2025.4.1):

- AddBias for `D=1152/2048/3456/4304/4608` over the complete raw-patch and merged-token bounds;
- ResidualAdd `[1152,P]` through image/video `P=65536/49152`;
- SigmoidMul `[256,16,T]` for `T=1..6/1024`;
- tanh GELU `[4304,P]` and exact-erf GELU `[4608,V]`, `V=P/4`.

This is standalone Op qualification. It does not qualify the Linear Ops that produce these
activations or register a 35B Engine target.

## Implementation

The four semantic Ops remain separate, while their private storage paths use one narrow
`Bf16PairPack<4>` definition for aligned 16-byte loads/stores. Finite host dispatch is based only
on shape and pointer alignment:

- every registered shape is 16-byte aligned and contains a multiple of eight BF16 elements;
- ResidualAdd and SigmoidMul use one BF16x8 grid-stride stream for their complete registered
  domains;
- AddBias preserves the fastest-dimension broadcast. Cache-sized work uses a two-dimensional
  BF16x8 kernel and reuses each bias pack across up to four columns; larger media use the existing
  high-occupancy BF16x2 stream;
- GELU likewise uses BF16x8 packs through 32 Mi elements and switches to BF16x2 above that finite
  boundary;
- four-byte-aligned even storage retains BF16x2 paths, while odd dimensions or two-byte-aligned
  slices use scalar fallbacks.

The cache-sized boundary is 32 Mi BF16 elements, or a 64 MiB activation. It is shared by AddBias
and GELU because their retained RTX 5090 sweeps show the same transition: explicit packs sharply
reduce instruction/CTA overhead while the activation is cache-sized, whereas the BF16x2 stream
has higher throughput once the item is DRAM-sized.

Vectorization changes storage only. AddBias and ResidualAdd still convert BF16 operands to FP32,
perform one FP32 addition, and round once. SigmoidMul retains FP32 `expf` and multiply. The GELU
tanh/erf formulas remain separate compile-time implementations with their original FP32 operation
order and one final BF16 rounding.

## Correctness

```bash
cmake --build build -j --target \
  ninfer_residual_add_test ninfer_sigmoid_mul_test ninfer_vision_elementwise_test
ctest --test-dir build \
  -R '^ninfer_(residual_add|sigmoid_mul|vision_elementwise)_test$' --output-on-failure
```

The independent BF16-rounded FP64 oracles pass for:

- ResidualAdd `[1152,1/6/4096]` plus retained 27B, odd-tail, and unaligned cases;
- SigmoidMul every `T=1..6`, `T=1024`, retained 27B shapes, large-magnitude sigmoid inputs, and
  unaligned storage;
- every registered AddBias D, `P=4096`, a BF16x2-only D, an odd D, and an unaligned scalar case;
- tanh GELU `[4304,8/256/4096]` and exact GELU `[4608,2/64/1024]`.

Changing P, V, or T only changes the number of independent elements/columns after those dispatch
boundaries; the benchmark below exercises both complete maximum item sizes.

## Release timing

CUDA-event medians are microseconds. Repeated cache-sized payloads can report an effective traffic
rate above physical DRAM bandwidth; those rows are judged against an identical-topology payload
control rather than a nominal DRAM number.

### E4 ResidualAdd

The control reads both inputs, writes the output, and uses the same 256-thread/grid-stride launch.

| P | Op | Control | Effective traffic |
|---:|---:|---:|---:|
| 8 | 9.89 | 10.50 | fixed launch floor |
| 256 | 10.08 | 9.70 | finite-grid floor |
| 4096 | 10.08 | 9.99 | cache-sized control |
| 49152 | 220.35 | 219.89 | 1541.8 GB/s |
| 65536 | 294.86 | 294.27 | 1536.3 GB/s |

### E5 SigmoidMul

`T=1..6` measures 9.84-10.05 us versus 9.73-9.92 us for the payload control. At `T=1024`, the
new BF16x8 route measures 9.92 us versus 9.51 us control; the former BF16x2 route measured 12.77
us. All Small-T cases are fixed-launch limited.

### E6/E7/E8 GELU

| Route | Columns | Op | Control | Effective traffic |
|---|---:|---:|---:|---:|
| tanh `[4304,P]` | 8 | 10.32 | 9.83 | fixed launch floor |
| tanh `[4304,P]` | 4096 | 19.07 | 15.48 | cache-sized; former route 44.00 us |
| tanh `[4304,P]` | 49152 | 544.35 | 547.36 | 1554.5 GB/s |
| tanh `[4304,P]` | 65536 | 724.16 | 737.60 | 1558.0 GB/s |
| exact `[4608,V]` | 2 | 9.75 | 9.73 | fixed launch floor |
| exact `[4608,V]` | 1024 | 10.68 | 9.45 | cache-sized control |
| exact `[4608,V]` | 12288 | 144.22 | 144.34 | 1570.5 GB/s |
| exact `[4608,V]` | 16384 | 194.23 | 198.07 | 1554.8 GB/s |

### E1/E2/E3 AddBias

The default benchmark retains all 25 combinations. Representative points cover every D and both
column families:

| Domain | Columns | Op | Control | Interpretation |
|---|---:|---:|---:|---|
| D=1152, P | 8 | 9.84 | 9.81 | fixed launch floor |
| D=1152, P | 65536 | 199.67 | 200.34 | 1512.4 GB/s |
| D=3456, P | 4096 | 15.58 | 12.95 | former E1 route 36.51 us |
| D=3456, P | 65536 | 596.64 | 599.04 | 1518.5 GB/s |
| D=4304, P | 4096 | 19.20 | 15.72 | cache-sized control |
| D=4304, P | 65536 | 772.74 | 770.43 | measured payload ceiling |
| D=2048, V | 2 | 9.82 | 9.79 | fixed launch floor |
| D=2048, V | 16384 | 30.22 | 25.97 | cache-sized control |
| D=4608, V | 1024 | 10.10 | 10.04 | cache-sized control |
| D=4608, V | 16384 | 200.44 | 201.10 | 1506.7 GB/s |

D=4304 maximum items sustain a lower nominal traffic rate than the other widths, but production
and the same-grid payload control coincide throughout that pair (`1451-1465 GB/s`). This is the
measured topology/memory ceiling rather than remaining AddBias arithmetic work.

## Basic NCU attribution

Following the basic-first workflow, one launch was captured for every implementation class. NCU
uses nine replay passes for this metric set, so duration is diagnostic and is not substituted for
the CUDA-event medians.

| Route | Grid | NCU us | Memory SOL | DRAM SOL | SM SOL | Occupancy | Registers |
|---|---:|---:|---:|---:|---:|---:|---:|
| E4 Residual BF16x8, P=65536 | 4096 | 305.31 | 81.67% | 81.67% | 6.88% | 87.87% | 29 |
| E5 Sigmoid BF16x8, T=1024 | 2048 | 16.29 | 69.96% | 69.96% | 17.39% | 77.42% | 38 |
| E7 tanh BF16x8, P=4096 | 8608 | 34.37 | 77.92% | 77.92% | 33.58% | 81.31% | 40 |
| E7 tanh BF16x2, P=65536 | 137728 | 857.57 | 97.62% | 74.42% | 22.97% | 83.28% | 18 |
| E3 bias BF16x8, D=4304/P=4096 | 3x1024 | 42.78 | 75.01% | 75.01% | 8.95% | 68.97% | 24 |
| E3 bias BF16x2, D=4304/P=65536 | 3x65535 | 917.22 | 92.54% | 69.53% | 15.06% | 78.19% | 24 |

Every captured kernel uses zero static shared memory. Reports are retained locally under
`profiles/ncu/qwen3_6_35b_a3b/pointwise/final/`; no detailed capture was needed after the basic
results identified the intended kernels and the event controls established the finite ceilings.

## Result

E1-E8 are supported for their complete inventory domains. The implementation uses one small,
finite family of aligned BF16x8, BF16x2, and scalar routes rather than target-labelled kernels or
per-shape APIs. Complete 35B Vision/Text execution remains unsupported until the remaining Linear
and sparse-MoE inventory gaps are closed.
