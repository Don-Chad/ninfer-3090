# Qwen3.6-35B FP32-to-BF16 Cast Qualification

Date: 2026-07-16

Target: Vision patch ingress for Qwen3.6-35B-A3B. The source and destination are contiguous
`[1536,P]` tensors in FP32 and BF16. The accepted media domain includes minimum video/image patch
counts `P=8/256`, the canonical `P=4096` case, maximum video `P=49152`, and maximum image
`P=65536`.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. CUDA-event values are medians after warmup. NCU preflight reported
`4 OK / 0 WARN / 0 FAIL`.

## Implementation

The public Op preserves flattened contiguous element order and converts every FP32 element to BF16
with round-to-nearest-even. Dispatch is dimension-driven rather than specialized for one model
shape:

- element count divisible by four, 16-byte-aligned FP32 source, and 8-byte-aligned BF16 destination:
  one thread loads a `float4`, converts two BF16x2 pairs, and writes one aligned 8-byte result;
- even element count with 8-byte/4-byte source/destination alignment: one `float2` load and one
  BF16x2 store;
- every other contiguous shape: scalar conversion.

All routes use 256-thread grid-stride kernels with a bounded grid. Every registered `[1536,P]`
Vision allocation selects the x4 route, while the same Op remains usable for ordinary aligned-even
and arbitrary contiguous dimensions. No per-P specialization or runtime repacking is involved.

## Correctness

```bash
cmake --build build --target ninfer_cast_test ninfer_cast_bench \
  ninfer_qwen3_6_27b_rtx5090 -j8
build/tests/ninfer_cast_test
```

The exact-bit CPU oracle covers the x4 route, x2 route, odd scalar route, deliberately unaligned
scalar route, and Vision shapes `P=8/256/4096`. It reports `OK cast_fp32_to_bf16`. Building the
existing 27B target together with the Op confirms that its call site remains source-compatible.

## Bandwidth qualification

```bash
build/bench/ninfer_cast_bench
build/bench/ninfer_cast_bench --control
```

The benchmark-local control has the same grid and block geometry, reads the same 16-byte source
payload, retains every input word, and writes the same 8-byte destination payload per thread. It
omits only FP32-to-BF16 conversion, so it measures the attainable launch/cache/DRAM floor for the
production kernel.

CUDA-event medians from the serial production and control runs are:

| P | Traffic | Production | Payload control | Production / control | Production throughput |
|---:|---:|---:|---:|---:|---:|
| 8 | 72 KiB | 9.85 us | 9.79 us | 1.006x | 7.5 GB/s |
| 256 | 2.25 MiB | 9.86 us | 10.00 us | 0.986x | 239.4 GB/s |
| 4096 | 36 MiB | 10.72 us | 10.59 us | 1.012x | 3522.2 GB/s |
| 49152 | 432 MiB | 297.76 us | 297.60 us | 1.001x | 1521.3 GB/s |
| 65536 | 576 MiB | 395.14 us | 394.88 us | 1.001x | 1528.5 GB/s |

Every point is within 1.3% of the matching payload control. Small inputs are launch/cache limited;
the two maximum media shapes are DRAM limited. Maximum image size reaches 85.3% of the benchmark's
1792 GB/s nominal device bandwidth while remaining indistinguishable from the raw-payload control.

## NCU confirmation

One narrow capture confirms the production x4 kernel and bottleneck at maximum image size:

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/cast/fp32_bf16_p65536__basic \
  --set basic --kernel-name regex:cast_fp32_to_bf16_x4_kernel \
  --launch-skip 0 --launch-count 1 \
  build/bench/ninfer_cast_bench --patch 65536 --profile-once
```

The captured launch is `4096 x 256`, uses 34 registers/thread and no static or dynamic shared
memory, and reaches 91.05% achieved occupancy. Under NCU's multi-pass profiling clocks it reports
76.05% DRAM SOL, 2.67% compute SOL, and identifies memory as the limiting resource. The NCU duration
is profiling evidence rather than the latency source; the serial CUDA-event production/control
pair above establishes attainable end-to-end kernel latency.

Retained report:

- `profiles/ncu/qwen3_6_35b_a3b/cast/fp32_bf16_p65536__basic.ncu-rep`

This qualifies I1 for the complete 35B Vision patch domain.
