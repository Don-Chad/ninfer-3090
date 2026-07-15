# Qwen3.6-35B A4 Segmented Vision Attention Qualification

Date: 2026-07-15

Target: packed, non-causal Vision attention with BF16 Q/K/V/O `[72,16,P]`. One media item has
`S=g_t` equal consecutive segments of length `L=g_h*g_w`, so `P=S*L`. The accepted target domain
is one image with `P<=65536` or one video with `P<=49152`. The required execution layout uses Q/K/V
views of packed QKV `[3456,P]`, with a 3456-element token stride, and contiguous output
`[1152,P]`.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. Timings are CUDA-event medians after warmup. NCU uses one selected
launch and application replay.

## Qualified implementation

For each segment, head, query `i`, and key `j`, the operation computes:

```text
score[i,j] = dot(Q[:,h,i], K[:,h,j]) / sqrt(72)
O[:,h,i]  = sum_j softmax(score[i,:])[j] * V[:,h,j]
```

The implementation is a fixed-head FlashAttention kernel:

- one CTA owns one query-row tile and one of the 16 heads;
- Q, K, and V use 16-byte asynchronous copies into XOR-swizzled shared memory;
- `mma.sync.m16n8k16` computes QK and probability-times-V with FP32 accumulators;
- online max, sum, and output state remain in registers across key tiles;
- the 72-element head is zero-padded to 80 for QK MMA and to a 128-element shared-memory pitch;
- no attention-score matrix or other quadratic workspace is written to global memory.

The 35B route receives the equal segment length directly. Every CTA derives `(segment,query_tile)`
from `blockIdx.x`; there is no `cu_seqlens` read, tile descriptor, workspace allocation, or setup
kernel. The existing arbitrary-segment descriptor overload remains available for the registered
27B target and is unchanged semantically, but it is not used to qualify the 35B item route.

Three square row/key tile instances cover the target domain:

| Tile | Threads/CTA | Dynamic shared memory | Query rows per warp |
|---:|---:|---:|---:|
| 16 | 32 | 12.29 KiB | 16 |
| 32 | 64 | 24.58 KiB | 16 |
| 64 | 128 | 49.15 KiB | 16 |

The host dispatch minimizes padded tile area divided by measured relative rates `3:6:8` for the
16/32/64 instances. This is a small arithmetic decision over the explicit segment length; it does
not inspect tensor data or introduce a setup launch.

## Correctness and preservation

```bash
cmake --build build-release --target \
  ninfer_vision_attention_test ninfer_vision_attention_bench -j8
build-release/tests/ninfer_vision_attention_test
cmake --build build-release --target ninfer_qwen3_6_27b_rtx5090 -j8
```

The independent CPU oracle covers contiguous and packed-QKV layouts, arbitrary descriptor
segments, the public equal-segment overload, and tail rows for every 16/32/64 production instance.
The operation test completes with `OK vision_attention correctness`; its BF16 relative-L2 errors
remain approximately 0.20-0.23%. The original arbitrary-segment cases still exercise the 27B
descriptor route.

## Tile dispatch evidence

The benchmark can force each compiled tile without changing the input layout or operation. Selected
CUDA-event medians are:

| S | L | Tile 16 | Tile 32 | Tile 64 | Auto |
|---:|---:|---:|---:|---:|---:|
| 384 | 4 | **12.96 us** | 23.11 us | 58.26 us | 16 |
| 128 | 16 | **9.59 us** | 11.17 us | 23.17 us | 16 |
| 64 | 20 | 10.64 us | **10.06 us** | 15.12 us | 32 |
| 32 | 36 | 10.96 us | 11.09 us | **10.75 us** | 64 |
| 32 | 64 | 16.70 us | 12.89 us | **12.40 us** | 64 |
| 32 | 68 | 22.95 us | **18.94 us** | 23.20 us | 32 |
| 8 | 196 | 35.64 us | 25.16 us | **24.18 us** | 64 |
| 1 | 4096 | 1372.13 us | 689.95 us | **523.55 us** | 64 |

An exhaustive `L=4..256` sweep in steps of four, with `S` chosen near `2048/L`, found no dispatch
choice more than 8% slower than the measured best instance. Separate `S=2` and `S=384` boundary
sweeps were within 3.1% of the measured best. The largest observed apparent mismatch was a
measurement-level tie near a dispatch boundary, not a distinct performance regime.

## Canonical target matrix

Example invocation:

```bash
build-release/bench/ninfer_vision_attention_bench \
  --segments 32 --length 68 --warmup 5 --repeat 50 --min-time-ms 200
```

The benchmark allocates the real packed-QKV input layout. Auto-dispatch results are:

| S | L | P | Tile | Median | Mathematical TFLOP/s | Issued-MMA TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 4 | 1,536 | 16 | 12.95 us | 2.19 | 36.94 |
| 128 | 16 | 2,048 | 16 | 9.56 us | 15.80 | 16.68 |
| 64 | 20 | 1,280 | 32 | 10.28 us | 11.47 | 31.00 |
| 32 | 36 | 1,152 | 64 | 10.80 us | 17.70 | 59.05 |
| 32 | 64 | 2,048 | 64 | 12.16 us | 49.65 | 52.41 |
| 32 | 68 | 2,176 | 32 | 18.98 us | 35.93 | 75.59 |
| 8 | 196 | 1,568 | 64 | 24.28 us | 58.34 | 105.05 |
| 8 | 256 | 2,048 | 64 | 24.84 us | 97.27 | 102.67 |
| 1 | 256 | 256 | 64 | 10.76 us | 28.06 | 29.61 |
| 1 | 1,024 | 1,024 | 64 | 41.98 us | 115.09 | 121.48 |
| 1 | 4,096 | 4,096 | 64 | 524.48 us | 147.40 | 155.59 |
| 12,288 | 4 | 49,152 | 16 | 364.86 us | 2.48 | 41.94 |
| 2 | 24,576 | 49,152 | 64 | 33.253 ms | 167.39 | 176.69 |
| 1 | 65,536 | 65,536 | 64 | 118.069 ms | 167.62 | 176.94 |

Mathematical work is `4*S*L^2*72*16`. Issued-MMA work counts both the 80-wide QK contraction and
tile padding:

```text
2*S*ceil(L/Br)*Br*ceil(L/Bc)*Bc*16*(80+72)
```

For small or tail-heavy segments, mathematical TFLOP/s is intentionally not a utilization metric:
padding and finite CTA work dominate. Those cases use the fixed-resource control below.

## Removal of the descriptor/setup floor

The prior fixed-64 arbitrary-segment route builds descriptors and launches its capacity grid. The
direct equal-segment path removes that work and selects the appropriate tile:

| S | L | Descriptor route | Direct route | Speedup |
|---:|---:|---:|---:|---:|
| 384 | 4 | 90.51 us | 12.95 us | 6.99x |
| 128 | 16 | 36.09 us | 9.56 us | 3.78x |
| 32 | 68 | 29.76 us | 18.98 us | 1.57x |
| 8 | 256 | 29.59 us | 24.84 us | 1.19x |
| 1 | 4,096 | 524.90 us | 524.48 us | 1.00x |

The long one-segment case already needed no descriptors, so its unchanged timing is the expected
control. The improvement is concentrated where equal video segments previously paid setup and
64-row padding costs.

## Fixed-resource qualification

`--control` launches a benchmark-only kernel with the same grid, threads/CTA, dynamic shared-memory
reservation, and Q/K/V/O global payload as the selected production instance. It copies every valid
Q row to O, reads every K/V row once per query tile, and retains the reads through a checksum. It
omits QK, softmax, and PV. The checksum sink is excluded from the modeled payload. This is a
same-shape fixed-resource and transfer lower control, not an alternate implementation.

Paired event medians:

| S | L | Production | Payload control |
|---:|---:|---:|---:|
| 384 | 4 | 13.01 us | 9.74 us |
| 128 | 16 | 9.55 us | 10.28 us |
| 32 | 68 | 19.14 us | 10.84 us |
| 8 | 256 | 24.70 us | 13.14 us |
| 12,288 | 4 | 364.93 us | 409.82 us |

NCU basic captures remove the batched event interval and verify equal launch resources:

| Shape and route | Grid x block | Dynamic smem | NCU duration | Compute SOL | Memory SOL | DRAM SOL | Achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| S384 L4 production | `6144 x 32` | 12.29 KiB | 18.24 us | 30.25% | 37.96% | 37.96% | 13.10% |
| S384 L4 control | `6144 x 32` | 12.29 KiB | 17.89 us | 6.43% | 42.28% | 42.28% | 13.03% |
| S32 L68 production | `1536 x 64` | 24.58 KiB | 28.06 us | 30.71% | 48.33% | 38.02% | 14.93% |
| S32 L68 control | `1536 x 64` | 24.58 KiB | 18.18 us | 5.00% | 47.25% | 47.25% | 14.65% |

The four-row case is at the same fixed launch/resource floor as its control while executing the
attention math. The L=68 production launch issues 1.434 billion tensor FLOPs; at the 209.5 TFLOP/s
device peak used by the existing RTX 5090 benchmarks, ideal MMA time is 6.85 us. The conservative
`control + ideal-MMA` envelope is therefore 25.03 us versus the observed 28.06 us, within 12.1%,
including softmax, online state, synchronization, and tails. At the maximum number of short video
segments, production is already faster than the same-payload checksum control.

## Long-segment compute roofline

The maximum video and image cases sustain 176.69 and 176.94 issued-MMA TFLOP/s, respectively
84.3-84.5% of the 209.5 TFLOP/s RTX 5090 dense BF16/FP32-accumulate peak used by the project.
Representative NCU captures show the transition to the compute-limited regime:

| Shape | NCU duration | Compute SOL | Memory SOL | DRAM SOL | Achieved/theoretical occupancy | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|
| S1 L4096 | 725.92 us | 67.19% | 50.19% | 2.77% | 15.83% / 16.67% | 3.01 |
| S2 L24576 | 48.62 ms | 80.66% | 59.26% | 0.94% | 16.57% / 16.67% | 36.14 |

The maximum-video capture reaches 99.4% of the instance's resource-limited theoretical occupancy
and NCU classifies it above 80% Compute SOL. Low DRAM SOL is expected because each K/V tile is
reused through shared memory across query rows; tensor work, shared/L2 feed, and synchronization
are the relevant limits. NCU replay inflates absolute latency, so CUDA-event values are used for
throughput.

## NCU commands and retained reports

```bash
ncu --set basic --replay-mode application --launch-skip 0 --launch-count 1 \
  --kernel-name regex:'vision_attention_flash_kernel' --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_vision_attention/uniform_s32_l68_t32__basic \
  build-release/bench/ninfer_vision_attention_bench \
  --segments 32 --length 68 --profile-once

ncu --set basic --replay-mode application --launch-skip 0 --launch-count 1 \
  --kernel-name regex:'vision_attention_payload_control_kernel' --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_vision_attention/control_s32_l68_t32__basic \
  build-release/bench/ninfer_vision_attention_bench \
  --segments 32 --length 68 --control --profile-once
```

Retained reports:

- `profiles/ncu/qwen3_6_35b_vision_attention/uniform_s384_l4_t16__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_vision_attention/control_s384_l4_t16__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_vision_attention/uniform_s32_l68_t32__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_vision_attention/control_s32_l68_t32__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_vision_attention/uniform_s1_l4096_t64__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_vision_attention/uniform_s2_l24576_t64__basic.ncu-rep`

This evidence qualifies the former A4 exact-4096 case and A5 remaining equal-segment cases as one
complete segmented Vision-attention operation, now listed as A4 in the active inventory.
