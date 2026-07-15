# Qwen3.6-35B Vision Position-Embedding Qualification

Date: 2026-07-16

Target: the fused four-corner position-table gather, weighted interpolation, and in-place residual
add used by the Qwen3.6-35B-A3B Vision stem. The exact tensors are BF16 table `[1152,2304]`, I32
indices `[4,P]`, FP32 weights `[4,P]`, and BF16 activation `[1152,P]`. The accepted media domain
extends from minimum video/image `P=8/256` through maximum video/image `P=49152/65536`.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. CUDA-event values are medians after warmup. NCU preflight reported
`4 OK / 0 WARN / 0 FAIL`.

## Implementation

The table's 1152 channels are handled as 576 BF16 pairs. Both optimized routes load four
coalesced table rows, accumulate the weighted interpolation in FP32, round the position result to
BF16, add it to the BF16 activation, and write BF16 in place:

- `P<1024`: warp-tiled. One 32-thread block owns one tile of a patch. Lanes 0-3 load the four
  indices and weights and broadcast them with warp shuffles. The number of tiles per patch is
  chosen from P to expose up to roughly 4096 blocks, bounded by the 18 useful pair tiles.
- `P>=1024`: patch CTA. One 128-thread block owns one patch, stages its 32 bytes of index/weight
  control in shared memory, and walks the 576 pairs cooperatively.

The small-P route removes the underfilled one-CTA-per-patch topology without burdening the
bandwidth-saturated large-P route. Because valid indices are part of the Op contract, the optimized
routes do not repeat device bounds checks for every channel. The existing contiguous scalar route
continues to cover other D values.

At `P=256`, the original one-CTA route measured 14.01 us. Warp tiling reduces this to 9.45 us, a
32.5% latency reduction. The dispatch boundary is smooth: final production medians at
`P=1023/1024` are 9.68/9.42 us.

## Correctness and compatibility

```bash
cmake --build build --target ninfer_vision_pos_embed_test \
  ninfer_vision_pos_embed_bench ninfer_qwen3_6_27b_rtx5090 -j8
build/tests/ninfer_vision_pos_embed_test
```

The independent host oracle constructs valid table gathers and normalized interpolation weights,
then compares the BF16 output of the exact D=1152 optimized route. It reports
`OK vision_pos_embed correctness`. The existing 27B target builds against the same Op; its D=1152
Vision call retains the same contract and now shares the P-dependent dispatch.

## Fixed-work and bandwidth qualification

```bash
build/bench/ninfer_vision_pos_embed_bench
build/bench/ninfer_vision_pos_embed_bench --control
```

Each benchmark-local payload control has the same route, grid, block geometry, four indexed table
loads, x read/write, and index/weight loads as production. Its output depends on every input but it
omits BF16 conversion and FP32 interpolation. The printed GB/s counts the minimum external traffic
after the 5.1 MiB table is resident in L2: x read/write plus indices and weights. Production/control
latency is the qualification metric because the required table gathers dominate logical traffic.

| P | Production | Payload control | Production / control | Minimum external throughput |
|---:|---:|---:|---:|---:|
| 8 | 10.59 us | 9.78 us | 1.083x | 3.5 GB/s |
| 256 | 9.45 us | 9.19 us | 1.028x | 125.7 GB/s |
| 4096 | 11.04 us | 11.05 us | 0.999x | 1721.1 GB/s |
| 49152 | 149.95 us | 150.38 us | 0.997x | 1520.9 GB/s |
| 65536 | 202.34 us | 201.06 us | 1.006x | 1502.9 GB/s |

The minimum video case is launch/work limited and lies within 8.3% of its control. Every other
registered point lies within 2.9%. At canonical and maximum sizes the interpolation arithmetic is
fully hidden behind the required cache/DRAM traffic.

## NCU confirmation

One narrow capture uses NCU's default kernel replay and targets the production CTA at maximum
image size:

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/vision_pos_embed/d1152_p65536__basic \
  --set basic --kernel-name regex:vision_pos_embed_add_d1152_kernel \
  --launch-skip 0 --launch-count 1 \
  build/bench/ninfer_vision_pos_embed_bench --patch 65536 --profile-once
```

NCU profiled the intended `65536 x 128` launch. It reports 185.76 us, 87.98% memory/L2 SOL,
81.72% DRAM SOL, 32.14% compute SOL, 90.19% achieved occupancy, 26 registers/thread, 32 bytes of
static shared memory, and 32.13 waves/SM. The limiting resource is the required L2-served table
gather rather than interpolation arithmetic.

Retained report:

- `profiles/ncu/qwen3_6_35b_a3b/vision_pos_embed/d1152_p65536__basic.ncu-rep`

This qualifies I6 for the complete 35B Vision patch domain.
