# Qwen3.6-35B Visual-Embedding Scatter Qualification

Date: 2026-07-16

Target: exact BF16 column scatter from visual embeddings `[2048,V]` through I32 indices `[V]`
into Text or MTP embeddings `[2048,T]`. The runtime normally invokes the Op on chunk-local windows
with `V<=T<=1024`. The same route covers complete one-item envelopes through maximum video/image
sizes `V=12288/16384`.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. CUDA-event values are medians after warmup. NCU preflight reported
`4 OK / 0 WARN / 0 FAIL`.

## Implementation

One block owns one source column, reads its destination-column index, and copies along contiguous
D. Index validity and uniqueness are caller contracts, so the kernel does not add a shared-memory
barrier or repeat device bounds handling around every column. Dispatch is dimension-driven:

- D divisible by eight with 16-byte-aligned source and destination: BF16x8/`uint4` loads and stores
  with a 128-thread block;
- other aligned even D: BF16x2 loads and stores with a 256-thread block;
- every other contiguous D: scalar BF16 copies with a 256-thread block.

For D=2048, each column is exactly 256 `uint4` values, so every x8 thread copies two vectors. The
registered 27B D=5120 route also selects x8; the x2 and scalar paths preserve the broader Op
domain. There is no workspace, repacking, intermediate output, or change to untouched destination
columns.

The previous BF16x2/shared-barrier path measured 31.88 us at maximum video V=12288 and 77.40 us at
maximum image V=16384. The final path measures 25.10/74.13 us, improving those points by 21.3% and
4.2%, respectively.

## Exact correctness and compatibility

```bash
cmake --build build --target ninfer_scatter_test \
  ninfer_qwen3_6_27b_visual_scatter_test ninfer_qwen3_6_27b_rtx5090 -j8
build/tests/ninfer_scatter_test
build/tests/ninfer_qwen3_6_27b_visual_scatter_test
```

The shared Op test compares BF16 storage bits exactly, including non-monotonic destination indices,
untouched columns, target D=2048 x8, an even-D x2 case, and an odd-D scalar case. It reports
`OK scatter exact copy`. The existing 27B shifted-visual composition test reports exact zero error
for both chunk-boundary cases, and the 27B target builds with the widened x8 route. Its retained
`D=5120,V=2040` point measures 11.29 us versus an 11.14 us payload control, statistically
unchanged from the previous 11.16 us qualification.

## Fixed-work and bandwidth qualification

```bash
build/bench/ninfer_scatter_bench
build/bench/ninfer_scatter_bench --control
```

The benchmark control uses the same grid, block, BF16x8 source reads, and destination writes, but
uses the known sequential destination column directly instead of loading it through the index
array. It is the lower fixed-work floor for a pure scatter copy. The target benchmark itself uses
the equivalent sorted `indices[j]=j+1` pattern produced by multimodal placeholder spans.

| V | Production | Payload control | Production / control | Useful throughput |
|---:|---:|---:|---:|---:|
| 1 | 10.38 us | 10.28 us | 1.010x | 0.8 GB/s |
| 2 | 10.27 us | 9.94 us | 1.033x | 1.6 GB/s |
| 64 | 10.14 us | 9.79 us | 1.036x | 51.7 GB/s |
| 1024 | 9.51 us | 9.42 us | 1.010x | 882.3 GB/s |
| 12288 | 25.10 us | 25.19 us | 0.996x | 4012.9 GB/s |
| 16384 | 74.13 us | 74.32 us | 0.997x | 1811.6 GB/s |

Every point is within 3.6% of the matching payload floor. Small and chunk-sized cases are bounded
by launch size and cache residency. Maximum video remains largely cache-served; maximum image
exceeds cache capacity and reaches the DRAM streaming regime.

## NCU confirmation

One narrow capture uses NCU's default kernel replay and targets maximum image size:

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/scatter/d2048_v16384_b128__basic \
  --set basic --kernel-name regex:scatter_bf16x8_kernel \
  --launch-skip 0 --launch-count 1 \
  build/bench/ninfer_scatter_bench --vision 16384 --profile-once
```

NCU profiled the intended `16384 x 128` production launch. It reports 75.97 us, 80.75% memory and
DRAM SOL, 36.30% L2 SOL, 5.65% compute SOL, 86.14% achieved occupancy, 30 registers/thread, zero
static/dynamic shared memory, and 8.03 waves/SM. The kernel is DRAM-bound; useful event throughput
and the same-topology control establish that the remaining time is the required source-read and
destination-write payload rather than index or address overhead.

Retained report:

- `profiles/ncu/qwen3_6_35b_a3b/scatter/d2048_v16384_b128__basic.ncu-rep`

This qualifies I3 for the complete 35B Text/MTP multimodal insertion domain.
