# Qwen3.6-35B W8 multi-output input projections

This report retains the implementation and RTX 5090 qualification evidence for the W8 Attention
`[9216,2048]` and GDN `[12288,2048]` input projections. It is performance provenance, not a 35B
Engine-support claim.

## Contract and implementation

Both Ops consume contiguous BF16 `X[2048,T]` and one `W8G32_F16S` RowSplit parent. Attention
writes independent contiguous Q `[4096,T]`, K `[512,T]`, gate `[4096,T]`, and V `[512,T]` buffers.
GDN writes contiguous Q/K/V `[8192,T]` and independent Z `[4096,T]`. No route materializes a
`[9216,T]` or `[12288,T]` parent output, performs a follow-up extract, or uses transient workspace.

The finite dispatch is:

| Op | T domain | Schedule |
|---|---:|---|
| Attention | 1 | eight warp rows per CTA, direct K=2048 decode |
| Attention | 2..12 | SIMT R8 C4 |
| Attention | 13..128 | MMA R32 C128 |
| Attention | 129..INT32_MAX | MMA R64 C128 |
| GDN | 1 | eight warp rows per CTA, direct K=2048 decode |
| GDN | 2..16 | SIMT R8 C4 |
| GDN | 17..INT32_MAX | MMA R64 C128 |

All split boundaries are CTA-tile aligned. The shared W8 kernel templates select one output tile
per CTA, so inner accumulation and weight staging are unchanged while stores use the final
allocation's leading dimension and parent-row origin.

## Correctness

The test builds patterned packed W8 weights, exact-decodes signed codes and FP16 scales in an
independent FP64 projection oracle, and verifies samples from every logical output. It covers
decode, both sides of every route seam, full/predicated tiles, and `T=1024`. Guard regions protect
all independent output allocations. Both T=1 Ops are captured and replayed through a CUDA Graph.

The representative production-versus-oracle relative L2 errors remain below `0.004`; existing W8
Linear and LinearAdd regressions pass after the split-output template refactor.

## Complete-Op performance

Hardware/toolchain: NVIDIA GeForce RTX 5090, SM 12.0 (`sm_120a` build), CUDA 13.1, driver 591.86.
Each sample uses a 256 MiB L2 flush; the table reports medians from 50 repetitions after ten
warmups. The control is the same packed parent Linear followed by the required four or two BF16
extracts, so it has the same public output topology.

```bash
./build/bench/ninfer_w8_input_proj_bench \
  --op all --t-sweep 1,2,4,8,12,13,16,17,32,64,128,129,256,512,1024 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/w8_input_proj_final.csv
```

| Op | T | Production (us) | Parent + extract (us) | Latency reduction | Useful TFLOP/s |
|---|---:|---:|---:|---:|---:|
| Attention | 1 | 19.424 | 29.600 | 34.4% | 1.94 |
| Attention | 13 | 46.336 | 54.112 | 14.4% | 10.59 |
| Attention | 129 | 65.536 | 78.720 | 16.7% | 74.30 |
| Attention | 1024 | 214.272 | 246.592 | 13.1% | 180.40 |
| GDN | 1 | 21.792 | 29.696 | 26.6% | 2.31 |
| GDN | 17 | 66.688 | 70.624 | 5.6% | 12.83 |
| GDN | 128 | 66.816 | 76.480 | 12.6% | 96.42 |
| GDN | 1024 | 296.192 | 332.608 | 10.9% | 174.01 |

At the measured seams, Attention MMA32 is 4.0% faster than SIMT at `T=13`, and MMA64 is 17.2%
faster than MMA32 at `T=129`. GDN MMA64 is 6.0% faster than SIMT at `T=17`.

## Nsight Compute

Nsight Compute 2025.4.1 profiled exactly one matching production launch per command. The basic
captures used `--set basic`, demangled narrow regexes, `--launch-skip 0`, and `--launch-count 1`.
Scheduler/warp-state captures used explicit `SchedulerStats` and `WarpStateStats` sections.

| Kernel/workload | NCU duration | SM throughput | DRAM throughput | Registers/thread | Static shared | Achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|
| Attention decode, T=1 | 18.62 us | 18.69% | 63.76% | 76 | 0 B | 42.64% |
| GDN decode, T=1 | 23.94 us | 19.30% | 73.10% | 78 | 0 B | 43.15% |
| Attention MMA64, T=1024 | 303.49 us | 75.36% | 8.06% | 113 | 46.08 KiB | 31.15% |
| GDN MMA64, T=1024 | 413.70 us | 73.87% | 8.26% | 113 | 46.08 KiB | 31.60% |

The decode kernel is DRAM/latency limited: 85.0% of its 28.03 warp cycles per issued instruction
are long-scoreboard stalls. The large-T MMA kernel is compute-pipeline limited: its largest stall
class is execution-pipe busy at 41.7% of 14.24 cycles per issued instruction. Detailed captures
report zero local- or shared-memory spilling for the representative Attention kernels.

Reports and exported CSV/text views are under
`profiles/ncu/qwen3_6_35b_a3b/input_proj/`; the complete-Op CSV is
`profiles/bench/w8_input_proj_final.csv`. These local artifacts are ignored by Git.

The qualification is Op-level and covers measured points through `T=1024`. Every positive T is
functionally admitted, but this report does not claim end-to-end 35B inference performance or
register the future target.
