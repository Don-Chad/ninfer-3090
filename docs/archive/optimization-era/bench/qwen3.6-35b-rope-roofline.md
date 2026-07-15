# Qwen3.6-35B RoPE Qualification

Date: 2026-07-15

Target: the standalone Qwen3.6-35B-A3B positional transforms on an RTX 5090:

- R1 Text RoPE/MRoPE: BF16 Q `[256,16,T]`, K `[256,2,T]`, positions `[T]` or `[T,3]`,
  rotary dimension 64, theta `1e7`, decode/verification `T=1..6`, and canonical prefill
  `T=1024`;
- R2 packed Vision 2-D RoPE: BF16 Q/K `[72,16,P]` views of `[3456,P]`, positions `[P,2]`,
  theta `10000`, from minimum video `P=8` through maximum video/image `P=49152/65536`.

This report qualifies the exact fixed routes. The dimension-generic fallback remains functional
machinery and is not evidence for another domain.

## Implementation

One CTA owns one token or patch. The first `rotary_dim/2` threads calculate the fixed-frequency
cosine/sine table once, followed by one block barrier. A warp owns a Q or K head and keeps its two
adjacent coefficient pairs in registers while it processes all heads assigned to that warp.
Adjacent rotary pairs use BF16x2 loads and stores, so the 36-pair Vision case no longer needs a
second scalar `32+4` lane pass.

The exact fixed dispatches are:

- Text `24Q/4KV` and `16Q/2KV`, `D/R=256/64`, theta `1e7`, one or three position axes;
- Vision `16Q/16K`, `D/R=72/72`, theta `10000`, two position axes, including the packed QKV token
  stride;
- corresponding single-Q or single-K calls used by split scheduling.

Text uses one warp per head at `T<=6`, a 256-thread CTA through `T=1020`, and a 192-thread CTA for
the full `T=1024` wave boundary, capped by the number of available head warps for single-tensor
calls. The required K-only route therefore uses 64 threads rather than launching idle warps.
Vision uses 128 threads for every `P`. Fixed inverse-frequency tables remove per-head `powf`; only
one `sincosf` is evaluated per token/pair. The vector route is selected only when the BF16 base and
token stride are BF16x2-aligned. Non-aligned semantic inputs retain the scalar generic route; every
registered Text and packed-Vision layout selects the fixed route.

The previous shared-staging experiment is retained in
[`vision-rope-roofline.md`](vision-rope-roofline.md): staging all 72 dimensions regressed the
Vision kernel because the extra shared traffic and barriers cost more than the global-sector
savings. The final BF16x2 route obtains coalesced global accesses without staging.

## Correctness

```bash
cmake --build build --target ninfer_rope_test ninfer_rope_bench -j
ctest --test-dir build -R '^ninfer_rope_test$' --output-on-failure
```

The test passes its independent FP64 oracle from BF16-rounded inputs. It covers the current 27B
Text domain, exact 35B `16Q/2KV` one-axis and three-axis calls at `T=1/1024`, pair/single API parity,
bit-exact preservation of Text dimensions `64:256`, the packed-stride Vision route, and the
unaligned scalar fallback.

## Fixed-resource control

The benchmark control uses the same grid, block policy, positions, Q/K payload, fixed-frequency
lookup, `sincosf` generation, shared coefficient tables, and block barrier. It replaces the
rotation with ideal 16-byte in-place payload copies. It is therefore a same-grid fixed-resource and
transfer lower control, not an alternate implementation. Event timing uses 20 warmups and at least
500 ms of batched samples.

```bash
./build/bench/ninfer_rope_bench \
  --text --geometry 35b --axes both --tokens 1,2,3,4,5,6,1024
./build/bench/ninfer_rope_bench \
  --text --geometry 35b --axes both --tokens 1,2,3,4,5,6,1024 --control
./build/bench/ninfer_rope_bench --vision --patches 8,256,4096,49152,65536
./build/bench/ninfer_rope_bench --vision --patches 8,256,4096,49152,65536 --control
```

CUDA-event medians are microseconds:

| T | Text 1-D | 1-D control | Text MRoPE | MRoPE control |
|---:|---:|---:|---:|---:|
| 1 | 10.16 | 10.09 | 9.93 | 9.80 |
| 2 | 10.11 | 9.93 | 9.89 | 9.80 |
| 3 | 10.08 | 9.89 | 9.85 | 9.88 |
| 4 | 10.45 | 9.78 | 9.80 | 9.85 |
| 5 | 9.92 | 9.83 | 9.78 | 9.89 |
| 6 | 9.77 | 9.86 | 9.82 | 9.84 |
| 1024 | 9.43 | 10.09 | 9.32 | 9.61 |

The complete Text matrix is at the fixed launch/work floor: production is no more than 6.9% slower
than control, and all but the single 1-D `T=4` sample are within 2%. Both full-chunk routes are
faster than the lower control in the measured event distribution. A 27B regression sweep at
`T=1,6,1024` for both position modes measures 9.41--10.05 us and confirms that the shared family
kernel did not regress the registered geometry.

The MTP schedule also requires separate Q-only and K-only calls. The same axes and T matrix was run
with `--form q` and `--form k`: Q-only remains within 3.1% of its same-grid control, while K-only is
never more than 0.2% slower. At `T=1024`, Q-only measures 9.55--9.60 us and K-only measures
9.72--9.73 us. These are fixed template instances, not generic fallbacks.

| P | Domain point | Vision RoPE | Control | Production / control |
|---:|---|---:|---:|---:|
| 8 | minimum video | 9.31 | 9.40 | 0.990x |
| 256 | minimum image | 9.20 | 9.62 | 0.956x |
| 4096 | canonical item | 12.95 | 12.82 | 1.010x |
| 49152 | maximum video | 409.02 | 402.72 | 1.016x |
| 65536 | maximum image | 540.96 | 537.44 | 1.007x |

The complete Vision matrix is within 1.6% of its same-grid control. At `P=4096`, repeated launches
retain data in cache, so the benchmark's logical GB/s can exceed physical DRAM bandwidth; that
number is not used as a hardware-roofline claim.

## NCU qualification

Hardware/toolchain: GeForce RTX 5090, 170 SMs, `sm_120a`, CUDA 13.1.115, driver 591.86, Nsight
Compute 2025.4.1. The build uses `-lineinfo` and no device-debug `-G`. Preflight reported `4 OK / 0
WARN / 0 FAIL`. All commands use NCU's default kernel replay mode. Initial `launch-skip 0` captures
first verified both regexes; retained headline captures skip the benchmark's 20 warmups:

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r1_mrope_t1_prod_basic.ncu-rep \
  --set basic --kernel-name regex:'rope_fixed_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --text --geometry 35b --axes 3 --tokens 1

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r1_mrope_t1_control_basic.ncu-rep \
  --set basic --kernel-name regex:'rope_payload_control_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --text --geometry 35b --axes 3 --tokens 1 --control

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r1_mrope_t1024_prod_basic.ncu-rep \
  --set basic --kernel-name regex:'rope_fixed_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --text --geometry 35b --axes 3 --tokens 1024

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r1_mrope_t1024_control_basic.ncu-rep \
  --set basic --kernel-name regex:'rope_payload_control_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --text --geometry 35b --axes 3 --tokens 1024 --control

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r2_vision_p65536_prod_basic.ncu-rep \
  --set basic --kernel-name regex:'rope_fixed_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --vision --patches 65536

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r2_vision_p65536_control_basic.ncu-rep \
  --set basic --kernel-name regex:'rope_payload_control_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --vision --patches 65536 --control
```

Each report profiles the intended single kernel. The matching CSV and text exports are retained
beside each `.ncu-rep`.

| Route | Grid x block | Duration | DRAM SOL | Compute SOL | Occupancy | Registers | Static shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| MRoPE `T=1` | `1 x 576` | 3.55 us | 1.26% | 0.03% | 36.48% | 20 | 256 B |
| MRoPE `T=1` control | `1 x 576` | 3.33 us | 0.45% | 0.04% | 36.29% | 21 | 256 B |
| MRoPE `T=1024` | `1024 x 192` | 5.28 us | 26.76% | 14.49% | 68.27% | 20 | 256 B |
| MRoPE `T=1024` control | `1024 x 192` | 5.50 us | 24.50% | 16.04% | 68.21% | 21 | 256 B |
| Vision `P=65536` | `65536 x 128` | 560.03 us | 61.32% | 12.77% | 92.97% | 20 | 288 B |
| Vision `P=65536` control | `65536 x 128` | 551.97 us | 62.71% | 12.07% | 92.71% | 21 | 288 B |

The Vision production route retains 97.8% of the measured control DRAM SOL and 98.6% of its
duration ceiling. Its final detailed capture is:

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r2_vision_p65536_prod_detailed.ncu-rep \
  --set detailed --kernel-name regex:'rope_fixed_kernel' \
  --launch-skip 0 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --vision --patches 65536

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/rope_clean/r2_vision_p65536_prod_stalls.ncu-rep \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:'rope_fixed_kernel' --launch-skip 0 --launch-count 1 \
  ./build/bench/ninfer_rope_bench --vision --patches 65536
```

The detailed report measures 557.86 us, 1.08 TB/s, 60.34% DRAM SOL, 12.83% Compute SOL, 95.18%
occupancy, and no local or shared-memory spilling. It no longer reports excessive global sectors.
Hoisting coefficients out of the head loop reduces shared-memory wavefronts eightfold; the
remaining bank pattern is 1,048,576 excessive wavefronts over the whole 65,536-CTA launch and does
not separate production from its transfer control. WarpState identifies L1TEX scoreboard waits as
the dominant stall at 75.0 of 84.1 cycles between issued instructions (89.2%), consistent with the
measured memory-bound route.

## Decision

R1 and R2 are supported for their complete registered 35B domains. Both exact geometries select a
finite fixed kernel, pass the independent mathematical oracle, and reach their applicable
same-grid fixed-resource ceiling. No additional standalone shape specialization or Vision shared
staging is warranted; future fusion may remove the launch or intermediate Q/K traffic but is not a
condition of standalone support.
