#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// The Q4 draft head. The table that arrived with the 2026-09-17 catch-up sent everything above
// eight columns straight to the 128-wide tile, and on sm_86 that is the single worst route in the
// whole `linear` registry: the draft and verify widths a DFlash2/MTP round actually uses cost
// 2.4-3.6x what the right tile costs. Re-measured 2026-09-17 with
// bench/ops/linear_schedule_bench.cu (`q4:131072x5120`), cold, median of 11, confirmed at 15 (us,
// chosen route vs the 128-wide tile that shipped):
//
//   T=12   ksplit24   716.8 vs 2598.9  (3.63x)   T=16   ksplit24   841.7 vs 2597.9  (3.09x)
//   T=24   ksplit24  1113.1 vs 2656.3  (2.39x)   T=32   r32_c32   1338.4 vs 2672.6  (2.00x)
//   T=48   r64_c48   1586.2 vs 2747.4  (1.73x)   T=64   r64_c64   1678.3 vs 2793.5  (1.66x)
//   T=80   r64_c80   2261.0 vs 2786.3  (+23%)    T=96   r64_c96   2502.7 vs 2855.9  (+14%)
//   T=112  r64_c112  2540.5 vs 2879.5  (+13%)    T=128  r64_c128  2793.5 (best; unchanged)
//   T=160  r64_c80   4393.0 vs 5534.7  (+26%)    T=192  r64_c96   4966.4 vs 5661.7  (+14%)
//   T=224  r64_c112  5062.7 vs 5744.6  (+13%)    T=256+ r64_c128  best at every point to 1024
//
// Above 128 the bounds are tile-count edges and repeat the 65..128 pattern one wave up, the same
// rule n34816_k5120.cu records. T=1..8 and T>=256 are upstream's values and are best here too.
Q4Launch select_q4_n131072_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_ksplit<131072, 5120, 4>;
    if (tokens <= 8) return launch_q4_ksplit<131072, 5120, 8>;
    if (tokens <= 24) return launch_q4_ksplit<131072, 5120, 24>;
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 48) return launch_q4_a16_mma_r64_t48;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 160) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
