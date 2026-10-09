#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {

// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q5:6144x5120`), cold,
// median of 11 then confirmed at median of 21 with min..p95. The K-split ladder and the 12..64
// band are upstream's and are best here; three things move (us, new vs shipped):
//
//   T=8   ksplit7/8 70.7 vs simt 85.0 (+20%) -- the ladder stopped at six columns and the SIMT
//                                               tile that took 7..11 is 20% behind it at eight
//   T=112 r64_c64 231.4 vs r64_c32s3 286.7 (+24%); T=80 228.4 vs 241.7 (+6%)
//   T=160 r64_c32s3 347.1 vs r64_c128 423.9 (+22%); T=144 354.3 vs 419.8 (+19%)
//
// The 129..160 window is a wave effect, not a drift: 128 columns is exactly one tile of the
// 128-wide schedule and it wins there outright (240.6), while a 129..160 extent bills a second
// almost-empty tile as a full wave and the 32-wide tile covers the same extent for less. 176 and
// up return to the 128-wide tile, where the margin is inside the spread.
Q5Launch select_q5_n6144_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 2) return launch_q5_a16_direct_exact<5120, 2, 4>;
    if (tokens <= 3) return launch_q5_a16_direct_exact<5120, 3, 4>;
    if (tokens <= 4) return launch_q5_a16_direct_exact<5120, 4, 4>;
    if (tokens <= 5) return launch_q5_a16_direct_exact<5120, 5, 4>;
    if (tokens <= 6) return launch_q5_a16_direct_exact<5120, 6, 4>;
    if (tokens <= 7) return launch_q5_a16_direct_exact<5120, 7, 4>;
    if (tokens <= 8) return launch_q5_a16_direct_exact<5120, 8, 4>;
    if (tokens <= 11) return launch_q5_a16_simt_r8_t4_w1_g16_s2;
    if (tokens <= 64) return launch_q5_a16_mma_r64_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 112) return launch_q5_a16_mma_r64_t64_k64_wr32_wt32_s2_a2_b3_pingpong;
    if (tokens <= 128) return launch_q5_a16_mma_r64_t128;
    if (tokens <= 160) return launch_q5_a16_mma_r64_t32_k64_wr16_wt16_s3_a3_b2;
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
