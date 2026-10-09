#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {

// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q5:7168x5120`), cold,
// median of 11 then confirmed at median of 21. Two entries move (us, new vs shipped):
//
//   T=8   ksplit7/8 80.9 vs simt 95.2 (+18%)
//   T=112 r64_c64 234.5 vs r64_c32s3 342.0 (+46%); T=80 235.5 vs 282.6 (+20%); T=96 +20%
//
// The 12..16 band prefers the 16-wide tile by 1-4% and the 129..176 band prefers the 32-wide one
// by 4-6%; both are inside this bench's spread on this shape and are left where upstream put them.
// n6144_k5120.cu takes the 129..160 change because there the same margin is 19-22%.
Q5Launch select_q5_n7168_k5120(std::int32_t tokens) {
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
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
