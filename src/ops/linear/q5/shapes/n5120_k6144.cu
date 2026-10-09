#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {

// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q5:5120x6144`), cold,
// median of 11 then confirmed at median of 21 with min..p95. The 17..96 band is upstream's and is
// best here; everything else in the narrow and wide ends moves (us, new vs shipped):
//
//   T=8   ksplit7/8 63.5 vs simt 87.0 (+37%)
//   T=12  r64_c16 102.4 vs simt 129.0 (+31%); T=15 101.4 vs 155.6 (+54%); T=16 95.2 vs 111.6 (+17%)
//   T=112 r32_c128 239.6 vs r64_c32s3 295.9 (+24%)
//   T=144 r64_c64 280.6 vs r32_c128 443.4 (+58%); T=160 +60%; T=176 +56%
//   T=192 r64_c128 299.0 vs r32_c128 458.8 (+53%); T=224 +55%; T=256 +41%
//
// The 32-row 128-wide tile upstream gave 129..256 is the mirror of the Q4 finding: at 5120 rows
// its grid is 160 row-blocks per column tile, so from a second column tile on it runs more than a
// wave behind the 64-row tiles, and the deficit grows with T rather than closing.
Q5Launch select_q5_n5120_k6144(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_a16_direct_r1_t1_w4_k6144;
    if (tokens <= 2) return launch_q5_a16_direct_exact<6144, 2, 2>;
    if (tokens <= 3) return launch_q5_a16_direct_exact<6144, 3, 2>;
    if (tokens <= 4) return launch_q5_a16_direct_exact<6144, 4, 2>;
    if (tokens <= 5) return launch_q5_a16_direct_exact<6144, 5, 2>;
    if (tokens <= 6) return launch_q5_a16_direct_exact<6144, 6, 2>;
    if (tokens <= 7) return launch_q5_a16_direct_exact<6144, 7, 2>;
    if (tokens <= 8) return launch_q5_a16_direct_exact<6144, 8, 2>;
    if (tokens <= 16) return launch_q5_a16_mma_r64_t16_k64_wr16_wt8_s2_a2_b3;
    if (tokens <= 96) return launch_q5_a16_mma_r64_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 128) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 176) return launch_q5_a16_mma_r64_t64_k64_wr32_wt32_s2_a2_b3_pingpong;
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
