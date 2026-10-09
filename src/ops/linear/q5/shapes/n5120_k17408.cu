#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {

// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q5:5120x17408`), cold,
// median of 11 then confirmed at median of 21 with min..p95. Same story as n5120_k6144.cu, and
// larger because K is longer (us, new vs shipped):
//
//   T=8   ksplit7/8 184.3 vs simt 255.0 (+38%)
//   T=12  r64_c16 284.7 vs simt 377.9 (+33%); T=15 285.7 vs 495.6 (+74%); T=16 274.4 vs 306.2 (+12%)
//   T=112 r32_c128 664.6 vs r64_c32s3 825.3 (+24%)
//   T=144 r64_c128 779.3 vs r32_c128 1226.8 (+57%); T=176 +63%; T=224 +64%; T=256 +48%
//
// 17..96 and 113..128 keep upstream's routes and are best here.
Q5Launch select_q5_n5120_k17408(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_a16_direct_r1_t1_w4_k17408;
    if (tokens <= 2) return launch_q5_a16_direct_exact<17408, 2, 2>;
    if (tokens <= 3) return launch_q5_a16_direct_exact<17408, 3, 2>;
    if (tokens <= 4) return launch_q5_a16_direct_exact<17408, 4, 2>;
    if (tokens <= 5) return launch_q5_a16_direct_exact<17408, 5, 2>;
    if (tokens <= 6) return launch_q5_a16_direct_exact<17408, 6, 2>;
    if (tokens <= 7) return launch_q5_a16_direct_exact<17408, 7, 2>;
    if (tokens <= 8) return launch_q5_a16_direct_exact<17408, 8, 2>;
    if (tokens <= 16) return launch_q5_a16_mma_r64_t16_k64_wr16_wt8_s2_a2_b3;
    if (tokens <= 96) return launch_q5_a16_mma_r64_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 128) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
