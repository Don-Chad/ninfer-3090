#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q4:1024x5120`), cold,
// median of 11 then confirmed at median of 31 with min..p95. The table that arrived with the
// 2026-09-17 catch-up was an RTX 5090 sweep and is wrong here in four places:
//
//   * It gave 2..56 to a four-row SIMT tile. The K-split capacity kernels this shape did not
//     register are 33-95% faster over most of that range (us): T=8 ksplit8 14.3 vs simt 21.5;
//     T=16 ksplit24 20.5 vs 31.7; T=24 ksplit24 22.5 vs 44.0; T=32 ksplit32 29.7 vs 55.3.
//     From 33 the 16x32 tile it already used above 56 carries the band (T=48 61.4 vs 72.7,
//     T=56 62.5 vs 82.9). The eight-warp 32x32 tile is 1-3% ahead of it at 40..56 and is not
//     registered for that: a second compiled instance is not worth 1-3% on one band.
//   * 193..320 prefers the 32-row tile over the 16-row one (T=320 106.5 vs 129.0, +21%).
//   * 321..1344 went to a 32x64 tile. Its wave profile collapses as the extent grows: T=384
//     r64_c80 122.9 vs 139.3, T=640 r64_c64 153.6 vs 171.0, T=1024 r64_c128 232.4 vs 357.4 (+54%).
Q4Launch select_q4_n1024_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 8) return launch_q4_ksplit<1024, 5120, 8>;
    if (tokens <= 24) return launch_q4_ksplit<1024, 5120, 24>;
    if (tokens <= 32) return launch_q4_ksplit<1024, 5120, 32>;
    if (tokens <= 192) return launch_q4_a16_mma_r16_t32_k64_wr16_wt8_s2_a2_b2;
    if (tokens <= 320) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 448) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 768) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
