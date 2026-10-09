#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q4:7168x5120`), cold,
// median of 11 and confirmed at median of 25. This shape carried the most detailed 5090 table of
// the group and almost none of it holds here; the 129..192 and 321..384 bands are the exceptions
// and are kept unchanged. Measured (us, new vs the route that shipped):
//
//   T=1    ksplit4 31.7 vs gemv 37.9      (+19%)
//   T=12   ksplit24 48.1 vs ksplit16 79.9 (+66%),  T=16 55.3 vs 74.8 (+35%)
//   T=40   r32_c64 148.5 vs r32_c32w 196.6 (+32%), T=64 149.5 vs 193.5 (+29%)
//   T=80   r64_c80 177.2 vs r32_c32 280.6 (+58%),  T=96 r64_c96 198.7 vs 272.4 (+37%)
//   T=112  r64_c112 205.8 vs r32_c64 313.3 (+52%), T=128 r64_c64 210.9 vs r64_c48 318.5 (+51%)
//   T=224  r64_c112 353.3 vs r64_c96 499.7 (+41%), T=256 r64_c128 387.1 vs 511.0 (+32%)
//   T=320  r64_c64 482.3 vs r64_c128 615.4 (+28%), T=512 r64_c128 684.0 vs r64_c96 896.0 (+31%)
//
// The 25..32 move from the eight-warp 32x32 tile to the four-warp one is the one entry here below
// the 10% bar this sweep used (93.2 vs 98.3, +5.5%); it is taken because it also merges a band.
Q4Launch select_q4_n7168_k5120(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<7168, 5120, 8>;
    if (tokens <= 24) return launch_q4_ksplit<7168, 5120, 24>;
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 320) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
