#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q4:5120x6144`), cold,
// median of 11 and confirmed at median of 25. The shape is new in the 2026-09-17 catch-up, so none
// of it had been seen on this card. Its K-split ladder and its 33..64 band hold up; the rest does
// not (us, chosen vs the route that shipped):
//
//   T=1    ksplit4      27.6 vs gemv     31.7  (+15%)
//   T=12   ksplit24     49.2 vs ksplit16 70.7  (+44%),  T=16 55.3 vs 59.4 (+7%)
//   T=32   r32_c32w     91.1 vs ksplit32 100.4 (+10%)
//   T=80   r64_c80     143.4 vs r32_c32 227.3  (+59%),  T=96 r64_c96 159.7 vs 216.1 (+35%)
//   T=112  r64_c64     194.6 vs r32_c64 335.9  (+73%),  T=128 188.4 vs 334.8 (+78%)
//   T=160  r64_c80     216.1 vs r32_c64 337.9  (+56%),  T=192 r64_c96 234.5 vs 335.9 (+43%)
//   T=224  r64_c112    245.8 vs r64_c128 273.4 (+11%),  T=320 r64_c80 417.8 vs 492.5 (+18%)
//
// 33..64 keeps the 32x32 tile: the 64-row tiles are 2-8% ahead there and that is inside this
// bench's own run-to-run spread, so the band is left where upstream put it.
Q4Launch select_q4_n5120_k6144(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<5120, 6144, 8>;
    if (tokens <= 24) return launch_q4_ksplit<5120, 6144, 24>;
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt8_s4_a4_b2;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 160) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 320) return launch_q4_a16_mma_r64_t80;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
