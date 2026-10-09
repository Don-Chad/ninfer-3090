#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q4:6144x5120`), cold,
// median of 11 and confirmed at median of 25. The 25..32 and 321..384 bands of the RTX 5090 table
// survive; four others do not (us):
//
//   T=1    ksplit4 27.6 vs gemv 32.8  (+19%)  -- the one-warp-per-row GEMV no longer fills 6144
//                                                rows worth of grid on this card
//   T=12   ksplit24 46.1 vs ksplit16 73.7 (+60%), T=16 52.2 vs 66.6 (+28%)
//   T=40   r32_c64 142.3 vs r32_c32 173.1 (+22%), T=64 143.4 vs 165.9 (+16%)
//   T=80   r64_c80 178.2 vs r32_c32 223.2 (+25%), T=96 r64_c96 196.6 vs 222.2 (+13%)
//   T=112  r64_c112 203.8 vs r32_c64 286.7 (+41%), T=128 r64_c64 208.9 vs 316.4 (+52%)
//   T=256  r64_c64 347.1 vs r64_c128 380.9 (+10%), T=320 414.7 vs 461.8 (+11%)
//
// The catch-up's `384 < T <= 640 -> 64x64` window is the one route here that is measurably
// backwards on sm_86: at its own anchor the 128-wide tile wins, T=512 594.9 vs 682.0 (+15%) and
// T=640 685.1 vs 807.9 (+18%), so the window is removed rather than moved.
Q4Launch select_q4_n6144_k5120(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<6144, 5120, 8>;
    if (tokens <= 24) return launch_q4_ksplit<6144, 5120, 24>;
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 320) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
