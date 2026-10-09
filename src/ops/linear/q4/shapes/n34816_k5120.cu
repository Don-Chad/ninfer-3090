#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// Five entries of this table were re-measured on sm_86 2026-09-17 against the wider candidate set
// bench/ops/linear_schedule_bench.cu sweeps (`q4:34816x5120`), cold, median of 11 then 21. Most of
// the 2026-09 sm_86 sweep below survives unchanged, which is the point of recording it; what moved
// (us, new vs old):
//
//   T=1    gemv_r4_w1  129.0 vs gemv_r1_q8 157.7 (+22%) -- at 34816 rows the four-row GEMV's grid
//                                                         is what fills the device, not the
//                                                         one-row kernel's wider per-row tile
//   T=12   ksplit24    200.7 vs ksplit16  329.7 (+64%),  T=16 237.6 vs 305.2 (+28%)
//   T=56   r64_c64     508.9 vs r64_c56   583.7 (+15%),  T=64 501.8 vs r32_c64 693.2 (+38%)
//   T=112  r64_c112    748.5 vs r64_c120 1001.5 (+34%)
//   T=160  r64_c80    1227.8 vs r64_c96  1401.9 (+14%);  T=192 is a tie (0.4%) and keeps c96.
Q4Launch select_q4_n34816_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    // The K-split tile is masked to the capacity rounded up to a multiple of eight, so each
    // capacity below is the narrowest masked tile that still covers its interval.
    if (tokens <= 4) return launch_q4_ksplit<34816, 5120, 4>;
    if (tokens <= 8) return launch_q4_ksplit<34816, 5120, 8>;
    if (tokens <= 24) return launch_q4_ksplit<34816, 5120, 24>;
    // Above the K-split ceiling the narrowest column tile that covers the extent wins: at these
    // row counts the 128-wide tile that served every T >= 17 before this change spends a whole
    // 128-column tile on T = 17. Each bound is a tile width, not a threshold copied from another
    // shape; the measured cost of merging an interval into its neighbour is in the PR body.
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 48) return launch_q4_a16_mma_r64_t48;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t128;
    // Above the hot interval a route is applied once per column tile, so the winner is the tile
    // that splits the extent into the fewest slices, and among those the narrowest: two tiles of
    // 96 cover 129..192, of 112 cover 193..224, of 120 cover 225..240 and of 128 cover 241..256;
    // three tiles of 96 then beat three wider ones up to 288, and 128 carries the 512 and 1024
    // anchors. Each bound is a tile-count edge, not a threshold copied from another shape; the
    // measured cost of folding one of these bands into its neighbour is in the PR body.
    if (tokens <= 160) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 240) return launch_q4_a16_mma_r64_t120;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 288) return launch_q4_a16_mma_r64_t96;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
