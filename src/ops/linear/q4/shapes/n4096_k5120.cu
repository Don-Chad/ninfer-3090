#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q4:4096x5120`), cold,
// median of 11 and confirmed at median of 31. Two bands of the RTX 5090 table the catch-up brought
// in are wrong on this card, both by a lot:
//
//   * 9..16 took the capacity-16 tile; capacity 24 is faster at both ends (T=12 35.8 vs 47.1 us,
//     +31%; T=16 41.0 vs 45.1). The capacity is a mask, not a promise, so a wider tile covering a
//     narrower extent is legal and here it is cheaper.
//   * 33..320 went to the 32x32 tiles. On sm_86 the 64-row tiles win that whole range by 25-58%,
//     and which one wins is set by the tile-count edge, exactly as on n34816 (us):
//     T=48 c48 102.4 vs 141.3; T=64 c64 111.6 vs 142.3; T=80 c80 122.9 vs 173.1; T=96 c96 131.1
//     vs 165.9; T=128 c64 153.6 vs 191.5; T=192 c96 193.5 vs 278.5; T=224 c112 200.7 vs 299.0;
//     T=256 c128 232.4 vs 366.6; T=320 c64 290.8 vs 430.1.
//
// The K-split T=1 entry replaces the GEMV, which is 14% slower here (24.6 vs 21.5 us); at this row
// count the GEMV's one-warp-per-row grid no longer fills the device.
Q4Launch select_q4_n4096_k5120(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<4096, 5120, 8>;
    if (tokens <= 24) return launch_q4_ksplit<4096, 5120, 24>;
    if (tokens <= 32) return launch_q4_ksplit<4096, 5120, 32>;
    if (tokens <= 48) return launch_q4_a16_mma_r64_t48;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 160) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 384) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
