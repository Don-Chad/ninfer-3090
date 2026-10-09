#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {

// Re-measured on sm_86 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q5:1024x5120`), cold,
// median of 11 then confirmed at median of 21 with min..p95. This shape had no K-split ladder at
// all -- the other four Q5 shapes do -- and four of its bounds are in the wrong place (us, new vs
// shipped):
//
//   T=2   ksplit2 12.3 vs simt 16.4 (+33%); T=4 14.3 vs 19.5 (+36%); T=6 16.4 vs 21.5 (+31%)
//   T=48  r64_c16 74.8 vs simt 80.9 (+8%)
//   T=128 r64_c32s3 87.0 vs r64_c16 101.4 (+17%); T=160 88.1 vs 101.4 (+15%); T=96 +14%
//   T=512 r32_c128 175.1 vs r64_c32s3 223.2 (+28%)
//   T=768 r64_c128 235.5 vs r32_c128 330.8 (+40%); T=1024 238.6 vs 334.8 (+40%)
//
// The last one matters most: upstream measured the 32-row 128-wide tile as the winner to 1,280
// columns on an RTX 5090 and wrote that bound into the file. On this card it loses 40% at 768,
// which is inside the range a prefill chunk reaches.
//
// 7..32 keeps the SIMT tile (the eight-column K-split rung is 5% ahead at T=8, inside the spread,
// and would cost another compiled instance), and 65..80 keeps the 16-wide tile, which is 26%
// ahead of the 32-wide one there.
Q5Launch select_q5_n1024_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 2) return launch_q5_a16_direct_exact<5120, 2, 4>;
    if (tokens <= 3) return launch_q5_a16_direct_exact<5120, 3, 4>;
    if (tokens <= 4) return launch_q5_a16_direct_exact<5120, 4, 4>;
    if (tokens <= 5) return launch_q5_a16_direct_exact<5120, 5, 4>;
    if (tokens <= 6) return launch_q5_a16_direct_exact<5120, 6, 4>;
    if (tokens <= 32) return launch_q5_a16_simt_r8_t4_w1_g16_s2;
    if (tokens <= 80) return launch_q5_a16_mma_r64_t16_k64_wr16_wt8_s2_a2_b3;
    if (tokens <= 448) return launch_q5_a16_mma_r64_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 704) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
