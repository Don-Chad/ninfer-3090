#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n248320_k5120(std::int32_t tokens) {
    // Swept on sm_86 2026-10-02 with bench/ops/linear_schedule_bench.cu q6:248320x5120 (cold,
    // median of 9-31, us). The per-row GEMV streams the 993 MB head at 857 GB/s at T=1 but grows
    // with T (T=3 1382, T=4 1651); the small-T MMA is flat to T=8 and replaces both it and the
    // 64-row MMA, which held ~2.4 ms from T=5 (T=4 1154 vs 1651, T=8 1175 vs 2450, T=16 1650 vs
    // 2420). T=2 is a tie (GEMV 1143-1198, small-T 1157-1198) and T=1 stays on the GEMV (1160 vs
    // 1304). A speculative verify is T=4 (MTP3) or T=8 (DFlash2 K=7). The four-tile variant takes
    // 17..32, where an MTP3 cohort of five to eight lanes verifies (T=17 1793 vs 2602 for the
    // 64-row MMA, T=24 2063 vs 2620, T=32 2386 vs 2891); from T=40 the 64-row MMA leads again.
    if (tokens == 1) return launch_q6_gemv_t1;
    if (tokens == 2) return launch_q6_gemv_t2;
    if (tokens <= 8) return launch_q6_small_t_c8;
    if (tokens <= 16) return launch_q6_small_t_c16;
    if (tokens <= 32) return launch_q6_small_t_c32;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
