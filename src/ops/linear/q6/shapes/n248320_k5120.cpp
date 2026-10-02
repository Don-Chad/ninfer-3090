#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n248320_k5120(std::int32_t tokens) {
    // Swept on sm_86 2026-10-02 with bench/ops/linear_schedule_bench.cu q6:248320x5120 (cold,
    // median of 20, us): the per-row GEMV against the 8-row SIMT tile it replaces,
    //   T=1 1159 vs 1381   T=2 1170 vs 1619   T=3 1338 vs 2475   T=4 1593 vs 2509,
    // T=1 streaming the 993 MB head at 857 GB/s. From T=5 the 64-row MMA holds at ~2.34 ms while
    // the SIMT tiles the table used to pick ran 3.1-6.1 ms (T=5 2340 vs 5209, T=7 2380 vs 3106).
    if (tokens == 1) return launch_q6_gemv_t1;
    if (tokens == 2) return launch_q6_gemv_t2;
    if (tokens == 3) return launch_q6_gemv_t3;
    if (tokens == 4) return launch_q6_gemv_t4;
    if (tokens <= 16) return launch_q6_mma_r64_c16_k128;
    if (tokens <= 24) return launch_q6_mma_r64_c24_k128;
    if (tokens <= 32) return launch_q6_mma_r64_c32_k128;
    if (tokens <= 48) return launch_q6_mma_r64_c48_k128;
    return launch_q6_mma_r64_c128;
}

} // namespace ninfer::ops::detail
