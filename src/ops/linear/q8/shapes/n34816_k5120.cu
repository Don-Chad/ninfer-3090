#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N34816K5120;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using W8 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using W16 =
    Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

} // namespace

// sm_86, measured 2026-09-17 (`q8:34816x5120`, cold, median of 11; us, new vs shipped):
//   T=1   W8 219.1 vs 242.7 (+11%)   T=8 222.2 vs 228.4 (+3%)
//   T=12  W16 234.5 vs 281.6 (+20%)   T=16 235.5 vs 284.7 (+21%) -- eight K warps fit here
//   T=32  S32 416.8 vs 629.8 (+51%)
//   T=40  r48_c64 471.0 vs 828.4 (+76%)    T=56 477.2 vs 1279.0 (2.68x)
//   T=80  r128_c80 573.4 vs 780.3 (+36%)   T=96 r96_c96 644.1 vs 785.4 (+22%)
//   T=160 r128_c80 1006.6 vs 1541.1 (+53%) T=192 r96_c96 1118.2 vs 1556.5 (+39%)
// T=17..24, 41..48 and 97..128 keep upstream's routes and are best here; in particular the
// 48-column k128 tile at 41..48 beats every general tile by 19%, which is why that one band
// stays inside an otherwise replaced range.
Q8Launch select_q8_n34816_k5120(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, W8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, W16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, S32>;
    if (tokens <= 40) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 48) return launch_q8_a16_mma_r64x16_t48_k128_a1;
    if (tokens <= 64) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 80) return launch_q8_a16_mma_r128_t80;
    if (tokens <= 96) return launch_q8_a16_mma_r96_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 160) return launch_q8_a16_mma_r128_t80;
    if (tokens <= 192) return launch_q8_a16_mma_r96_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
