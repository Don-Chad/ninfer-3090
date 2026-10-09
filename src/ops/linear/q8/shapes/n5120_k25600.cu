#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N5120K25600;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C8       = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S48 =
    Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;

} // namespace

Q8Launch select_q8_n5120_k25600(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    // sm_86, measured 2026-09-17 (bench/ops/linear_schedule_bench.cu `q8:5120x25600`, cold,
    // median of 11; us, new vs the route that shipped):
    //   T=24  S32 213.0 vs 257.0 (+21%)     T=32  S32 256.0 vs 343.0 (+34%)
    //   T=40  S48 345.1 vs 390.1 (+13%)     T=48  S48 389.1 vs 473.1 (+22%)
    //   T=56  r32_c64 475.1 vs 506.9 (+7%)  T=64  455.7 vs 471.0 (+3%)
    //   T=80  r32_c96 514.0 vs 725.0 (+41%) T=96  521.2 vs 742.4 (+42%)
    //   T=112 r32_c128 617.5 vs 760.8 (+23%) T=128 644.1 vs 788.5 (+22%)
    //   T=160 r64_c96 900.1 vs 1076.2 (+20%) T=192 877.6 vs 1085.4 (+24%)
    // T<=16 and T>=256 are upstream's values and are best here too. The two `launch_tiled`
    // row-split tiles this shape introduced are what 65..128 loses 22-42% to; they are no longer
    // selected anywhere and are gone with them.
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, S32>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, S48>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r64_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
