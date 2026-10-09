#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N5120K10240;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C48 = Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;

using S32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

} // namespace

Q8Launch select_q8_n5120_k10240(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    // sm_86, measured 2026-09-17 (`q8:5120x10240`, cold, median of 11; us, new vs shipped):
    //   T=24  S32 92.2 vs 109.6 (+19%)        T=32 S32 103.4 vs 138.2 (+34%)
    //   T=56  r32_c64 196.6 vs 290.8 (+48%)   T=64 193.5 vs 293.9 (+52%)
    //   T=80  r32_c96 214.0 vs 294.9 (+38%)   T=96 230.4 vs 298.0 (+29%)
    //   T=112 r32_c128 259.1 vs 299.0 (+15%)
    //   T=160 r64_c96 373.8 vs 451.6 (+21%)   T=192 346.1 vs 449.5 (+30%)
    // T<=16, 33..48 and T>=256 keep upstream's routes and measure best here.
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, S32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r64_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
