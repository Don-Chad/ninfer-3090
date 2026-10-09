#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N5120K6144;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C48 = Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C56 = Q8A16SlicedKMmaSchedule<56, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C64 = Q8A16SlicedKMmaSchedule<64, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using S32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

} // namespace

Q8Launch select_q8_n5120_k6144(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    // sm_86, measured 2026-09-17 (`q8:5120x6144`, cold, median of 11; us, new vs shipped):
    //   T=32  S32 80.9 vs 96.3 (+19%)
    //   T=40  r32_c64 128.0 vs 150.5 (+18%)  T=56 128.0 vs 169.0 (+32%)  T=64 128.0 vs 197.6 (+54%)
    //   T=80  r32_c96 129.0 vs 145.4 (+13%)
    //   T=160 r64_c96 220.2 vs 269.3 (+22%)  T=192 210.9 vs 271.4 (+29%)
    // T<=24 keeps upstream's eight-warp rungs, which beat every four-warp candidate here, and
    // 97..128 and T>=256 keep their routes as well.
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, S32>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r64_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
