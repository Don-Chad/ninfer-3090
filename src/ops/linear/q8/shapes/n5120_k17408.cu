#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N5120K17408;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C48 = Q8A16SlicedKMmaSchedule<48, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C56 = Q8A16SlicedKMmaSchedule<56, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C64 = Q8A16SlicedKMmaSchedule<64, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S24 = Q8A16SlicedKMmaSchedule<24, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S48 =
    Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;

} // namespace

Q8Launch select_q8_n5120_k17408(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    // sm_86, measured 2026-09-17 (`q8:5120x17408`, cold, median of 11; us, new vs shipped):
    //   T=24  S24 137.2 vs 188.4 (+37%) -- four K warps, not eight
    //   T=32  S32 189.4 vs 257.0 (+36%)
    //   T=48  S48 284.7 vs 370.7 (+30%) -- runtime-active staging, not active-only
    //   T=56  r32_c64 326.7 vs 446.5 (+37%)   T=64 331.8 vs 435.2 (+31%)
    //   T=160 r64_c96 625.7 vs 746.5 (+19%)   T=192 589.8 vs 745.5 (+26%)
    // T<=16, 33..40, 65..128 and T>=256 keep upstream's routes: measured within 9% there, which is
    // this bench's own run-to-run spread on this shape.
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, S24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, S32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, S48>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r64_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
