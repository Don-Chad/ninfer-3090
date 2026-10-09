#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N2048K4096;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
// Four K warps on sm_8x (q8_instances.cuh, Q8SlicedKSm8xFourWarp): the sixteen- and eight-warp tiles from C16 and C40
// up do not fit its static shared memory.
template <int KWarps, int TileTokens, int MinBlocks, Access A, Cache AC, Cache WC, Stage S>
using Tile = Q8SlicedKSm8xFourWarp<KWarps, TileTokens, MinBlocks, A, AC, WC, S>;
using C4   = Tile<16, 8, 1, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C8   = Tile<16, 8, 1, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16  = Tile<16, 16, 1, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24  = Tile<8, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32  = Tile<8, 32, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40  = Tile<8, 40, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C48  = Tile<8, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C56  = Tile<8, 56, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C64  = Tile<8, 64, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
} // namespace

Q8Launch select_q8_n2048_k4096(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_a16_sliced<Geometry, 64, C64>;
    if (tokens < 896) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
