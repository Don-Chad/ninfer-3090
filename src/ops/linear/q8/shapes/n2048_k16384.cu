#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"
#if !defined(NINFER_SM8X_COMPAT)
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#endif

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N2048K16384;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;

#if defined(NINFER_SM8X_COMPAT)
// sm_86 ladder, measured 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q8:2048x16384`),
// cold, median of 11. Three things separate it from the sm_120 ladder below, and all three recur
// on every Q8 shape this bench swept:
//
//   * `cg` activations lose to `ca`. That is upstream's 028eb61e applied to this card, and it is
//     backwards here: T=24 79.9 vs 91.1 us (+14%), T=32 87.0 vs 103.4 (+19%), T=48 129.0 vs 143.4
//     (+11%). The L1 that `cg` bypasses is where the staged activation slab wants to live.
//   * Eight K warps fit up to a 32-column tile, so a blanket four-warp fallback gives away the
//     narrow rungs: T=8 52.2 vs 60.4 (+16%), T=12 56.3 vs 68.6 (+22%), T=16 58.4 vs 71.7 (+23%).
//   * The 49..128 composite -- 32-column capacity tiles plus a tail, this fork's own sm_8x stand-in
//     for the grouped medium kernels -- is beaten by a plain 32x64 MMA tile from 65 up and by the
//     capacity ladder below that: T=56 141.3 vs 193.5 (+37%), T=64 173.1 vs 204.8 (+18%),
//     T=96 263.2 vs 304.1 (+16%), T=112 276.5 vs 372.7 (+35%), T=128 258.0 vs 404.5 (+57%).
//     The composite is therefore gone; 65..128 simply joins the 129..384 band.
using S8 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using S24 =
    Q8A16SlicedKMmaSchedule<24, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S32 =
    Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S40 =
    Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S48 = Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using S56 =
    Q8A16SlicedKMmaSchedule<56, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S64 =
    Q8A16SlicedKMmaSchedule<64, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
#else
using C8       = Q8A16SlicedKMmaSchedule<8, 16, 2, 1, Access::Shared>;
using Wide32   = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Access::Shared>;
using Medium16 = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Access::Shared>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C24 =
    Q8A16SlicedKMmaSchedule<24, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C32 =
    Q8A16SlicedKMmaSchedule<32, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8A16SlicedKMmaSchedule<40, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg,
                                    Stage::RuntimeActive>;
using C48 =
    Q8A16SlicedKMmaSchedule<48, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;

void launch_grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Q8A16GroupedSlicedKMmaSchedule<128, 2, 4, 2, 1, 0, Cache::cg, Cache::cg, true, false>;
    launch_q8_a16_grouped_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        LinearIdentityEpilogue{}, stream);
}
#endif
} // namespace

Q8Launch select_q8_n2048_k16384(std::int32_t tokens) {
    if (tokens == 1) return launch_q8_a16_gemv_r4_w1_k16384;
#if defined(NINFER_SM8X_COMPAT)
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, S8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, S16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, S24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, S32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, S40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, S48>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, S56>;
    if (tokens <= 64) return launch_q8_a16_sliced<Geometry, 64, S64>;
#else
    // Selected on RTX 5090.
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 64) return launch_q8_a16_sliced<Geometry, 32, Wide32>;
    if (tokens <= 80) return launch_q8_a16_sliced<Geometry, 16, Medium16>;
    if (tokens <= 128) return launch_grouped;
#endif

    // Broad throughput regions; each selected MMA handles its own complete and partial tiles.
    // On sm_86 the 32x64 tile also carries 65..128 (see the ladder note above); T=144..192 was
    // re-measured there and keeps this bound (r32_c96 is 4% ahead at 144/160 and behind at 192,
    // inside this bench's spread).
    if (tokens <= 384) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 480) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 640) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 704) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 960) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 1344) return launch_q8_a16_mma_r128_t64;
    if (tokens <= 1680) return launch_q8_a16_mma_r128_t80;
    if (tokens <= 2016) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 2112) return launch_q8_a16_mma_r96_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
