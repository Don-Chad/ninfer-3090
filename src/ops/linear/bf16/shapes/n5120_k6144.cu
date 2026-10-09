#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16A16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
using C2   = Bf16A16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C4   = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
#if defined(NINFER_SM8X_COMPAT)
// sm_8x keeps the SIMT ladder and the 64x128 MMA tile this shape has always run there. The RTX 5090
// table below reaches TMA tiles from T=65, and TMA does not exist before sm_90.
using C8  = Bf16A16SimtSchedule<4, 1, 4, 16, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C12 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 2>;
using C16 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 4>;
using C20 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C24 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 4>;
using C28 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 4>;
using C32 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using Mma = Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
#endif
} // namespace

Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 6144>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<C2, 6144, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<C4, 6144, 4>>;
#if defined(NINFER_SM8X_COMPAT)
    if (tokens <= 8) return launch_bf16_simt<Bf16ScheduleInstance<C8, 6144, 8>>;
    if (tokens <= 12) return launch_bf16_simt<Bf16ScheduleInstance<C12, 6144, 12>>;
    if (tokens <= 16) return launch_bf16_simt<Bf16ScheduleInstance<C16, 6144, 16>>;
    if (tokens <= 20) return launch_bf16_simt<Bf16ScheduleInstance<C20, 6144, 20>>;
    if (tokens <= 24) return launch_bf16_simt<Bf16ScheduleInstance<C24, 6144, 24>>;
    if (tokens <= 28) return launch_bf16_simt<Bf16ScheduleInstance<C28, 6144, 28>>;
    if (tokens <= 32) return launch_bf16_simt<Bf16ScheduleInstance<C32, 6144, 32>>;
    return launch_bf16_mma<Bf16ScheduleInstance<Mma, 6144>>;
#else
    if (tokens <= 32)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 6144>>;
    if (tokens <= 64) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K192S2, 6144>>;
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 6144>>;
    if (tokens <= 192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 6144>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 6144>>;
#endif
}
} // namespace ninfer::ops::detail
