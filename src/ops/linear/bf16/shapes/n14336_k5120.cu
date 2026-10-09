#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16A16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using C2   = Bf16A16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C4   = Bf16A16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
#if defined(NINFER_SM8X_COMPAT)
// sm_8x keeps the SIMT ladder and the 64x128 MMA tile this shape has always run there. The RTX 5090
// table below reaches TMA tiles from T=33, and TMA does not exist before sm_90.
using C8  = Bf16A16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C12 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 2>;
using C16 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 4>;
using C20 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 3>;
using C24 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 3>;
using Mma = Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
#endif
} // namespace

Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 5120>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<C2, 5120, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<C4, 5120, 4>>;
#if defined(NINFER_SM8X_COMPAT)
    if (tokens <= 8) return launch_bf16_simt<Bf16ScheduleInstance<C8, 5120, 8>>;
    if (tokens <= 12) return launch_bf16_simt<Bf16ScheduleInstance<C12, 5120, 12>>;
    if (tokens <= 16) return launch_bf16_simt<Bf16ScheduleInstance<C16, 5120, 16>>;
    if (tokens <= 20) return launch_bf16_simt<Bf16ScheduleInstance<C20, 5120, 20>>;
    if (tokens <= 24) return launch_bf16_simt<Bf16ScheduleInstance<C24, 5120, 24>>;
    return launch_bf16_mma<Bf16ScheduleInstance<Mma, 5120>>;
#else
    if (tokens <= 16)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 5120>>;
    if (tokens <= 32) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
    if (tokens <= 64) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 5120>>;
    if (tokens <= 96) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 5120>>;
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 5120>>;
    if (tokens <= 192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 5120>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 5120>>;
#endif
}
} // namespace ninfer::ops::detail
