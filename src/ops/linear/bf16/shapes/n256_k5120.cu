#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
Bf16Launch select_bf16_n256_k5120(std::int32_t tokens) {
#if defined(NINFER_SM8X_COMPAT)
    // sm_8x keeps the sliced-K pair this shape has always run there (16 K warps x 8 tokens, then
    // 8 K warps x 16 tokens); the wide MMA and TMA tiles below are RTX 5090 selections and TMA
    // does not exist before sm_90.
    if (tokens <= 76)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 16>, 5120>>;
    return launch_bf16_sliced_k_mma<
        Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8>, 5120>>;
#else
    if (tokens <= 76)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 16>, 5120>>;
    if (tokens <= 160)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8>, 5120>>;
    if (tokens <= 640) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 5120>>;
    if (tokens <= 1024) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
    if (tokens <= 1280)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T32K64S3, 5120>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 5120>>;
#endif
}
} // namespace ninfer::ops::detail
