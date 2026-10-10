#pragma once
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail {
template <class Schedule>
void launch_bf16_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_gemv<Schedule>(bf16_a16_operands(x, w),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                   LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_simt<Schedule>(bf16_a16_operands(x, w),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                   LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_mma<Schedule>(bf16_a16_operands(x, w),
                                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                  LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_sliced_k_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_sliced_k_mma<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_tma_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      LinearIdentityEpilogue{}, stream);
}

#if defined(NINFER_SM8X_COMPAT)
// The cp.async MMA schedule with the same block, warp, stage and raster geometry as a TMA schedule,
// keeping its static-K, token-capacity and row/K-tail wrappers. sm_8x has no TMA.
template <class Schedule>
struct Bf16Sm8xCpAsyncTile;

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Bf16MmaRaster Raster, int RasterGroupRows, int ConsumerKUnroll>
struct Bf16Sm8xCpAsyncTile<
    Bf16A16TmaMmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, Stages,
                          MinBlocksPerSm, Raster, RasterGroupRows, ConsumerKUnroll>> {
    using type = Bf16A16MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, Stages,
                                    MinBlocksPerSm, Cache::cg, Cache::cg,
                                    Bf16MmaFragmentPipeline::PingPong, Raster,
                                    Bf16MmaSwizzle::Xor64, RasterGroupRows>;
};

template <class Schedule, int K, int Capacity, bool ExactTokens>
struct Bf16Sm8xCpAsyncTile<Bf16ScheduleInstance<Schedule, K, Capacity, ExactTokens>> {
    using type = Bf16ScheduleInstance<typename Bf16Sm8xCpAsyncTile<Schedule>::type, K, Capacity,
                                      ExactTokens>;
};

template <class Schedule>
struct Bf16Sm8xCpAsyncTile<Bf16RowTailSchedule<Schedule>> {
    using type = Bf16RowTailSchedule<typename Bf16Sm8xCpAsyncTile<Schedule>::type>;
};

template <class Schedule>
struct Bf16Sm8xCpAsyncTile<Bf16KTailSchedule<Schedule>> {
    using type = Bf16KTailSchedule<typename Bf16Sm8xCpAsyncTile<Schedule>::type>;
};
#endif

// A shape-table tier whose RTX 5090 selection is a TMA schedule. On sm_8x the same tile runs
// through the cp.async MMA kernel instead: the tables that use this were tuned on sm_120 and have
// not been re-measured on the RTX 3090, so the tile is a legal starting point, not a tuned route.
// Shapes measured on sm_8x keep their own NINFER_SM8X_COMPAT tables and do not use this.
template <class Schedule>
void launch_bf16_tma_tile(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
#if defined(NINFER_SM8X_COMPAT)
    launch_bf16_mma<typename Bf16Sm8xCpAsyncTile<Schedule>::type>(x, w, out, stream);
#else
    launch_bf16_tma_mma<Schedule>(x, w, out, stream);
#endif
}
} // namespace ninfer::ops::detail
