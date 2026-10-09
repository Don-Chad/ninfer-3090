#include "ops/linear_add/q8/q8_add_epilogue.cuh"
#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_mma_launch.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
void launch_variant(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0]},
        Q8AddMmaEpilogue{}, stream);
}

} // namespace

void q8_linear_add_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 32, 64, 32, 16, 2, 4>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c48_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 48, 64, 32, 16, 2, 4>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 112, 64, 32, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 80, 64, 48, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c32_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 32, 64, 64, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c48_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 48, 64, 64, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 80, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 112, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

// Exact-group-scale twins. `with_exact_group_scale` restates the tile above it, so the pair can
// never drift apart in anything but where the Q8G32 scale is applied -- and, where it is spelled,
// MIN_BLOCKS. The exact body holds an FP32 group partial and this tile's row scales, and left at
// the default tile's MIN_BLOCKS ptxas spends the slack on registers and drops a resident block
// (`cuobjdump -res-usage`, sm_86): r32_c64 92 -> 136 registers and 4 blocks -> 3, r32_c96 90 ->
// 136 and 3 -> 2, r64_c64 120 -> 209 and 3 -> 2. The three below are raised to the block count
// the shared-memory footprint allows, which is the one their default twin already runs at; every
// other tile here keeps its own because it was already at that count. None of them spill.
void q8_linear_add_mma_exact_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule =
        Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>::with_exact_group_scale::with_min_blocks<4>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule =
        Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>::with_exact_group_scale::with_min_blocks<3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r48_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r48_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule =
        Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>::with_exact_group_scale::with_min_blocks<3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 112, 64, 64, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                            cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
