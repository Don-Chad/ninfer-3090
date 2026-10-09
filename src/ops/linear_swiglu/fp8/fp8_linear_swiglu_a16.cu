#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry              = Fp8N34816K5120;
constexpr int kIntermediate = Geometry::kOutputRows / 2;
} // namespace

void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    using Rows = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, kIntermediate>;
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, 4>>(
        fp8_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate}, Fp8SwiGluEpilogue{},
        stream, Rows{});
}

void fp8_linear_swiglu_matrix_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    const auto p = fp8_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const auto sliced = [&]<int T, int W, int Stages>() {
        using S = Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, 5120>;
        launch_fp8_a16_sliced_k_mma<S>(p, output, Fp8SwiGluEpilogue{}, stream,
                                       SwiGluRowMajorMmaRows<S>{});
    };
    const auto mma = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        launch_fp8_a16_mma<S>(p, output, SwiGluRowMajorMmaEpilogue{}, stream,
                              SwiGluRowMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 8) return sliced.template operator()<8, 4, 1>();
    if (x.ne[1] <= 16) return sliced.template operator()<16, 4, 1>();
    if (x.ne[1] <= 24) return sliced.template operator()<32, 8, 1>();
    if (x.ne[1] <= 32) return sliced.template operator()<32, 4, 1>();
    if (x.ne[1] <= 64)
        return mma.template operator()<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    if (x.ne[1] <= 96)
        return mma.template operator()<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>();
    mma.template operator()<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}
} // namespace ninfer::ops::detail
