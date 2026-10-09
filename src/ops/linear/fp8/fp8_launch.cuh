#pragma once
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
template <class Geometry, class Schedule>
void fp8_linear_a16_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, int Capacity, class Schedule, bool Exact = false>
void fp8_linear_a16_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, Capacity, Exact>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void fp8_linear_a16_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void fp8_linear_a16_sliced_k(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

#if defined(NINFER_SM8X_COMPAT)
// The A8 MMA emits mma.sync...kind::f8f6f4 and the TMA route needs sm_90+, so sm_8x must not
// instantiate either. Shapes still name their SM120 routes; A8 is never admitted there
// (kFp8TextPolicy is A16Only) and this turns a mis-selection into a precise error. See
// fp8_sm86_stubs.cpp.
template <class Geometry, class Schedule>
void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    throw std::runtime_error("FP8 A8 execution requires an sm_100a or sm_120a GPU");
}

template <class Geometry, class Schedule>
void launch_fp8_a8_tma(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    throw std::runtime_error("FP8 A8 execution requires an sm_100a or sm_120a GPU");
}
#else
template <class Geometry, class Schedule>
void launch_fp8_a8(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                   cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Geometry, class Schedule>
void launch_fp8_a8_tma(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                       cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_fp8_a8_tma_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream, scratch.partials);
}
#endif
} // namespace ninfer::ops::detail
