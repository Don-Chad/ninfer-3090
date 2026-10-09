#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_simt_launch.cuh"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_gemv_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kHidden    = 5120;

using Q4GdnSimtR8T4Schedule = Q4A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8T8Schedule = Q4A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_gemv<q4_instances::GemvR1W8K5120>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_simt<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

template <std::int32_t Capacity>
void launch_q4_sliced_exact(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, kHidden, Capacity>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

void launch_q4_sliced_band(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 GDN K-split requires padded K == hidden");
    }
    switch (x.ne[1]) {
    case 7:
        launch_q4_sliced_exact<7>(x, weight, out, stream);
        return;
    case 8:
        launch_q4_sliced_exact<8>(x, weight, out, stream);
        return;
    case 9:
        launch_q4_sliced_exact<9>(x, weight, out, stream);
        return;
    case 10:
        launch_q4_sliced_exact<10>(x, weight, out, stream);
        return;
    case 11:
        launch_q4_sliced_exact<11>(x, weight, out, stream);
        return;
    case 12:
        launch_q4_sliced_exact<12>(x, weight, out, stream);
        return;
    default:
        throw std::invalid_argument("Q4/Q5 GDN K-split band covers T in [7,12]");
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, out, stream);
        return;
    case 2:
    case 3:
    case 4:
        launch_q4_simt<Q4GdnSimtR8T4Schedule>(x, weight, out, stream);
        return;
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
        // The K-split MMA arms all 8 warps of a CTA onto K instead of waiting out the weight stream
        // of a row. Complete-op measurement (both parents, one graph, one probe run per column
        // count) at T=9..12: the split form with this parent is 101.6-105.7 us against 120.1 us for
        // the grouped kernel R6 chose, while at T=13 the grouped kernel wins again (120.1 against
        // 126.2), so the band ends at 12.
        launch_q4_sliced_band(x, weight, out, stream);
        return;
    default:
        if (x.ne[1] <= 15) {
            launch_q4_simt<Q4GdnSimtR8T8Schedule>(x, weight, out, stream);
            return;
        }
        throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
    }
}

auto q5_projection_output(Tensor& first, Tensor& second) {
    return LinearBf16SplitOutput2<kValueRows>{
        {static_cast<__nv_bfloat16*>(first.data), std::int64_t(first.nb[1] / sizeof(__nv_bfloat16)),
         0},
        {static_cast<__nv_bfloat16*>(second.data),
         std::int64_t(second.nb[1] / sizeof(__nv_bfloat16)), 0}};
}

// Where the Q5 half hands over from the split4 route to simt_r8_c8.
//
// 8, measured. split4 beats the grouped c8 tile that used to be routed at these widths by 5.9% at
// width 7 and 13.2% at width 8, with the two spreads disjoint at both, and collapses at 9 on a
// register cliff. The route table now sends {1,8} to IndependentDirectFixed and the c8 band is
// gone. Numbers and method are in q4_q5_gdn_input_plan.cpp; the reason 9 collapses is on
// launch_q5_split4_exact below. simt_r8_c8 keeps widths 9..15, which the route table does not
// reach but the schedule bench does.
constexpr std::int32_t kQ5Split4LastCols = 8;

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK5120>(q5_linear_operands(x, weight),
                                                           q5_projection_output(value, z),
                                                           LinearIdentityEpilogue{}, stream);
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    using Schedule = Q5A16DirectSimtSchedule<1, Cols, 4, 4, 10, kHidden, true>;
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, weight),
                                        q5_projection_output(value, z), LinearIdentityEpilogue{},
                                        stream);
}

// Instantiated to 16, not 6, and the extra ten are the point.
//
// This kernel is the highest-parallelism thing in the Op by an order of magnitude: one block per
// output row with four warps splitting K and reducing through s_part, so 12,288 blocks and 49,152
// warps -- 12.5 machine-fulls, against the grouped MMA tile's 0.26. TODO section 2c establishes
// that the narrow-extent kernels here are parallelism-starved rather than bandwidth-bound, and this
// is the one route that is not.
//
// It used to throw above T=6, which is why `launch_q5` fell back to the 3.12-machine-full
// `simt_r8_c8` at exactly width 7 -- and that is where the schedule bench's `independent` column
// jumps 188.4 -> 330.8 us for one extra column, with nothing else changing at that boundary (the q4
// half moved to R8C8 back at T=4). When this cycle measured "extend the direct route to {{1, 15}}"
// and recorded -3.6% at width 7, it was measuring `simt_r8_c8`; split4 was never tested above 6.
//
// Widening it is instantiation, not new code: the kernel is already generic in its column count
// (`acc[kTt]`, `s_part[4][kTt]`, and a `tt` loop over `x + tt * kStride`). Watch the register
// budget rather than the shared arena -- it carries `__launch_bounds__(128, 10)`, so 51 registers
// per thread, and `acc[kTt]` costs one register per column; past roughly T=10 expect spills, which
// will look like a loss for a reason that is not the geometry.
void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, value, z, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, value, z, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, value, z, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, value, z, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, value, z, stream);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, value, z, stream);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, value, z, stream);
        return;
    default:
        throw std::invalid_argument("GDN Q5 split4 requires T in [2,8]");
    }
}

template <int kColsPerTile>
void launch_q5_simt_cols(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream) {
    using Schedule = Q5A16SimtSchedule<8, kColsPerTile, 1, 16, 2, Cache::ca, 1>;
    launch_q5_a16_simt<Schedule>(q5_linear_operands(x, weight), q5_projection_output(value, z),
                                 LinearIdentityEpilogue{}, stream);
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= kQ5Split4LastCols) {
        launch_q5_split4_exact(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 12) {
        // c4 SIMT: one output row per warp, up to four columns per column tile, with the quantized
        // weight planes staged in shared memory and activations read from the input tensor.
        // Retained in this interval on complete-Op measurements; both shapes are legal at every
        // count in [2,15].
        launch_q5_simt_cols<4>(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 15) {
        launch_q5_simt_cols<8>(x, weight, value, z, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
}

void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule = Q4GdnSimtR8T4Schedule;
    using Q5Schedule = Q5A16DirectSimtSchedule<1, 4, 4, 4, 10, kHidden, true>;
    // Q5 and Q4 publish disjoint rows; Q4 joins the Q5 producer at kernel exit.
    launch_q5_a16_direct_simt<Q5Schedule, true>(q5_linear_operands(x, value_z_weight),
                                                q5_projection_output(value, z),
                                                LinearIdentityEpilogue{}, stream);
    launch_q4_a16_simt<Q4Schedule, false, true, true>(
        q4_linear_operands(x, qk_weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(qk.data),
                                static_cast<std::int64_t>(qk.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

} // namespace

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 4) {
        launch_t4_pdl(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    launch_q4(x, qk_weight, qk, stream);
    launch_q5(x, value_z_weight, value, z, stream);
}

} // namespace ninfer::ops::detail
