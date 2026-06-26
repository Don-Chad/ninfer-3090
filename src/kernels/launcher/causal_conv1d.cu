// qus::kernels - causal_conv1d launcher: grid/block/stream configuration + kernel launch.
#include "kernels/launcher/causal_conv1d.h"

#include "kernels/kernel/causal_conv1d.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels::detail {
namespace {

int grid_for(std::int64_t n, int block, const char* label) {
    const std::int64_t grid = (n + static_cast<std::int64_t>(block) - 1) / block;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string("causal_conv1d: ") + label +
                                  " grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

} // namespace

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight, Tensor& conv_state,
                                  Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    const std::int64_t n = static_cast<std::int64_t>(C) * static_cast<std::int64_t>(T);

    causal_conv1d_prefill_kernel<<<grid_for(n, kBlock, "prefill output"), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state.data),
        static_cast<__nv_bfloat16*>(out.data), C, T, n);
    CUDA_CHECK(cudaGetLastError());

    causal_conv1d_prefill_state_kernel<<<grid_for(C, kBlock, "prefill state"), kBlock, 0,
                                         stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(conv_state.data),
        C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, Tensor& conv_state,
                                 Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t C = x.ne[0];

    causal_conv1d_decode_kernel<<<grid_for(C, kBlock, "decode"), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(conv_state.data), static_cast<__nv_bfloat16*>(out.data), C);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
