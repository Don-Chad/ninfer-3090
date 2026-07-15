#include "ops/launcher/scatter.h"

#include "ops/kernel/scatter.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream) {
    constexpr int block        = 256;
    constexpr int vector_block = 128;
    const int d                = src.ne[0];
    const int vision           = src.ne[1];
    const auto src_addr        = reinterpret_cast<std::uintptr_t>(src.data);
    const auto dst_addr        = reinterpret_cast<std::uintptr_t>(dst.data);
    if ((d % 8) == 0 && ((src_addr | dst_addr) & 0xfu) == 0) {
        scatter_bf16x8_kernel<<<vision, vector_block, 0, stream>>>(
            static_cast<const uint4*>(src.data), static_cast<const std::int32_t*>(indices.data),
            static_cast<uint4*>(dst.data), d / 8);
    } else if ((d & 1) == 0 && ((src_addr | dst_addr) & 0x3u) == 0) {
        scatter_bf16x2_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat162*>(dst.data),
            d / 2);
    } else {
        scatter_scalar_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat16*>(dst.data),
            d);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
