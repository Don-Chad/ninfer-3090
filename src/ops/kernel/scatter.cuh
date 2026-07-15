#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void scatter_bf16x8_kernel(const uint4* src, const std::int32_t* indices, uint4* dst,
                                      std::int32_t vectors_per_column) {
    const std::int32_t src_col  = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col  = indices[src_col];
    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * vectors_per_column;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * vectors_per_column;
    for (std::int32_t vector = static_cast<std::int32_t>(threadIdx.x); vector < vectors_per_column;
         vector += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + vector] = src[src_base + vector];
    }
}

__global__ void scatter_bf16x2_kernel(const __nv_bfloat162* src, const std::int32_t* indices,
                                      __nv_bfloat162* dst, std::int32_t pairs) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col = indices[src_col];

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * pairs;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * pairs;
    for (std::int32_t pair = static_cast<std::int32_t>(threadIdx.x); pair < pairs;
         pair += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + pair] = src[src_base + pair];
    }
}

__global__ void scatter_scalar_kernel(const __nv_bfloat16* src, const std::int32_t* indices,
                                      __nv_bfloat16* dst, std::int32_t d) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col = indices[src_col];

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * d;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * d;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < d;
         row += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + row] = src[src_base + row];
    }
}

} // namespace ninfer::ops
