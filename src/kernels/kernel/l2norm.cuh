#pragma once

// qus::kernels - l2norm kernel. One CUDA block handles one row, reducing over ne[0].

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__launch_bounds__(128) __global__ void l2norm_kernel(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                      std::int32_t d, std::int64_t rows,
                                                      float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    const std::int64_t d64 = static_cast<std::int64_t>(d);

    float sum = 0.0f;
    for (std::int64_t i = threadIdx.x; i < d64; i += blockDim.x) {
        const float xv = __bfloat162float(x[base + i]);
        sum += xv * xv;
    }

    __shared__ float scratch[128];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] + eps);
    for (std::int64_t i = threadIdx.x; i < d64; i += blockDim.x) {
        const std::int64_t idx = base + i;
        const float v = __bfloat162float(x[idx]) * inv;
        out[idx] = __float2bfloat16_rn(v);
    }
}

} // namespace qus::kernels
