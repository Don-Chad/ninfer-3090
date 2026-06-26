#pragma once

// qus::kernels - causal_conv1d kernel: depthwise causal k=4 with fused SiLU.
// SiLU is computed as x / (1 + exp(-x)) in fp32, with no polynomial approximation.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ float causal_conv1d_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void causal_conv1d_prefill_kernel(const __nv_bfloat16* x,
                                             const __nv_bfloat16* weight,
                                             const __nv_bfloat16* conv_state,
                                             __nv_bfloat16* out, std::int32_t C,
                                             std::int32_t T, std::int64_t n) {
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64 = static_cast<std::int64_t>(C);

    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t c = static_cast<std::int32_t>(i % C64);
        const std::int32_t t = static_cast<std::int32_t>(i / C64);
        float acc = 0.0f;
        for (std::int32_t j = 0; j < 4; ++j) {
            const std::int32_t src_t = t - 3 + j;
            const std::int64_t value_idx =
                (src_t < 0)
                    ? static_cast<std::int64_t>(3 + src_t) * C64 + c
                    : static_cast<std::int64_t>(src_t) * C64 + c;
            const __nv_bfloat16 xv = (src_t < 0) ? conv_state[value_idx] : x[value_idx];
            const float wv = __bfloat162float(weight[static_cast<std::int64_t>(j) * C64 + c]);
            acc += wv * __bfloat162float(xv);
        }
        out[i] = __float2bfloat16_rn(causal_conv1d_silu_f32(acc));
    }
}

__global__ void causal_conv1d_prefill_state_kernel(const __nv_bfloat16* x,
                                                   __nv_bfloat16* conv_state, std::int32_t C,
                                                   std::int32_t T) {
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64 = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 old0 = conv_state[c];
        const __nv_bfloat16 old1 = conv_state[C64 + c];
        const __nv_bfloat16 old2 = conv_state[2 * C64 + c];

        for (std::int32_t s = 0; s < 3; ++s) {
            const std::int32_t seq_pos = T + s;
            __nv_bfloat16 v;
            if (seq_pos == 0) {
                v = old0;
            } else if (seq_pos == 1) {
                v = old1;
            } else if (seq_pos == 2) {
                v = old2;
            } else {
                v = x[static_cast<std::int64_t>(seq_pos - 3) * C64 + c];
            }
            conv_state[static_cast<std::int64_t>(s) * C64 + c] = v;
        }
    }
}

__global__ void causal_conv1d_decode_kernel(const __nv_bfloat16* x,
                                            const __nv_bfloat16* weight,
                                            __nv_bfloat16* conv_state, __nv_bfloat16* out,
                                            std::int32_t C) {
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64 = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = conv_state[c];
        const __nv_bfloat16 s1 = conv_state[C64 + c];
        const __nv_bfloat16 s2 = conv_state[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        float acc = 0.0f;
        acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
        acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
        acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
        acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

        out[c] = __float2bfloat16_rn(causal_conv1d_silu_f32(acc));
        conv_state[c] = s1;
        conv_state[C64 + c] = s2;
        conv_state[2 * C64 + c] = x0;
    }
}

} // namespace qus::kernels
