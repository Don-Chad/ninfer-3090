#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_down.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN = 5120;
constexpr int kK = 17408;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kBytesPerGroup = 40;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockThreads = kWarpsPerBlock * 32;

__device__ __forceinline__ int sign_extend_q5(int v) {
    return (v & 0x10) ? (v - 32) : v;
}

__device__ __forceinline__ float accumulate_group(const __nv_bfloat162* __restrict__ x2,
                                                  std::uint32_t lane_word,
                                                  std::uint16_t scale_bits,
                                                  int source_lane_base, int lane, int group,
                                                  float acc) {
    const float scale = __half2float(__ushort_as_half(scale_bits));

    const int bitpos = lane * 10;
    const int wi = bitpos >> 5;
    const int sh = bitpos & 31;
    const std::uint32_t w0 = __shfl_sync(0xffffffffu, lane_word, source_lane_base + wi);
    const std::uint32_t w1_next =
        __shfl_sync(0xffffffffu, lane_word, source_lane_base + (wi < 9 ? wi + 1 : 9));
    const std::uint32_t w1 = wi < 9 ? w1_next : 0u;
    const std::uint32_t bits = __funnelshift_r(w0, w1, sh);

    const int q0 = sign_extend_q5(static_cast<int>(bits & 0x1fu));
    const int q1 = sign_extend_q5(static_cast<int>((bits >> 5) & 0x1fu));
    const int k0 = group * kGroupK + lane * 2;
    const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
    acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
    acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    return acc;
}

__global__ void linear_rowsplit_gemv_mlp_down_q5_kernel(const __nv_bfloat16* __restrict__ x,
                                                        const std::uint8_t* __restrict__ codes,
                                                        const std::uint8_t* __restrict__ scales,
                                                        __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.0f;
    int group = 0;
#pragma unroll 1
    for (; group + 2 < kGroups; group += 3) {
        const std::uint8_t* code_group0 = code_row + group * kBytesPerGroup;
        const std::uint8_t* code_group1 = code_group0 + kBytesPerGroup;
        const std::uint8_t* code_group2 = code_group1 + kBytesPerGroup;
        std::uint32_t lane_word = 0;
        if (lane < 10) {
            lane_word = *reinterpret_cast<const std::uint32_t*>(code_group0 + lane * 4);
        } else if (lane < 20) {
            lane_word = *reinterpret_cast<const std::uint32_t*>(code_group1 + (lane - 10) * 4);
        } else if (lane < 30) {
            lane_word = *reinterpret_cast<const std::uint32_t*>(code_group2 + (lane - 20) * 4);
        }

        std::uint16_t scale0_bits = 0;
        std::uint16_t scale1_bits = 0;
        std::uint16_t scale2_bits = 0;
        if (lane == 0) {
            const std::uint8_t* sp = scale_row + group * 2;
            scale0_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        } else if (lane == 1) {
            const std::uint8_t* sp = scale_row + (group + 1) * 2;
            scale1_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        } else if (lane == 2) {
            const std::uint8_t* sp = scale_row + (group + 2) * 2;
            scale2_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        }
        scale0_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale0_bits, 0));
        scale1_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale1_bits, 1));
        scale2_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale2_bits, 2));

        acc = accumulate_group(x2, lane_word, scale0_bits, 0, lane, group, acc);
        acc = accumulate_group(x2, lane_word, scale1_bits, 10, lane, group + 1, acc);
        acc = accumulate_group(x2, lane_word, scale2_bits, 20, lane, group + 2, acc);
    }

    if (group < kGroups) {
        const std::uint8_t* code_group0 = code_row + group * kBytesPerGroup;
        const std::uint8_t* code_group1 = code_group0 + kBytesPerGroup;
        std::uint32_t lane_word = 0;
        if (lane < 10) {
            lane_word = *reinterpret_cast<const std::uint32_t*>(code_group0 + lane * 4);
        } else if (lane < 20) {
            lane_word = *reinterpret_cast<const std::uint32_t*>(code_group1 + (lane - 10) * 4);
        }

        std::uint16_t scale0_bits = 0;
        std::uint16_t scale1_bits = 0;
        if (lane == 0) {
            const std::uint8_t* sp = scale_row + group * 2;
            scale0_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        } else if (lane == 1) {
            const std::uint8_t* sp = scale_row + (group + 1) * 2;
            scale1_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        }
        scale0_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale0_bits, 0));
        scale1_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale1_bits, 1));

        acc = accumulate_group(x2, lane_word, scale0_bits, 0, lane, group, acc);
        acc = accumulate_group(x2, lane_word, scale1_bits, 10, lane, group + 1, acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

} // namespace

void linear_rowsplit_gemv_mlp_down_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: MLP down Q5 tuned GEMV requires 5120x17408");
    }
    const int grid = (kN + kWarpsPerBlock - 1) / kWarpsPerBlock;
    linear_rowsplit_gemv_mlp_down_q5_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
