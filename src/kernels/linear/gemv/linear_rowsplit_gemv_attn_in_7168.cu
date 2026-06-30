#include "kernels/linear/gemv/linear_rowsplit_gemv_attn_in_7168.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN = 7168;
constexpr int kK = 5120;
constexpr int kProjRows = 6144;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kQ4WarpsPerRow = 4;
constexpr int kQ4GroupsPerWarp = kGroups / kQ4WarpsPerRow;
constexpr int kQ4BlockThreads = kQ4WarpsPerRow * 32;
constexpr int kQ4ProjBlocks = kProjRows / kQ4WarpsPerRow;
constexpr int kQ5WarpsPerRow = 8;
constexpr int kQ5GroupsPerWarp = kGroups / kQ5WarpsPerRow;
constexpr int kQ5BlockThreads = kQ5WarpsPerRow * 32;
constexpr int kQ5ProjBlocks = kProjRows / kQ5WarpsPerRow;

constexpr int kQ4BytesPerGroup = 32;
constexpr int kQ5NibbleBytesPerGroup = 32;
constexpr int kQ5HighBytesPerGroup = 8;

__device__ __forceinline__ int sign_extend_q4(int v) {
    return (v & 0x08) ? (v - 16) : v;
}

__device__ __forceinline__ int sign_extend_q5(int v) {
    return (v & 0x10) ? (v - 32) : v;
}

__device__ __forceinline__ std::uint16_t load_scale_bits(const std::uint8_t* scale_row,
                                                         int group) {
    const std::uint8_t* sp = scale_row + group * 2;
    return static_cast<std::uint16_t>(sp[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
}

__device__ __forceinline__ std::uint32_t load_scale_pair_bits(const std::uint8_t* scale_row,
                                                              int group) {
    return static_cast<std::uint32_t>(load_scale_bits(scale_row, group)) |
           (static_cast<std::uint32_t>(load_scale_bits(scale_row, group + 1)) << 16);
}

__device__ __forceinline__ float accumulate_q5_group(const __nv_bfloat162* __restrict__ x2,
                                                     const std::uint8_t* __restrict__ nibble_group,
                                                     const std::uint8_t* __restrict__ high_group,
                                                     std::uint16_t scale_bits,
                                                     int lane, int group, float acc) {
    const float scale = __half2float(__ushort_as_half(scale_bits));

    const std::uint8_t low = nibble_group[lane];
    const std::uint8_t high = high_group[lane >> 2] >> ((lane & 3) * 2);

    const int q0 = sign_extend_q5(static_cast<int>((low & 0x0fu) | ((high & 0x01u) << 4)));
    const int q1 = sign_extend_q5(static_cast<int>((low >> 4) | ((high & 0x02u) << 3)));
    const int k0 = group * kGroupK + lane * 2;
    const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
    acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
    acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    return acc;
}

__device__ __forceinline__ float warp_reduce_sum(float acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__global__ void linear_rowsplit_gemv_attn_in_7168_q4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    if (static_cast<int>(blockIdx.x) < kQ4ProjBlocks) {
        const int row = static_cast<int>(blockIdx.x) * kQ4WarpsPerRow + warp;
        const std::uint8_t* code_row =
            codes + static_cast<std::int64_t>(row) * kGroups * kQ4BytesPerGroup;
        const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;

        float acc = 0.0f;
#pragma unroll 1
        for (int tile = 0; tile < kGroups; tile += 32) {
            const int tile_count = (kGroups - tile) < 32 ? (kGroups - tile) : 32;
            std::uint16_t lane_scale_bits = 0;
            if (lane < tile_count) { lane_scale_bits = load_scale_bits(scale_row, tile + lane); }

#pragma unroll
            for (int tile_group = 0; tile_group < 32; ++tile_group) {
                if (tile_group >= tile_count) { break; }
                const int group = tile + tile_group;
                const auto scale_bits = static_cast<std::uint16_t>(
                    __shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
                const float scale = __half2float(__ushort_as_half(scale_bits));

                const std::uint8_t packed = code_row[group * kQ4BytesPerGroup + lane];
                const int q0 = sign_extend_q4(packed & 0x0f);
                const int q1 = sign_extend_q4(packed >> 4);
                const int k0 = group * kGroupK + lane * 2;
                const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
                acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
                acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
            }
        }

        acc = warp_reduce_sum(acc);
        if (lane == 0) { out[row] = __float2bfloat16(acc); }
        return;
    }

    const int row = kProjRows + (static_cast<int>(blockIdx.x) - kQ4ProjBlocks);
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kQ4BytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    __shared__ float partials[kQ4WarpsPerRow];
    const int group_begin = warp * kQ4GroupsPerWarp;
    float acc = 0.0f;

    std::uint16_t lane_scale_bits = 0;
    if (lane < kQ4GroupsPerWarp) {
        lane_scale_bits = load_scale_bits(scale_row, group_begin + lane);
    }

#pragma unroll
    for (int tile_group = 0; tile_group < kQ4GroupsPerWarp; ++tile_group) {
        const int group = group_begin + tile_group;
        const auto scale_bits =
            static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const std::uint8_t packed = code_row[group * kQ4BytesPerGroup + lane];
        const int q0 = sign_extend_q4(packed & 0x0f);
        const int q1 = sign_extend_q4(packed >> 4);
        const int k0 = group * kGroupK + lane * 2;
        const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
        acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
        acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { partials[warp] = acc; }
    __syncthreads();

    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(partials[0] + partials[1] + partials[2] + partials[3]);
    }
}

__global__ void linear_rowsplit_gemv_attn_in_7168_q5_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high_bits,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    if (static_cast<int>(blockIdx.x) < kQ5ProjBlocks) {
        const int row = static_cast<int>(blockIdx.x) * kQ5WarpsPerRow + warp;
        const std::uint8_t* code_row =
            codes + static_cast<std::int64_t>(row) * kGroups * kQ5NibbleBytesPerGroup;
        const std::uint8_t* high_row =
            high_bits + static_cast<std::int64_t>(row) * kGroups * kQ5HighBytesPerGroup;
        const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;

        float acc = 0.0f;
#pragma unroll 1
        for (int group = 0; group < kGroups; group += 4) {
            const std::uint8_t* code_group0 = code_row + group * kQ5NibbleBytesPerGroup;
            const std::uint8_t* code_group1 = code_group0 + kQ5NibbleBytesPerGroup;
            const std::uint8_t* code_group2 = code_group1 + kQ5NibbleBytesPerGroup;
            const std::uint8_t* code_group3 = code_group2 + kQ5NibbleBytesPerGroup;
            const std::uint8_t* high_group0 = high_row + group * kQ5HighBytesPerGroup;
            const std::uint8_t* high_group1 = high_group0 + kQ5HighBytesPerGroup;
            const std::uint8_t* high_group2 = high_group1 + kQ5HighBytesPerGroup;
            const std::uint8_t* high_group3 = high_group2 + kQ5HighBytesPerGroup;

            std::uint32_t scale01_bits = 0;
            std::uint32_t scale23_bits = 0;
            if (lane == 0) {
                scale01_bits = load_scale_pair_bits(scale_row, group);
            } else if (lane == 1) {
                scale23_bits = load_scale_pair_bits(scale_row, group + 2);
            }
            scale01_bits = static_cast<std::uint32_t>(__shfl_sync(0xffffffffu, scale01_bits, 0));
            scale23_bits = static_cast<std::uint32_t>(__shfl_sync(0xffffffffu, scale23_bits, 1));
            const auto scale0_bits = static_cast<std::uint16_t>(scale01_bits & 0xffffu);
            const auto scale1_bits = static_cast<std::uint16_t>(scale01_bits >> 16);
            const auto scale2_bits = static_cast<std::uint16_t>(scale23_bits & 0xffffu);
            const auto scale3_bits = static_cast<std::uint16_t>(scale23_bits >> 16);

            acc = accumulate_q5_group(x2, code_group0, high_group0, scale0_bits, lane, group, acc);
            acc = accumulate_q5_group(x2, code_group1, high_group1, scale1_bits, lane, group + 1,
                                      acc);
            acc = accumulate_q5_group(x2, code_group2, high_group2, scale2_bits, lane, group + 2,
                                      acc);
            acc = accumulate_q5_group(x2, code_group3, high_group3, scale3_bits, lane, group + 3,
                                      acc);
        }

        acc = warp_reduce_sum(acc);
        if (lane == 0) { out[row] = __float2bfloat16(acc); }
        return;
    }

    const int row = kProjRows + (static_cast<int>(blockIdx.x) - kQ5ProjBlocks);
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kQ5NibbleBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * kQ5HighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    __shared__ float partials[kQ5WarpsPerRow];
    const int group_begin = warp * kQ5GroupsPerWarp;
    const int group_end = group_begin + kQ5GroupsPerWarp;
    float acc = 0.0f;

    int group = group_begin;
#pragma unroll 1
    for (; group + 2 < group_end; group += 3) {
        const std::uint8_t* code_group0 = code_row + group * kQ5NibbleBytesPerGroup;
        const std::uint8_t* code_group1 = code_group0 + kQ5NibbleBytesPerGroup;
        const std::uint8_t* code_group2 = code_group1 + kQ5NibbleBytesPerGroup;
        const std::uint8_t* high_group0 = high_row + group * kQ5HighBytesPerGroup;
        const std::uint8_t* high_group1 = high_group0 + kQ5HighBytesPerGroup;
        const std::uint8_t* high_group2 = high_group1 + kQ5HighBytesPerGroup;

        std::uint32_t scale01_bits = 0;
        std::uint16_t scale2_bits = 0;
        if (lane == 0) {
            scale01_bits = load_scale_pair_bits(scale_row, group);
        } else if (lane == 1) {
            scale2_bits = load_scale_bits(scale_row, group + 2);
        }
        scale01_bits = static_cast<std::uint32_t>(__shfl_sync(0xffffffffu, scale01_bits, 0));
        scale2_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale2_bits, 1));
        const auto scale0_bits = static_cast<std::uint16_t>(scale01_bits & 0xffffu);
        const auto scale1_bits = static_cast<std::uint16_t>(scale01_bits >> 16);

        acc = accumulate_q5_group(x2, code_group0, high_group0, scale0_bits, lane, group, acc);
        acc = accumulate_q5_group(x2, code_group1, high_group1, scale1_bits, lane, group + 1, acc);
        acc = accumulate_q5_group(x2, code_group2, high_group2, scale2_bits, lane, group + 2, acc);
    }

    if (group < group_end) {
        const std::uint8_t* code_group0 = code_row + group * kQ5NibbleBytesPerGroup;
        const std::uint8_t* code_group1 = code_group0 + kQ5NibbleBytesPerGroup;
        const std::uint8_t* high_group0 = high_row + group * kQ5HighBytesPerGroup;
        const std::uint8_t* high_group1 = high_group0 + kQ5HighBytesPerGroup;

        std::uint32_t scale01_bits = 0;
        if (lane == 0) {
            scale01_bits = static_cast<std::uint32_t>(load_scale_bits(scale_row, group));
            if (group + 1 < group_end) {
                scale01_bits |= static_cast<std::uint32_t>(load_scale_bits(scale_row, group + 1))
                                << 16;
            }
        }
        scale01_bits = static_cast<std::uint32_t>(__shfl_sync(0xffffffffu, scale01_bits, 0));
        const auto scale0_bits = static_cast<std::uint16_t>(scale01_bits & 0xffffu);
        const auto scale1_bits = static_cast<std::uint16_t>(scale01_bits >> 16);

        acc = accumulate_q5_group(x2, code_group0, high_group0, scale0_bits, lane, group, acc);
        if (group + 1 < group_end) {
            acc = accumulate_q5_group(x2, code_group1, high_group1, scale1_bits, lane, group + 1,
                                      acc);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { partials[warp] = acc; }
    __syncthreads();

    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(partials[0] + partials[1] + partials[2] + partials[3] +
                                    partials[4] + partials[5] + partials[6] + partials[7]);
    }
}

} // namespace

void linear_rowsplit_gemv_attn_in_7168_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: Attention q/k Q4 fused GEMV requires 7168x5120");
    }
    constexpr int kBlocks = kQ4ProjBlocks + (kN - kProjRows);
    linear_rowsplit_gemv_attn_in_7168_q4_kernel<<<kBlocks, kQ4BlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void linear_rowsplit_gemv_attn_in_7168_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: Attention gate/v Q5 fused GEMV requires 7168x5120");
    }
    constexpr int kBlocks = kQ5ProjBlocks + (kN - kProjRows);
    linear_rowsplit_gemv_attn_in_7168_q5_kernel<<<kBlocks, kQ5BlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
