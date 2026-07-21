#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <std::int32_t Rows, std::int32_t RowsPerCta, class Output>
__global__ __launch_bounds__(RowsPerCta * 32,
                             2) void w8_k2048_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                                            const std::uint8_t* __restrict__ codes,
                                                            const std::uint8_t* __restrict__ scales,
                                                            Output output) {
    static_assert(Rows > 0 && RowsPerCta > 0 && (Rows % RowsPerCta) == 0);
    static_assert(RowsPerCta * 32 <= 1024);
    constexpr int kK                 = 2048;
    constexpr int kGroup             = 32;
    constexpr int kGroupsPerRow      = kK / kGroup;
    constexpr int kValuesPerLane     = 16;
    constexpr int kValuesPerPhase    = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase    = kValuesPerPhase / kGroup;
    constexpr int kPhases            = kK / kValuesPerPhase;
    constexpr unsigned kFullWarpMask = 0xffffffffu;

    const int lane                 = static_cast<int>(threadIdx.x) & 31;
    const int warp                 = static_cast<int>(threadIdx.x) >> 5;
    const int cta_row0             = static_cast<int>(blockIdx.x) * RowsPerCta;
    const int row                  = cta_row0 + warp;
    const W8OutputTile output_tile = output.tile(cta_row0);

    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kK;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;

    float accumulator = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kGroupsPerPhase) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        scale_bits        = __shfl_sync(kFullWarpMask, scale_bits, lane >> 1);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint4 packed = load_vec<uint4>(code_row + phase_k);
        float weights[kValuesPerLane];
#pragma unroll
        for (int word_index = 0; word_index < 4; ++word_index) {
            const std::uint32_t word = (&packed.x)[word_index];
            weights[word_index * 4 + 0] =
                static_cast<float>(static_cast<std::int8_t>(word & 0xffu)) * scale;
            weights[word_index * 4 + 1] =
                static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xffu)) * scale;
            weights[word_index * 4 + 2] =
                static_cast<float>(static_cast<std::int8_t>((word >> 16) & 0xffu)) * scale;
            weights[word_index * 4 + 3] =
                static_cast<float>(static_cast<std::int8_t>((word >> 24) & 0xffu)) * scale;
        }

        const uint4 values0 = load_vec<uint4>(x + phase_k);
        const uint4 values1 = load_vec<uint4>(x + phase_k + 8);
        const float2 x0     = bf16x2_bits_to_float2(values0.x);
        const float2 x1     = bf16x2_bits_to_float2(values0.y);
        const float2 x2     = bf16x2_bits_to_float2(values0.z);
        const float2 x3     = bf16x2_bits_to_float2(values0.w);
        const float2 x4     = bf16x2_bits_to_float2(values1.x);
        const float2 x5     = bf16x2_bits_to_float2(values1.y);
        const float2 x6     = bf16x2_bits_to_float2(values1.z);
        const float2 x7     = bf16x2_bits_to_float2(values1.w);
        accumulator         = fmaf(weights[0], x0.x, accumulator);
        accumulator         = fmaf(weights[1], x0.y, accumulator);
        accumulator         = fmaf(weights[2], x1.x, accumulator);
        accumulator         = fmaf(weights[3], x1.y, accumulator);
        accumulator         = fmaf(weights[4], x2.x, accumulator);
        accumulator         = fmaf(weights[5], x2.y, accumulator);
        accumulator         = fmaf(weights[6], x3.x, accumulator);
        accumulator         = fmaf(weights[7], x3.y, accumulator);
        accumulator         = fmaf(weights[8], x4.x, accumulator);
        accumulator         = fmaf(weights[9], x4.y, accumulator);
        accumulator         = fmaf(weights[10], x5.x, accumulator);
        accumulator         = fmaf(weights[11], x5.y, accumulator);
        accumulator         = fmaf(weights[12], x6.x, accumulator);
        accumulator         = fmaf(weights[13], x6.y, accumulator);
        accumulator         = fmaf(weights[14], x7.x, accumulator);
        accumulator         = fmaf(weights[15], x7.y, accumulator);
    }

    accumulator = warp_reduce_sum(accumulator);
    if (lane == 0) { *output_tile.at(row, 0) = __float2bfloat16_rn(accumulator); }
}

} // namespace ninfer::ops::detail
