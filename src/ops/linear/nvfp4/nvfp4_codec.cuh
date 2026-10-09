#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

__device__ __forceinline__ unsigned nvfp4_scaled_pair_bf16(std::uint8_t code, std::uint8_t scale) {
// The native widenings are Blackwell-only; sm_86 always takes the exact FP32 expansion.
#if (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) &&   \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // PTX 9.2 exposes native pair widening. E2M1 times a finite E4M3 scale needs
    // at most six significand bits, so the BF16 multiplication is exact.
    unsigned values, multiplier, result;
    asm("{\n"
        ".reg .b8 packed;\n"
        "mov.b32 {packed, _, _, _}, %1;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, packed;\n"
        "}\n"
        : "=r"(values)
        : "r"(static_cast<unsigned>(code)));
    const auto scale_pair = static_cast<std::uint16_t>(scale | (static_cast<unsigned>(scale) << 8));
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(multiplier) : "h"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(values), "r"(multiplier));
    return result;
#else
    // CUDA 13.1 retains the exact FP32 expansion.
    const float2 values    = decode_nvfp4_e2m1x2(code);
    const float multiplier = decode_nvfp4_e4m3(scale);

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    result.pair = __floats2bfloat162_rn(values.x * multiplier, values.y * multiplier);
    return result.bits;
#endif
}

__device__ __forceinline__ int nvfp4_a16_shared_col_64(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ std::int64_t nvfp4_scale_byte_offset(int row, int group, int k) {
    return static_cast<std::int64_t>((row / 128) * (k / 64) + group / 4) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4 + (group & 3);
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

// Encodes via the vendor __nv_fp4x2_e2m1(float2) constructor rather than the inline
// `cvt.rn.satfinite.e2m1x2.f32` PTX this replaced: that instruction is Blackwell-only, but the
// vendor type's constructor (cuda_fp4.hpp's __nv_cvt_float2_to_fp4x2) already carries a portable,
// bit-identical-to-hardware software fallback for __CUDA_ARCH__ < 1000, which sm_86/sm_89 take.
// Going through the same type decode already uses (confirmed working on sm_86 by
// nvfp4_small_t.cuh's existing decode-only usage) guarantees encode/decode agree on nibble order
// without re-deriving it by hand.
__device__ __forceinline__ std::uint8_t pack_e2m1x2(float2 pair) {
    __nv_fp4x2_e2m1 packed(pair);
    return packed.__x;
}

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    std::uint8_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { bytes[i] = pack_e2m1x2(values[i]); }
    codes_lo = static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
              (static_cast<std::uint32_t>(bytes[2]) << 16) |
              (static_cast<std::uint32_t>(bytes[3]) << 24);
    codes_hi = static_cast<std::uint32_t>(bytes[4]) | (static_cast<std::uint32_t>(bytes[5]) << 8) |
              (static_cast<std::uint32_t>(bytes[6]) << 16) |
              (static_cast<std::uint32_t>(bytes[7]) << 24);
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail
