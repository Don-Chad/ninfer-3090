#pragma once

// Exact E4M3-to-BF16 operand widening shared by the small-T and GEMM A16 routes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ unsigned fp8_e4m3x2_to_bf16x2_bits(unsigned packed) {
// The native widening is Blackwell-only; sm_86 always takes the exact FP16 bridge.
#if (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) &&   \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // PTX 9.2 exposes the native pair widening on the project's sm_120a target.
    unsigned result;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(static_cast<std::uint16_t>(packed)));
    return result;
#else
    // CUDA 13.1 retains the exact FP16 bridge.
    __nv_fp8x2_e4m3 fp8;
    fp8.__x                 = static_cast<std::uint16_t>(packed);
    const __half2 half_pair = static_cast<__half2>(fp8);

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    result.pair = __halves2bfloat162(__nv_bfloat16(__low2half(half_pair)),
                                     __nv_bfloat16(__high2half(half_pair)));
    return result.bits;
#endif
}

__device__ __forceinline__ int fp8_a16_shared_col_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

} // namespace ninfer::ops::detail
