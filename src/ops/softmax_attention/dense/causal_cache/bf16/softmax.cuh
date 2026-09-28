#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include <math_constants.h>
#include <limits>

namespace ninfer::ops::detail {

inline constexpr float kBf16KvNegativeInfinity = -std::numeric_limits<float>::infinity();
inline constexpr float kBf16KvLog2E            = 1.4426950408889634074f;

// Keep the maximum in raw QK units in the mainloop, allowing scale to fold into
// the exponent's FMA. Exported partial maxima are always in log2 units.
struct Bf16KvSoftmaxRow {
    float maximum;
    float sum;
    float scaled_maximum;

    __device__ Bf16KvSoftmaxRow()
        : maximum(kBf16KvNegativeInfinity), sum(0.0f), scaled_maximum(0.0f) {}

    template <bool MayBeEmpty = true>
    __device__ __forceinline__ float update(float tile_maximum, float scale_log2) {
        const float previous = maximum;
        maximum              = fmaxf(maximum, tile_maximum);
        scaled_maximum       = maximum * scale_log2;
        if constexpr (MayBeEmpty)
            if (maximum == kBf16KvNegativeInfinity) scaled_maximum = 0.0f;
        const float alpha = exp2_approx(__fmaf_rn(previous, scale_log2, -scaled_maximum));
        return alpha;
    }

    __device__ __forceinline__ void accumulate(float alpha, float tile_sum) {
        sum = __fmaf_rn(sum, alpha, tile_sum);
    }

    __device__ __forceinline__ float probability(float score, float scale_log2) const {
        return exp2_approx(__fmaf_rn(score, scale_log2, -scaled_maximum));
    }

    __device__ __forceinline__ void finish() { sum = warp_sum<4>(sum, 0xffffffffu); }
};

__device__ __forceinline__ float bf16_kv_state_weight(float maximum, float sum, float combined) {
    return sum > 0.0f ? exp2_approx(maximum - combined) : 0.0f;
}

} // namespace ninfer::ops::detail
