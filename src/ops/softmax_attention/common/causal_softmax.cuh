#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include <math_constants.h>

namespace ninfer::ops::detail {


// Grouped scores already include attention_scale; tiled scores fold it into exp2.
__device__ __forceinline__ float causal_exp_difference(float score, float maximum, float scale) {
    return exp2_approx((score - maximum) * scale);
}

__device__ __forceinline__ float causal_exp_scaled(float score, float scaled_maximum, float scale) {
    return exp2_approx(__fmaf_rn(score, scale, -scaled_maximum));
}

} // namespace ninfer::ops::detail
