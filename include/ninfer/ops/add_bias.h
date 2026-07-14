#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Adds one bias value to every element whose fastest-dimension index is the same:
 *
 *   x'[d,r] = BF16(float(x[d,r]) + float(bias[d])),  0 <= d < D.
 *
 * Here r flattens all dimensions above ne[0]. `bias` is contiguous BF16 [D] and `x` is a
 * contiguous BF16 tensor with ne[0]=D. The BF16 result is rounded once after the FP32 add.
 * `x` is updated in place; `bias` must not overlap `x`. There is no workspace or other state
 * side effect.
 */
void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
