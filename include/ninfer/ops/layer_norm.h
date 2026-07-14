#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies affine LayerNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   mean_r    = (1/D) * sum_d float(x[d,r])
 *   variance  = (1/D) * sum_d (float(x[d,r]) - mean_r)^2
 *   out[d,r]  = BF16((float(x[d,r])-mean_r)/sqrt(variance+eps) * weight[d] + bias[d]).
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors; weight and bias are contiguous BF16 [D];
 * eps is positive and finite. All inputs and output must be non-overlapping. Moments and affine
 * math use FP32, with one BF16 output round. There is no workspace or persistent state side
 * effect.
 */
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
                cudaStream_t stream);

} // namespace ninfer::ops
