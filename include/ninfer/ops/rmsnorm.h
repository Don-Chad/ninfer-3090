#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Applies RMSNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d float(x[d,r])^2 + eps)
 *   gain[d]  = unit_offset ? 1 + float(weight[d]) : float(weight[d])
 *   out[d,r] = BF16(float(x[d,r]) * inv_r * gain[d]).
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors, weight is contiguous BF16 [D], and eps
 * is positive and finite. Input, weight, and output must not overlap. Reduction and affine math
 * use FP32 with one BF16 output round. There is no workspace or persistent state side effect.
 */
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream);

} // namespace ninfer::ops
