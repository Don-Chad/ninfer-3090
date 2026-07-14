#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise residual update:
 *
 *   x'[i] = BF16(float(x[i]) + float(y[i])).
 *
 * `y` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. The Op updates all of x
 * in place, leaves y unchanged, and uses no workspace or other persistent state.
 */
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
