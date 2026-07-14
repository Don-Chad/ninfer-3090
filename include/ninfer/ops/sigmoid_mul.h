#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   x'[i] = BF16(float(x[i]) * (1 / (1 + exp(-float(gate[i]))))).
 *
 * `gate` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. Transcendental and
 * multiply use FP32, followed by one BF16 round. The Op updates all of x in place and uses no
 * workspace or other persistent state.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
