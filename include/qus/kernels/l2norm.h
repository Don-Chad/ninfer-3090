#pragma once

// qus::kernels - l2norm over the fastest dimension ne[0].
// out_i = x_i / sqrt(sum(row x^2) + eps), computed per row.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels
