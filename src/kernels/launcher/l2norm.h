#pragma once

// qus::kernels::detail - private launch prototype for l2norm.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void l2norm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
