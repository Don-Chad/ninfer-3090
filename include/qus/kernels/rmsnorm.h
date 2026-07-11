#pragma once

// qus::kernels - rmsnorm over the fastest dimension ne[0].
// unit_offset=true computes x/rms(x) * (1 + weight); false uses plain weight.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream);

} // namespace qus::kernels
