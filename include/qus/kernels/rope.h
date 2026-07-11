#pragma once

// qus::kernels - partial NeoX RoPE over q=[256,24,T] and k=[256,4,T], in place.
// Rotates the first rotary_dim channels and leaves remaining head dimensions untouched.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form. Q/K role is inferred from the head-count shape.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

} // namespace qus::kernels
