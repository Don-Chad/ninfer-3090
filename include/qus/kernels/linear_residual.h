#pragma once

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void linear_residual_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                         WorkspaceArena& ws, cudaStream_t stream);

} // namespace qus::kernels
