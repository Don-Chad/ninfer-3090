#pragma once

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void linear_rowsplit_gemv_proj_6144_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              WorkspaceArena& ws, cudaStream_t stream);

} // namespace qus::kernels::detail
