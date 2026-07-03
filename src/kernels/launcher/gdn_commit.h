#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace qus::kernels::detail {

void gdn_commit_launch(Tensor& conv_states, Tensor& ssm_states, const Tensor& accepted,
                       cudaStream_t stream);

} // namespace qus::kernels::detail
