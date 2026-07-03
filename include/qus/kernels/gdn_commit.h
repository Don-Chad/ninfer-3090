#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void gdn_commit(Tensor& conv_states, Tensor& ssm_states, const Tensor& accepted,
                cudaStream_t stream);

} // namespace qus::kernels
