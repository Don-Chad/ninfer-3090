#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);

} // namespace ninfer::ops::detail
