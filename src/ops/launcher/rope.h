#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 DeviceExecutionView execution);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        DeviceExecutionView execution);

} // namespace ninfer::ops::detail
