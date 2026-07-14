#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: finite typed scalar state transitions
 *
 * Math / indexing:
 *   set_i32_scalar:       destination' = value
 *   assign_i32_scalar:    destination' = source
 *   increment_i32_scalar: scalar' = scalar + 1
 *   increment_i64_scalar: scalar' = scalar + 1
 *
 * Logical shapes:
 *   Every tensor argument is one contiguous scalar element of the named dtype.
 *
 * Numeric:
 *   Increment callers keep the old value below the corresponding signed integer maximum.
 *
 * Effects:
 *   Each call writes only its destination scalar. assign_i32_scalar requires distinct source and
 *   destination storage; the other calls update their single destination in place.
 *
 * Workspace:
 *   None. There is no state side effect beyond the stated destination transition.
 */
void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream);
void assign_i32_scalar(const Tensor& source, Tensor& destination, cudaStream_t stream);
void increment_i32_scalar(Tensor& scalar, cudaStream_t stream);
void increment_i64_scalar(Tensor& scalar, cudaStream_t stream);

} // namespace ninfer::ops
