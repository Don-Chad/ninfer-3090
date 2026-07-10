#pragma once

#include "qus/core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace qus::kernels {

// Set logits rows [valid_vocab, logits.ne[0]) to -infinity for every column.  The physical row
// stride remains logits.ne[0], so this is safe for multi-column target verification.
void mask_invalid_token_logits(Tensor& logits, std::int32_t valid_vocab, cudaStream_t stream);

} // namespace qus::kernels
