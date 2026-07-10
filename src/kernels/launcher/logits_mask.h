#pragma once

#include "qus/core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void mask_invalid_token_logits_launch(Tensor& logits, std::int32_t valid_vocab,
                                      cudaStream_t stream);

} // namespace qus::kernels::detail
