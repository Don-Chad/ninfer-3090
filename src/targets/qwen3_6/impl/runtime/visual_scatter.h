#pragma once

// Identity-free Qwen3.6 family runtime helper.

#include "core/arena.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/mtp_alignment.h>

#include <cuda_runtime.h>

#include <cstddef>

#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
// source_column_base: item column index held by column 0 of visual_embeddings (nonzero when the
// caller staged only a window of the item's embeddings).
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const qwen3_6::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream,
                                       std::size_t source_column_base = 0);

} // namespace ninfer::targets::qwen3_6::detail
