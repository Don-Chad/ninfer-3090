#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule::detail {

// Selects only visual columns covered by the shifted MTP input window, then composes the generic
// scatter Op. Shift and chunk-boundary interpretation belong to this exact checkpoint schedule.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       std::span<const std::int32_t> scatter_indices,
                                       std::int32_t shifted_begin, std::int32_t prompt_tokens,
                                       WorkspaceArena& work, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule::detail
