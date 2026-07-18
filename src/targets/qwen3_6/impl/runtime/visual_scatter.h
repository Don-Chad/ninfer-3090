#pragma once

// Identity-free Qwen3.6 family runtime helper.

#include "core/arena.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/mtp_alignment.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       std::span<const std::int32_t> scatter_indices,
                                       std::uint32_t prompt_tokens,
                                       const qwen3_6::MtpAlignmentWindow& window,
                                       WorkspaceArena& work, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6::detail
