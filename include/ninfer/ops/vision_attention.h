#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Returns the descriptor upper bound
 *
 *   M(P,S) = S==1 ? 0 : ceil(P/64) + S - 1
 *
 * for positive patch and segment counts. Each descriptor occupies four contiguous I32 elements.
 * The query allocates no workspace and has no side effects.
 */
[[nodiscard]] std::int32_t vision_attention_scratch_tiles(std::int32_t patches,
                                                          std::int32_t segments);

/**
 * Packed, non-causal multi-head attention. For each segment [begin,end), head h, and
 * t in that segment:
 *
 *   score[j]    = dot(q[:,h,t], k[:,h,j]) / sqrt(72), begin <= j < end
 *   out[:,h,t]  = BF16(sum_j softmax(score)[j] * v[:,h,j]).
 *
 * q/k/v are BF16 [72,16,P] with contiguous feature and head dimensions; token strides may be
 * padded. out is contiguous BF16 [72,16,P]. cu_seqlens is contiguous I32 [S+1], begins at 0,
 * ends at P, and is strictly increasing. Inputs and output are mutually non-overlapping.
 *
 * For S=1, scratch_tiles is null (or has null data). Otherwise it points to distinct contiguous
 * I32 [4,M] storage with M>=vision_attention_scratch_tiles(P,S); the descriptors are opaque and
 * overwritten by the Op. There is no persistent state side effect.
 */
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      Tensor* scratch_tiles, Tensor& out, cudaStream_t stream);

/**
 * Arena overload with the same mathematical, shape, layout, and alias contract. It allocates the
 * required opaque tile descriptors from `workspace` for the duration of the call; no capacity is
 * consumed for a single segment.
 */
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Equal-segment overload of the same packed, non-causal attention operation. P must be divisible
 * by `segment_length`; consecutive ranges `[s*segment_length,(s+1)*segment_length)` are the
 * independent segments. The q/k/v/out shape, layout, dtype, alias, and numerical semantics are
 * identical to the cu_seqlens overload. Segment tiles are derived directly and no workspace or
 * descriptor-setup launch is used.
 */
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      std::int32_t segment_length, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
