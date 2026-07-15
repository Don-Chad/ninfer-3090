#pragma once

#include "core/kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns transient arena capacity for the fused attention route. It is nonzero for the
 * registered small-T (T=1..6) split-KV path and zero for prompt attention.
 */
[[nodiscard]] std::size_t gqa_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens);

/**
 * Appends K/V at the supplied absolute positions and computes causal grouped-query attention.
 * For query head h, kvh=floor(h/group), p=positions[t], and a populated cache history [0,p]:
 *
 *   score[j]      = scale * dot(q[:,h,t], K_cache[:,j,kvh]), 0 <= j <= p
 *   probability   = softmax_j(score)
 *   out[:,h,t]    = BF16(sum_{j=0..p} probability[j] * V_cache[:,j,kvh]).
 *
 * The registered q/k/v head geometries are `[256,24|4,T]` with group 6 and `[256,16|2,T]` with
 * group 8; out matches q. All q/k/v/out tensors are contiguous BF16, positions is contiguous I32
 * [T], and `scale` is 1/sqrt(256). Positions are valid cache slots in causal order and all earlier
 * slots read by attention are populated. For Hkv=4 or 2, the selected cache layer is either BF16
 * `[256,padded_context,Hkv]` or group-64 I8 codes with FP16 scales `[4,padded_context,Hkv]`. The
 * logical formula uses the values represented by that cache format.
 *
 * q/k/v/positions/out and cache storage must not overlap. The Op writes K/V (and I8 scales) at
 * positions but does not advance the host-side `kv.pos`; the caller owns that state transition.
 * `ws` provides transient capacity reported by gqa_attention_workspace_bytes(q.ne[1], T) and is
 * scoped to the call.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                   cudaStream_t stream);

/**
 * Performs only the cache-write part of gqa_attention. k/v use either registered contiguous BF16
 * shape `[256,4,T]` or `[256,2,T]`, and positions is contiguous I32 [T]; cache format, position,
 * alias, and `kv.pos` semantics are the same as above. Every addressed K/V code (and I8 scale) is
 * overwritten. No workspace is used.
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, KVCache& kv,
                   int layer, cudaStream_t stream);

/**
 * Performs only causal attention over an already populated cache, using the formula above.
 * q/out use either registered contiguous BF16 shape `[256,24,T]` or `[256,16,T]`, positions is
 * contiguous I32 [T], and scale is 1/sqrt(256). It writes all of out, does not mutate the cache or
 * `kv.pos`, and uses no caller workspace. Inputs, output, and cache storage must not overlap.
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale, KVCache& kv,
                          int layer, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
