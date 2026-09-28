#pragma once

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/schedule.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/operands.h"

namespace ninfer::ops::detail {

template <class G>
__device__ __forceinline__ std::int64_t bf16_kv_cache_index(int page, int head, int d, int offset) {
    return paged_kv_element_offset<G::kHeadDim, G::KVHeads>(page, head, offset, d);
}

template <class G>
__device__ __forceinline__ std::int64_t bf16_kv_q_index(int head, int d, int token = 0) {
    return d + static_cast<std::int64_t>(G::kHeadDim) *
                   (head + static_cast<std::int64_t>(G::QHeads) * token);
}

template <class G>
__device__ __forceinline__ std::int64_t bf16_kv_new_index(int head, int d, int token = 0) {
    return d + static_cast<std::int64_t>(G::kHeadDim) *
                   (head + static_cast<std::int64_t>(G::KVHeads) * token);
}

template <class G>
__device__ __forceinline__ std::int64_t bf16_kv_partial_index(int head, int d, int token, int split,
                                                              int tokens) {
    return d + static_cast<std::int64_t>(G::kHeadDim) *
                   (head + static_cast<std::int64_t>(G::QHeads) *
                               (token + static_cast<std::int64_t>(tokens) * split));
}

template <class G>
__device__ __forceinline__ std::int64_t bf16_kv_stat_index(int head, int token, int split,
                                                           int tokens) {
    return head + static_cast<std::int64_t>(G::QHeads) *
                      (token + static_cast<std::int64_t>(tokens) * split);
}

__device__ __forceinline__ int bf16_kv_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned bf16_kv_swizzle_address(unsigned base, unsigned column,
                                                            unsigned matrix, unsigned row) {
    return base + ((column | matrix) ^ row);
}

template <class G>
__device__ __forceinline__ void bf16_kv_zero_rows(__nv_bfloat16* out, int head, int begin, int end,
                                                  int tid, int threads) {
    for (int i = tid; i < (end - begin) * G::kHeadDim; i += threads)
        out[bf16_kv_q_index<G>(head, i % G::kHeadDim, begin + i / G::kHeadDim)] =
            __float2bfloat16(0.0f);
}

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous cache into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always
// drops) are zeroed so the padded/uninitialized cache tail never feeds NaNs into
// the tensor cores. Mirrors FA's predicated K/V cp.async + Clear_OOB path.
template <typename Geometry, class Schedule, typename Element>
__device__ __forceinline__ void bf16_kv_stage_tile(Element* dst, const Element* cache, int kv_head,
                                                   int k0, int max_query_abs, int physical_page,
                                                   int tid) {
    constexpr int D         = Geometry::kHeadDim;
    constexpr int Bc        = Schedule::kKeyRows;
    constexpr int Threads   = Schedule::kThreads;
    constexpr int VecPerRow = D / 8; // 8 two-byte elements per 16B cp.async
    const bool full_tile    = (k0 + Bc - 1) <= max_query_abs;
    // Block base pointer computed once (int64); per-element offsets stay 32-bit.
    const Element* cache_block =
        cache + paged_kv_element_offset<Geometry::kHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l = chunk / VecPerRow;       // / VecPerRow
            const int d     = (chunk % VecPerRow) * 8; // feature offset
            Element* p      = &dst[key_l * D + bf16_kv_swizzle(key_l, d)];
            cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l = chunk / VecPerRow;       // / VecPerRow
            const int d     = (chunk % VecPerRow) * 8; // feature offset
            Element* p      = &dst[key_l * D + bf16_kv_swizzle(key_l, d)];
            if ((k0 + key_l) <= max_query_abs) {
                cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
}

} // namespace ninfer::ops::detail
