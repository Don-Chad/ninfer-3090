#pragma once

// Shared implementation primitives for include/ninfer/ops/sampling.h and
// include/ninfer/ops/speculative_round.h. The ordering key is exact for finite BF16
// logits (including numeric-zero ties); candidate storage is bounded by the
// semantic top-20 cap and all global staging is supplied by the caller.

#include "ops/common/math.h"
#include "ops/common/sampling_workspace.h"
#include "ops/common/score_id_order.cuh"
#include "ninfer/ops/sampling.h"

#include <cub/block/block_merge_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

using SamplingPartialSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerItemsPerThread>;
using SamplingGroupSort =
    cub::BlockMergeSort<unsigned long long, kSamplerGroupBlock, kSamplerGroupItemsPerThread>;
inline constexpr int kSamplingTileWarps              = kSamplerBlock / 32;
inline constexpr int kSamplingTileIndexBits          = 9;
inline constexpr unsigned int kSamplingTileIndexMask = (1u << kSamplingTileIndexBits) - 1u;
static_assert(kSamplerPartialTileItems == (1 << kSamplingTileIndexBits));

using SamplingTileWarpSort     = cub::WarpMergeSort<unsigned long long, kSamplerItemsPerThread>;
using SamplingBf16TileWarpSort = cub::WarpMergeSort<unsigned int, kSamplerItemsPerThread>;

struct SamplingTileTopKStorage {
    typename SamplingTileWarpSort::TempStorage warp[kSamplingTileWarps];
    unsigned long long candidates[kSamplingTileWarps * kSamplerCandidateCap];
};

struct SamplingBf16TileTopKStorage {
    typename SamplingBf16TileWarpSort::TempStorage warp[kSamplingTileWarps];
    unsigned int candidates[kSamplingTileWarps * kSamplerCandidateCap];
};

union SamplingPartialTopKStorage {
    SamplingTileTopKStorage fp32;
    SamplingBf16TileTopKStorage bf16;
};

// True when the value is a real number. Used to fail a round whose logits diverged instead of
// sampling a plausible-looking token out of NaN (see kSamplerNonFiniteToken).
__device__ __forceinline__ bool sampling_value_is_finite(float value) {
    return value == value && value < CUDART_INF_F && value > -CUDART_INF_F;
}

__device__ __forceinline__ bool sampling_selected_logit_is_finite(const __nv_bfloat16* logits,
                                                                  std::int64_t base,
                                                                  std::int32_t index) {
    return index >= 0 && sampling_value_is_finite(__bfloat162float(logits[base + index]));
}

struct SamplingKeyGreater {
    __device__ __forceinline__ bool operator()(unsigned long long a, unsigned long long b) const {
        return a > b;
    }
};

struct SamplingUintGreater {
    __device__ __forceinline__ bool operator()(unsigned int a, unsigned int b) const {
        return a > b;
    }
};

__device__ inline unsigned long long sampling_block_max_key(unsigned long long key,
                                                            unsigned long long* warp_keys) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down_sync(kMask, key, offset);
        if (other > key) { key = other; }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_keys[warp] = key; }
    __syncthreads();
    key = (warp == 0 && lane < (blockDim.x >> 5)) ? warp_keys[lane] : 0ull;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(kMask, key, offset);
            if (other > key) { key = other; }
        }
    }
    return key;
}

__device__ __forceinline__ unsigned long long sampling_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform float in [0,1) from (seed, position, purpose, sub). Pure function of
// its inputs so it is safe under CUDA-graph replay (no mutable RNG state).
__device__ __forceinline__ float sampling_uniform(unsigned long long seed, int position,
                                                  int purpose, unsigned int sub) {
    unsigned long long key = seed;
    key                    = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40); // 24 bits
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// Candidate ordering: higher value wins, ties broken by lower vocab index.
__device__ __forceinline__ bool sampling_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

__device__ __forceinline__ unsigned long long sampling_sort_key(float v, int idx) {
    return score_id_order_key(v, idx);
}

__device__ __forceinline__ unsigned int sampling_bf16_tile_sort_key(__nv_bfloat16 v,
                                                                    int tile_index) {
    unsigned int bits = __bfloat16_as_ushort(v);
    if ((bits & 0x7fffu) == 0u) { bits = 0u; }
    const unsigned int ordered = (bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits ^ 0x8000u);
    return (ordered << kSamplingTileIndexBits) |
           (kSamplingTileIndexMask - static_cast<unsigned int>(tile_index));
}

__device__ __forceinline__ unsigned long long sampling_expand_bf16_tile_sort_key(unsigned int key,
                                                                                 int tile_start) {
    if (key == 0u) { return 0ull; }
    const unsigned int ordered = key >> kSamplingTileIndexBits;
    const unsigned int bits    = (ordered & 0x8000u) ? (ordered ^ 0x8000u) : ((~ordered) & 0xffffu);
    const int token            = tile_start + static_cast<int>(kSamplingTileIndexMask) -
                      static_cast<int>(key & kSamplingTileIndexMask);
    return sampling_sort_key(__uint_as_float(bits << 16), token);
}

__device__ __forceinline__ float sampling_key_float(unsigned long long key) {
    return score_from_order_key(key);
}

__device__ __forceinline__ int sampling_key_index(unsigned long long key) {
    return id_from_order_key(key);
}

__device__ __forceinline__ int sampling_candidate_cap(const SamplingConfig& cfg,
                                                      std::int32_t vocab) {
    // top_k is clamped to the pipeline cap: top_k <= 0 (no explicit limit) or a
    // top_k larger than the cap both select the full kSamplerCandidateCap set.
    int cap = kSamplerCandidateCap;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

__device__ __forceinline__ int sampling_partial_offset(const SamplingWorkspace& workspace, int col,
                                                       int partial, int j) {
    return ((col * workspace.partial_stride + partial) * kSamplerCandidateCap) + j;
}

__device__ inline void sampling_store_tile_topk(unsigned long long (&keys)[kSamplerItemsPerThread],
                                                int cap, SamplingWorkspace workspace, int col,
                                                int partial, SamplingTileTopKStorage& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingTileWarpSort(storage.warp[warp]).Sort(keys, SamplingKeyGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = lane * kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned long long key =
                storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int high     = static_cast<unsigned int>(key >> 32);
            const unsigned int max_high = __reduce_max_sync(kMergeMask, high);
            const unsigned int low      = high == max_high ? static_cast<unsigned int>(key) : 0u;
            const unsigned int max_low  = __reduce_max_sync(kMergeMask, low);
            const unsigned int winners =
                __ballot_sync(kMergeMask, high == max_high && low == max_low);
            const int source = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] =
                    (static_cast<unsigned long long>(max_high) << 32) | max_low;
            }
            if (lane == source) { ++position; }
        }
    }
}

__device__ inline void sampling_store_bf16_tile_topk(unsigned int (&keys)[kSamplerItemsPerThread],
                                                     int cap, int tile_start,
                                                     SamplingWorkspace workspace, int col,
                                                     int partial,
                                                     SamplingBf16TileTopKStorage& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingBf16TileWarpSort(storage.warp[warp]).Sort(keys, SamplingUintGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = lane * kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned int key     = storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int best    = __reduce_max_sync(kMergeMask, key);
            const unsigned int winners = __ballot_sync(kMergeMask, key == best);
            const int source           = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] = sampling_expand_bf16_tile_sort_key(best, tile_start);
            }
            if (lane == source) { ++position; }
        }
    }
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerCandidateCap + j;
}

// Whether mask position `column` of this decision may select token v (sampling.h SamplingMask).
__device__ __forceinline__ bool sampling_token_licensed(const SamplingConfig& c, int column, int v) {
    if (c.mask.words == nullptr) { return true; }
    const std::uint32_t word =
        c.mask.words[static_cast<std::int64_t>(column) * c.mask.stride + (v >> 5)];
    return ((word >> (static_cast<unsigned int>(v) & 31U)) & 1U) != 0U;
}

// True when the decision's scores are the raw logits: no penalty and no token mask. Only such rows
// may use the fused raw-argmax or raw-BF16 routes.
__device__ __forceinline__ bool sampling_uses_raw_logits(const SamplingConfig& c) {
    return c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f && c.mask.words == nullptr;
}

// Applies the token mask and presence/frequency penalties to a raw logit. `column` is the
// decision's mask position: 0 for an ordinary sample, the verification position for speculative
// acceptance. `overlay` carries a round-local count overlay of `column` tokens: tokens already
// committed earlier in the current speculative round but not yet flushed to the global
// `token_counts`. For speculative verify column `col` the overlay is exactly drafts[0..col-1]
// (statically known, since column `col` is only consumed when every earlier draft was accepted),
// so the penalty at each column sees the same prefix a per-token sampler would. Non-speculative
// callers pass no overlay. The scan is bounded by k and only runs when penalties are active, so it
// is free on the no-penalty path. An unlicensed token scores -inf before any filter, so top-k,
// min-p and top-p operate on the licensed tokens only.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v, const SamplingConfig& c,
                                                         const std::int32_t* overlay = nullptr,
                                                         int column                  = 0) {
    // Masking comes first: an unlicensed token must never win, not even through a NaN key.
    if (!sampling_token_licensed(c, column, v)) { return -CUDART_INF_F; }
    // A non-finite raw logit must sort as itself, unperturbed. Penalty subtraction on NaN is legal
    // IEEE-754 but not required to preserve the operand's bit pattern, and CUDA's hardware NaN
    // canonicalization does not: `NaN - presence_penalty` comes back as a different NaN encoding
    // than an untouched NaN, which score_id_order_key's total-order key treats as a *different,
    // higher* score. Since the penalty overlay is round-local (only already-drafted tokens get
    // subtracted), a poisoned verify column would then rank its drafted token above every other
    // (untouched-NaN) candidate in that same column -- letting a diverged forward pass spuriously
    // "match" the draft and licensing it as accepted instead of tripping the finiteness guard.
    // Returning raw unchanged keeps every non-finite entry in a column bit-identical, so ties fall
    // back to the id tie-break and the guard sees whichever one wins.
    if (!sampling_value_is_finite(raw)) { return raw; }
    float x = raw;
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) { return x; }
    int cnt = c.token_counts != nullptr ? c.token_counts[v] : 0;
    for (int j = 0; j < column; ++j) {
        if (overlay[j] == v) { ++cnt; }
    }
    if (cnt > 0) { x -= c.presence_penalty; }
    if (c.frequency_penalty != 0.0f) { x -= c.frequency_penalty * static_cast<float>(cnt); }
    return x;
}

__device__ __forceinline__ void sampling_insert_candidate(float* vals, int* idxs, int cap, float v,
                                                          int idx) {
    if (cap <= 0 || !sampling_better(v, idx, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && sampling_better(v, idx, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = v;
    idxs[pos] = idx;
}

__device__ inline void sampling_sort_tile_desc(float* vals, int* idxs) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= kSamplerTileItems; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            __syncthreads();
            if (other > tid) {
                const float ov        = vals[other];
                const int oi          = idxs[other];
                const bool descending = ((tid & size) == 0);
                const bool swap       = descending ? sampling_better(ov, oi, vals[tid], idxs[tid])
                                                   : sampling_better(vals[tid], idxs[tid], ov, oi);
                if (swap) {
                    const float tv = vals[tid];
                    const int ti   = idxs[tid];
                    vals[tid]      = ov;
                    idxs[tid]      = oi;
                    vals[other]    = tv;
                    idxs[other]    = ti;
                }
            }
            __syncthreads();
        }
    }
}

__device__ inline void sampling_normalize_support(const SamplingConfig& cfg, float* cand_val,
                                                  int* cand_idx, float* prob, int* n_support,
                                                  int n) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        while (n > 0 && !isfinite(cand_val[n - 1])) { --n; }
        if (n == 0 || !isfinite(cand_val[0])) {
            // No finite support: a diverged forward pass (NaN/inf logits) or a mask that licenses
            // nothing. Keep the best candidate as the single support entry with a NaN weight; the
            // callers' finiteness guards then publish kSamplerNonFiniteToken, which the runtime
            // rejects as a failed round. Trapping here instead would poison the CUDA context.
            prob[0]    = CUDART_NAN_F;
            *n_support = 1;
        } else {
            const float inv_temp = 1.0f / cfg.temperature;
            const float m        = cand_val[0] * inv_temp;
            float sum            = 0.0f;
            for (int j = 0; j < n; ++j) {
                const float e = __expf(cand_val[j] * inv_temp - m);
                prob[j]       = e;
                sum += e;
            }
            const float e0           = prob[0];
            const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
            const bool top_p_active  = (cfg.top_p < 1.0f);
            const float top_p_target = cfg.top_p * sum;
            float cum                = 0.0f;
            int support              = 0;
            for (int j = 0; j < n; ++j) {
                // Candidates are sorted, so a zero weight ends the support: every later candidate
                // is an underflowed token that must not be reachable by rounding.
                if (j != 0 && !(prob[j] > 0.0f)) { break; }
                if (min_p_thresh >= 0.0f && prob[j] < min_p_thresh) { break; }
                cum += prob[j];
                support = j + 1;
                if (top_p_active && cum >= top_p_target) { break; }
            }
            if (support < 1) { support = 1; }
            float ssum = 0.0f;
            for (int j = 0; j < support; ++j) { ssum += prob[j]; }
            const float inv = 1.0f / ssum;
            for (int j = 0; j < support; ++j) { prob[j] *= inv; }
            *n_support = support;
        }
    }
    __syncthreads();
}

// All threads of the block must call. Small-column exact truncated support
// builder used by unit tests and any <=256-token column. It sorts one shared
// tile and then applies the same top-p/min-p renormalization as the large path.
__device__ inline void
sampling_build_truncated_small(const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab,
                               const SamplingConfig& cfg, float* tile_val, int* tile_idx,
                               float* cand_val, int* cand_idx, float* prob, int* n_support,
                               const std::int32_t* overlay = nullptr, int column = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (tid < kSamplerTileItems) {
        if (tid < vocab) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + tid]), tid, cfg,
                                                    overlay, column);
            tile_val[tid] = x;
            tile_idx[tid] = tid;
        } else {
            tile_val[tid] = -CUDART_INF_F;
            tile_idx[tid] = INT_MAX;
        }
    }
    __syncthreads();
    sampling_sort_tile_desc(tile_val, tile_idx);
    if (tid < cap) {
        cand_val[tid] = tile_val[tid];
        cand_idx[tid] = tile_idx[tid];
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// Single-block fallback for large columns when the finite multi-block workspace
// route cannot represent the launch. It still reads each vocab entry once and
// keeps a bounded per-thread top-20, so it avoids a top_k*vocab global reread.
__device__ inline void sampling_build_truncated_block_fast(
    const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab, const SamplingConfig& cfg,
    float* merge_val, int* merge_idx, float* cand_val, int* cand_idx, float* prob, int* n_support,
    const std::int32_t* overlay = nullptr, int column = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab); // always <= kSamplerFastCandidates

    float local_val[kSamplerFastCandidates];
    int local_idx[kSamplerFastCandidates];
#pragma unroll
    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT_MAX;
    }

    const int fast_cap = cap;
    for (int v = tid; v < vocab; v += blockDim.x) {
        const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg, overlay,
                                                column);
        sampling_insert_candidate(local_val, local_idx, fast_cap, x, v);
    }

    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        const int off  = tid * kSamplerFastCandidates + j;
        merge_val[off] = local_val[j];
        merge_idx[off] = local_idx[j];
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < cap; ++j) {
            cand_val[j] = -CUDART_INF_F;
            cand_idx[j] = INT_MAX;
        }
        const int merge_n = blockDim.x * kSamplerFastCandidates;
        for (int p = 0; p < merge_n; ++p) {
            const int idx = merge_idx[p];
            if (idx == INT_MAX) { continue; }
            sampling_insert_candidate(cand_val, cand_idx, fast_cap, merge_val[p], idx);
        }
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, fast_cap);
}

// thread-0 helper: inverse-CDF pick over a normalized `prob[0..n-1]` support,
// optionally excluding `exclude` (a rejected draft token) and renormalizing over
// the remainder. `u` is a uniform in [0,1). Returns the chosen vocab id.
__device__ __forceinline__ int sampling_pick_from_support(const int* cand_idx, const float* prob,
                                                          int n, int exclude, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        mass += prob[j];
    }
    if (mass <= 0.0f) {
        // Degenerate: support is only the excluded token. Return it (caller only
        // reaches this when accept probability was ~1, i.e. it will not happen).
        return cand_idx[0];
    }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = -1;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        acc += prob[j];
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

} // namespace ninfer::ops
