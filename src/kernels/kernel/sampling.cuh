#pragma once

// qus::kernels - sample_column kernel. One CUDA block handles one logits column
// and reduces over vocab. The greedy branch (temperature <= 0) is an exact
// argmax with lowest-index tie-break; the sampling branch builds the truncated
// target distribution and draws a token with a stateless counter-based RNG.

#include "kernels/kernel/sampling_device.cuh"

namespace qus::kernels {

__launch_bounds__(kSamplerBlock) __global__ void sample_column_kernel(
    const __nv_bfloat16* logits, std::int32_t* out, const SamplingConfig* cfg_ptr,
    const std::int32_t* pos_base, std::int32_t purpose, std::int32_t vocab) {
    const int t              = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(t) * vocab;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // Greedy: exact argmax over raw logits. Bit-identical to argmax().
    if (!(cfg.temperature > 0.0f)) {
        float bv = -CUDART_INF_F;
        int bi   = INT_MAX;
        for (int v = tid; v < vocab; v += blockDim.x) {
            const float x = __bfloat162float(logits[base + v]);
            if (sampling_better(x, v, bv, bi)) {
                bv = x;
                bi = v;
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) { out[t] = red_idx[0]; }
        return;
    }

    const int partial_blocks = (vocab + kSamplerPartialTileItems - 1) / kSamplerPartialTileItems;
    const bool scratch_capable =
        (gridDim.x <= kSamplerScratchColumns) && (partial_blocks <= kSamplerScratchPartialBlocks);
    if (vocab > kSamplerTileItems && scratch_capable) { return; }

    __shared__ float cand_val[kSamplerMaxCandidates];
    __shared__ int cand_idx[kSamplerMaxCandidates];
    __shared__ float prob[kSamplerMaxCandidates];
    __shared__ int n_support;
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];

    if (vocab <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, vocab, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, vocab, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, *pos_base + t, purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j];  // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[t] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr, std::int32_t vocab) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = *cfg_ptr;
    const int partial_blocks = static_cast<int>(gridDim.x);
    if (!(cfg.temperature > 0.0f) || vocab <= kSamplerTileItems ||
        col >= kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ typename SamplingPartialBlockSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];
    int items[kSamplerItemsPerThread];

    const int cap = sampling_candidate_cap(cfg, vocab);
    const std::int64_t base = static_cast<std::int64_t>(col) * vocab;
    const int tile_start = partial * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < vocab) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg);
            keys[item] = sampling_sort_key(x, v);
            items[item] = v;
        } else {
            keys[item] = 0ull;
            items[item] = INT_MAX;
        }
    }
    SamplingPartialBlockSort(sort_storage).SortDescending(keys, items);

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off = sampling_partial_offset(col, partial, rank);
            sampling_partial_val[off] = sampling_key_float(keys[item]);
            sampling_partial_idx[off] = items[item];
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* pos_base,
    std::int32_t purpose, std::int32_t vocab, std::int32_t partial_blocks) {
    const int col            = static_cast<int>(blockIdx.x);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;
    if (!(cfg.temperature > 0.0f) || vocab <= kSamplerTileItems ||
        gridDim.x > kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ float merge_val[(kSamplerBlock / 32) * kSamplerFastCandidates];
    __shared__ int merge_idx[(kSamplerBlock / 32) * kSamplerFastCandidates];
    __shared__ typename SamplingFinalizeBlockSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerMaxCandidates];
    __shared__ int cand_idx[kSamplerMaxCandidates];
    __shared__ float prob[kSamplerMaxCandidates];
    __shared__ int n_support;

    sampling_merge_partials_to_support(col, partial_blocks, cfg, sort_storage, merge_val,
                                       merge_idx, cand_val, cand_idx, prob, &n_support, vocab);

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, *pos_base + col, purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j];
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[col] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

} // namespace qus::kernels
