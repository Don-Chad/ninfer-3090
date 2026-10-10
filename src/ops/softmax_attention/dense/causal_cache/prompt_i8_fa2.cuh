#pragma once

// FlashAttention-2 style causal prompt kernel for the INT8-G64 cache (sm_80/86/89).
//
// Adapted from Infernix's fast INT8 prompt kernel and its key-split merge
// (github.com/wallawalla47/infernix, commits 60e67d06 and dbd1f374,
// src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh and fast_prompt_common.cuh),
// Copyright the Infernix authors, licensed under the Apache License, Version 2.0. Changes for this
// fork: the 8-bit PV form is dropped, the split merge reads the batch metadata instead of a raw
// valid-column pointer, the FP16 MMA comes from ops/common/mma.cuh, and launch planning lives in
// prompt_i8_fa2_plan.h with constants measured on the RTX 3090.
//
//   * Each warp owns 16 query rows of one head for the whole key sweep (eight warps per CTA, or
//     four for launches too narrow to occupy every SM with 128-row CTAs), so scores,
//     probabilities and the D256 output accumulator never leave registers and no warp waits for
//     another between QK and PV. The only CTA barrier is the one that publishes a key tile.
//   * Q and cached K use the same fixed register-only D256 rotation before their private G64
//     encoders; QK stays INT8 through m16n8k32.s8 Tensor Cores with the per-group scales applied
//     in FP32.
//   * K and V codes are double-buffered as raw INT8 pages (one 64-key page per tile). V is decoded
//     in registers: ldmatrix.trans on byte pairs yields, per lane, two adjacent keys for two
//     adjacent dimensions, which is exactly one FP16 B fragment for an even-dimension and an
//     odd-dimension n8 tile. Codes widen exactly to FP16 and take their represented FP16 group
//     scale with one rounding, as the dequantizing kernels do.
//   * PV runs FP16 Tensor Cores with FP16 accumulation over the 64 keys of a tile and is promoted
//     into the FP32 output accumulator once per tile. Every probability is at most one, so a tile
//     partial is bounded by 64 * 127 * max_scale; a tile whose largest V scale could exceed the
//     FP16 range decodes V with its scales divided by an exact power of two and multiplies the
//     partial back at promotion.
//   * CTAs issue longest-first, so a causal prompt's heaviest row blocks do not form the tail.
//   * A split launch (gridDim.z > 1, eight-warp CTAs only) gives each CTA a contiguous run of key
//     pages; each CTA publishes its normalized FP32 rows and their (max, sum), and
//     causal_attention_prompt_i8_fa2_merge_kernel combines the splits in FP32.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/common/math.h"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptFa2Bc     = 64;
inline constexpr int kCausalPromptFa2Groups = kCausalPromptHeadDim / kKVCacheInt8Group;

inline constexpr int kCausalPromptFa2TileBytes = kCausalPromptFa2Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptFa2ScaleBytes =
    kCausalPromptFa2Bc * kCausalPromptFa2Groups * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptFa2StageBytes =
    2 * kCausalPromptFa2TileBytes + 2 * kCausalPromptFa2ScaleBytes;

template <int Warps>
struct CausalPromptFa2Shape {
    static_assert(Warps == 4 || Warps == 8);
    static constexpr int Threads   = Warps * 32;
    static constexpr int Br        = Warps * 16;
    static constexpr int QBytes    = Br * kCausalPromptHeadDim;
    static constexpr int SmemBytes = QBytes + 2 * kCausalPromptFa2StageBytes;
};

// A 64-key FP16 partial is bounded by 64 * 127 * max_scale (every probability is at most one).
// Keeping max_scale at or below 8 leaves that bound (65,024), with FP16 rounding slack, under the
// largest finite FP16 value 65,504.
inline constexpr float kCausalPromptFa2F16PartialScaleLimit = 8.0f;

static_assert(kCausalPromptFa2Bc == kPagedKVPageSize);
static_assert(kCausalPromptFa2Groups == 4);
// sm_86 allows 101,376 B of dynamic shared memory per block: one CTA per SM either shape.
static_assert(CausalPromptFa2Shape<8>::SmemBytes == 100352);
static_assert(CausalPromptFa2Shape<4>::SmemBytes == 83968);

// One ldmatrix.trans b16 lane of an INT8 [key][d] tile holds the codes
// {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}. Returns the FP16 B-fragment halves
// {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code widened exactly and multiplied
// once by its key's represented group scale.
__device__ __forceinline__ void causal_prompt_fa2_decode_v_pair(unsigned codes, unsigned scales,
                                                                unsigned& even, unsigned& odd) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte, so subtracting 1152
    // recovers the signed code exactly.
    const unsigned biased = codes ^ 0x80808080u;
    unsigned e            = __byte_perm(biased, 0x64646464u, 0x4240);
    unsigned o            = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), s2);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), s2);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

// Split partial layout for key split s, column c and query head h: the D256 row at
// ((s * width + c) * QHeads + h) * 256 and its (max, sum) pair at that row index.
template <typename Geometry>
__host__ __device__ __forceinline__ std::int64_t causal_prompt_fa2_partial_row(int split,
                                                                               int column,
                                                                               int q_head,
                                                                               int width) {
    return (static_cast<std::int64_t>(split) * width + column) * Geometry::QHeads + q_head;
}

template <typename Geometry, typename Metadata, int Warps, bool Split>
__global__ __launch_bounds__(CausalPromptFa2Shape<Warps>::Threads, 1) void
causal_attention_prompt_i8_fa2_kernel(const __nv_bfloat16* __restrict__ q,
                                      const std::int8_t* __restrict__ cache_k,
                                      const std::int8_t* __restrict__ cache_v,
                                      const __half* __restrict__ cache_k_scale,
                                      const __half* __restrict__ cache_v_scale, Metadata metadata,
                                      const std::int32_t* __restrict__ positions, float scale,
                                      __nv_bfloat16* __restrict__ out, std::int32_t width,
                                      float* __restrict__ partial_rows,
                                      float2* __restrict__ partial_stats) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int DB16          = D / 2;
    using Shape                 = CausalPromptFa2Shape<Warps>;
    constexpr int Threads       = Shape::Threads;
    constexpr int Br            = Shape::Br;
    constexpr int Bc            = kCausalPromptFa2Bc;
    constexpr int Groups        = kCausalPromptFa2Groups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int DBlocks       = D / 16;
    constexpr int GroupDBlocks  = kKVCacheInt8Group / 16;
    constexpr int PassDBlocks   = 2;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(GroupDBlocks == 4);
    static_assert(!Split || Warps == 8, "only eight-warp CTAs split");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Longest-first issue order: the first QHeads CTAs take the last row block of every head.
    const int row_blocks = static_cast<int>(gridDim.x);
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % Geometry::QHeads;
    const int q0         = (row_blocks - 1 - linear / Geometry::QHeads) * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    // Rows past the valid token count are inactive columns and publish zeros.
    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (token >= width) { return; }
        const bool valid   = token < tokens;
        const uint2 packed = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                        pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        store_vec(&out[causal_prompt_q_index<Geometry>(q_head, d0, token)], packed);
    };

    if (q0 >= tokens) {
        // The merge publishes a split launch's inactive columns.
        if constexpr (!Split) {
            for (int element = tid; element < Br * (D / 4); element += Threads) {
                const int row = element / (D / 4);
                store_row(row, (element - row * (D / 4)) * 4, 0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows                  = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    // Key pages [kb_begin, kb_end) of this split; an empty run publishes neutral statistics.
    const int split           = Split ? static_cast<int>(blockIdx.z) : 0;
    const int pages_per_split = Split ? div_up(key_blocks, static_cast<int>(gridDim.z)) : 0;
    const int kb_begin        = Split ? min(key_blocks, split * pages_per_split) : 0;
    const int kb_end          = Split ? min(key_blocks, kb_begin + pages_per_split) : key_blocks;

    const auto stage_base = [&](int stage) {
        return smem_raw + Shape::QBytes + stage * kCausalPromptFa2StageBytes;
    };

    // One tile is one physical page of this KV head: 16 KiB of K codes, 16 KiB of V codes and
    // their G64 scales, all contiguous in the cache. Keys past the CTA's last visible key are
    // zero-filled so masked columns stay finite.
    const auto issue_tile = [&](int kb) {
        unsigned char* base = stage_base(kb & 1);
        std::int8_t* k_s    = reinterpret_cast<std::int8_t*>(base);
        std::int8_t* v_s    = k_s + kCausalPromptFa2TileBytes;
        __half* ks_s        = reinterpret_cast<__half*>(v_s + kCausalPromptFa2TileBytes);
        __half* vs_s        = ks_s + Bc * Groups;
        const int page      = block_table[kb];
        const int valid     = min(Bc, max_query_abs + 1 - kb * Bc);
        const std::int64_t code_base =
            kv_cache_int8_quant_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
        for (int i = 0; i < kCausalPromptFa2TileBytes / 16 / Threads; ++i) {
            const int chunk        = tid + i * Threads;
            const int key          = chunk >> 4;
            const int c            = chunk & 15;
            const int dst          = key * D + ((c ^ (key & 7)) << 4);
            const int bytes        = key < valid ? 16 : 0;
            const std::int64_t src = code_base + key * D + c * 16;
            cp_async_zfill<16, Cache::cg>(k_s + dst, cache_k + src, bytes);
            cp_async_zfill<16, Cache::cg>(v_s + dst, cache_v + src, bytes);
        }
        if (tid < 2 * Bc) {
            const int key = tid & (Bc - 1);
            const std::int64_t src =
                kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, key);
            const __half* scales = tid < Bc ? cache_k_scale : cache_v_scale;
            __half* dst          = (tid < Bc ? ks_s : vs_s) + key * Groups;
            cp_async_zfill<8>(dst, scales + src, key < valid ? 8 : 0);
        }
        cp_commit();
    };

    // The first tile's copy overlaps the Q encoding below; the first loop barrier publishes both.
    if (kb_begin < kb_end) { issue_tile(kb_begin); }

    // Each warp rotates and encodes its own 16 rows, keeping the two rows each lane owns in the
    // MMA C layout as register scales. Rows go QRows at a time: one row is a chain of dependent
    // shuffles (the rotation's five lane stages and four group maxima) that interleaving
    // independent rows overlaps. The row-group loop stays rolled: unrolled, this once-per-CTA
    // prologue was ~8.7K instructions of straight-line code, fetched cold by every CTA, and set a
    // ~60 us floor under short launches.
    constexpr int QRows = 4;
    float q_scale_r[2][Groups] = {};
#pragma unroll 1
    for (int r0 = 0; r0 < 16; r0 += QRows) {
        float q_values[8 * QRows];
#pragma unroll
        for (int i = 0; i < QRows; ++i) {
            const int row    = warp * 16 + r0 + i;
            const bool valid = row < rows;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                q_values[8 * i + k] =
                    valid ? __bfloat162float(
                                q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * k, q0 + row)])
                          : 0.0f;
            }
        }
        normalized_hadamard_d256_rows_inplace<QRows>(q_values, lane);
        float mx[QRows * Groups];
#pragma unroll
        for (int j = 0; j < QRows * Groups; ++j) {
            mx[j] = fmaxf(fabsf(q_values[2 * j]), fabsf(q_values[2 * j + 1]));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int j = 0; j < QRows * Groups; ++j) {
                mx[j] = fmaxf(mx[j], __shfl_xor_sync(FullMask, mx[j], offset));
            }
        }
#pragma unroll
        for (int i = 0; i < QRows; ++i) {
            const int r   = r0 + i;
            const int row = warp * 16 + r;
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                const int d0    = grp * kKVCacheInt8Group + lane;
                const float x0  = q_values[8 * i + 2 * grp];
                const float x1  = q_values[8 * i + 2 * grp + 1];
                const float m   = mx[i * Groups + grp];
                const float qs  = m > 0.0f ? m / 127.0f : 0.0f;
                const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
                causal_prompt_store_byte_swizzled(q_i8, row, d0,
                                                  kv_cache_int8_quant_code(x0, inv));
                causal_prompt_store_byte_swizzled(q_i8, row, d0 + 32,
                                                  kv_cache_int8_quant_code(x1, inv));
                // r is not a compile-time index here; select into both halves so the scales stay
                // in registers.
                const bool mine = gid == (r & 7);
                q_scale_r[0][grp] = mine && r < 8 ? qs : q_scale_r[0][grp];
                q_scale_r[1][grp] = mine && r >= 8 ? qs : q_scale_r[1][grp];
            }
        }
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;

    // Rows this warp owns, for warp-uniform tile skipping and mask elision.
    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    float acc[DBlocks][2][4];
#pragma unroll
    for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    // What QK hands to PV for one tile: the probability A fragments, the row rescale factors, and
    // the exact power of two that keeps the tile's FP16 partials representable.
    unsigned pa[PVKs][4];
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    int tile_shift    = 0;
    bool tile_live    = false;

    const auto qk_softmax = [&](int kb) {
        tile_live    = false;
        const int k0 = kb * Bc;
        if (!warp_active || k0 > warp_max_qabs) { return; }
        const unsigned char* base  = stage_base(kb & 1);
        const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(base);
        const __half* ks_s =
            reinterpret_cast<const __half*>(base + 2 * kCausalPromptFa2TileBytes);
        const __half* vs_s = ks_s + Bc * Groups;

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            unsigned af[GroupKc][4];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
                const int keya  = nt * 8 + 2 * lid;
                const float ks0 = __half2float(ks_s[keya * Groups + grp]);
                const float ks1 = __half2float(ks_s[(keya + 1) * Groups + grp]);
                const float qs0 = q_scale_r[0][grp];
                const float qs1 = q_scale_r[1][grp];
                score[nt][0]    = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                score[nt][1]    = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                score[nt][2]    = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                score[nt][3]    = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
            }
        }

        // A tile wholly at or below the warp's first row needs no causal mask.
        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        float bm0            = -CUDART_INF_F;
        float bm1            = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        // A row with no visible key yet (an inactive row, or a split run wholly past this row's
        // position) keeps max -inf; exp2(-inf - 0) is then an exact zero probability.
        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        // Probabilities become the PV A fragments directly: k-step j covers keys 16j..16j+15,
        // which are score tiles 2j (keys 2t, 2t+1) and 2j+1 (keys 8+2t, 9+2t).
        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
        }
        // Row sums stay lane-partial; the quad is reduced once after the sweep.
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);

        // A tile whose largest V scale exceeds the limit decodes V with its scales divided by an
        // exact power of two and multiplies the partial back at promotion.
        static_assert(Bc * Groups == 32 * 8);
        const uint4 v8      = load_vec<uint4>(&vs_s[8 * lane]);
        const __half2 vmax2 = __hmax2(
            __hmax2(__habs2(load_vec<__half2>(&v8.x)), __habs2(load_vec<__half2>(&v8.y))),
            __hmax2(__habs2(load_vec<__half2>(&v8.z)), __habs2(load_vec<__half2>(&v8.w))));
        const float vmax = warp_max(fmaxf(__low2float(vmax2), __high2float(vmax2)), FullMask);
        tile_shift       = 0;
        if (vmax > kCausalPromptFa2F16PartialScaleLimit) {
            // An infinite or NaN scale would never halve below the limit; the largest finite FP16
            // value takes 13 halvings.
            const float bounded = fminf(vmax, 65504.0f);
            while (ldexpf(bounded, -tile_shift) > kCausalPromptFa2F16PartialScaleLimit) {
                ++tile_shift;
            }
        }
        tile_live = true;
    };

    const auto pv = [&](int kb) {
        if (!tile_live) { return; }
        const std::int8_t* v_s =
            reinterpret_cast<const std::int8_t*>(stage_base(kb & 1)) + kCausalPromptFa2TileBytes;
        const __half* vs_s =
            reinterpret_cast<const __half*>(v_s + kCausalPromptFa2TileBytes) + Bc * Groups;
        const __half2 mul   = __float2half2_rn(ldexpf(1.0f, -tile_shift));
        const float unscale = ldexpf(1.0f, tile_shift);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            // Group scales of the keys each lane supplies: (2t, 2t+1) and (8+2t, 9+2t) per k-step.
            unsigned vsc[PVKs][2];
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                const int key = j * 16 + 2 * lid;
                __half2 lo =
                    __halves2half2(vs_s[key * Groups + grp], vs_s[(key + 1) * Groups + grp]);
                __half2 hi =
                    __halves2half2(vs_s[(key + 8) * Groups + grp], vs_s[(key + 9) * Groups + grp]);
                if (tile_shift != 0) {
                    lo = __hmul2(lo, mul);
                    hi = __hmul2(hi, mul);
                }
                vsc[j][0] = load_vec<unsigned>(&lo);
                vsc[j][1] = load_vec<unsigned>(&hi);
            }
#pragma unroll
            for (int pass = 0; pass < GroupDBlocks / PassDBlocks; ++pass) {
                const int db0 = grp * GroupDBlocks + pass * PassDBlocks;
                unsigned h[PassDBlocks][2][2];
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) { h[b][p][0] = h[b][p][1] = 0u; }
                }
#pragma unroll
                for (int j = 0; j < PVKs; ++j) {
#pragma unroll
                    for (int q2 = 0; q2 < PassDBlocks / 2; ++q2) {
                        // Matrices: keys 16j+0..7 and 16j+8..15 of d-block db, then of db + 1.
                        const int db    = db0 + 2 * q2;
                        const int key   = j * 16 + ((a_mat & 1) << 3) + a_rin;
                        const int chunk = db + (a_mat >> 1);
                        unsigned r[4];
                        ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                      smem_addr(&v_s[key * D + ((chunk ^ (key & 7)) << 4)]));
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            unsigned even_lo, odd_lo, even_hi, odd_hi;
                            causal_prompt_fa2_decode_v_pair(r[2 * b], vsc[j][0], even_lo, odd_lo);
                            causal_prompt_fa2_decode_v_pair(r[2 * b + 1], vsc[j][1], even_hi,
                                                            odd_hi);
                            unsigned(&he)[2] = h[2 * q2 + b][0];
                            unsigned(&ho)[2] = h[2 * q2 + b][1];
                            mma_f16_acc16(he[0], he[1], pa[j][0], pa[j][1], pa[j][2], pa[j][3],
                                          even_lo, even_hi);
                            mma_f16_acc16(ho[0], ho[1], pa[j][0], pa[j][1], pa[j][2], pa[j][3],
                                          odd_lo, odd_hi);
                        }
                    }
                }
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        float(&a)[4] = acc[db0 + b][p];
                        float2 r0    = __half22float2(load_vec<__half2>(&h[b][p][0]));
                        float2 r1    = __half22float2(load_vec<__half2>(&h[b][p][1]));
                        if (tile_shift != 0) {
                            r0.x *= unscale;
                            r0.y *= unscale;
                            r1.x *= unscale;
                            r1.y *= unscale;
                        }
                        a[0] = __fmaf_rn(a[0], tile_alpha0, r0.x);
                        a[1] = __fmaf_rn(a[1], tile_alpha0, r0.y);
                        a[2] = __fmaf_rn(a[2], tile_alpha1, r1.x);
                        a[3] = __fmaf_rn(a[3], tile_alpha1, r1.y);
                    }
                }
            }
        }
    };

    // Every warp publishes one tile per barrier; the next tile's copy overlaps this tile's math.
#pragma unroll 1
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < kb_end) { issue_tile(kb + 1); }
        qk_softmax(kb);
        pv(kb);
    }

    running_l0         = warp_sum<4>(running_l0, FullMask);
    running_l1         = warp_sum<4>(running_l1, FullMask);
    const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
    const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;
    // Even n8 tiles hold dimensions 16b + 2n and odd tiles 16b + 2n + 1, so lane (g, t) owns the
    // four contiguous dimensions 16b + 4t .. 16b + 4t + 3 of its two rows.
    if constexpr (Split) {
        // Each split row is normalized by its own sum; the merge weights the splits.
        const auto publish = [&](int row, int e, float maximum, float sum, float inv) {
            if (row >= rows) { return; }
            const std::int64_t index =
                causal_prompt_fa2_partial_row<Geometry>(split, q0 + row, q_head, width);
            if (lid == 0) { partial_stats[index] = make_float2(maximum, sum); }
            float* target = partial_rows + index * D;
#pragma unroll
            for (int b = 0; b < DBlocks; ++b) {
                store_vec(target + b * 16 + 4 * lid,
                          make_float4(acc[b][0][e] * inv, acc[b][1][e] * inv,
                                      acc[b][0][e + 1] * inv, acc[b][1][e + 1] * inv));
            }
        };
        publish(row0, 0, running_m0, running_l0, inv_l0);
        publish(row1, 2, running_m1, running_l1, inv_l1);
    } else {
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            const int d0 = b * 16 + 4 * lid;
            store_row(row0, d0, acc[b][0][0] * inv_l0, acc[b][1][0] * inv_l0,
                      acc[b][0][1] * inv_l0, acc[b][1][1] * inv_l0);
            store_row(row1, d0, acc[b][0][2] * inv_l1, acc[b][1][2] * inv_l1,
                      acc[b][0][3] * inv_l1, acc[b][1][3] * inv_l1);
        }
    }
}

// Combines the key splits of one column and query head in FP32. Every split row is normalized by
// its own sum, so the merged row is sum_s w_s * row_s / sum_s w_s with
// w_s = sum_s * 2^((m_s - M) * scale * log2 e). Columns past the valid count publish zeros.
template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(kCausalPromptHeadDim) void causal_attention_prompt_i8_fa2_merge_kernel(
    const float* __restrict__ partial_rows, const float2* __restrict__ partial_stats,
    Metadata metadata, std::int32_t width, std::int32_t splits, float scale_l2,
    __nv_bfloat16* __restrict__ out) {
    const int column = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int d      = static_cast<int>(threadIdx.x);
    const int tokens = metadata.valid_tokens(width);
    const auto index = causal_prompt_q_index<Geometry>(q_head, d, column);
    if (column >= tokens) {
        out[index] = __float2bfloat16(0.0f);
        return;
    }
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        const float2 stats =
            partial_stats[causal_prompt_fa2_partial_row<Geometry>(split, column, q_head, width)];
        if (stats.y > 0.0f) { maximum = fmaxf(maximum, stats.x); }
    }
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t row =
            causal_prompt_fa2_partial_row<Geometry>(split, column, q_head, width);
        const float2 stats = partial_stats[row];
        if (stats.y > 0.0f) {
            const float weight = stats.y * exp2f((stats.x - maximum) * scale_l2);
            numerator = __fmaf_rn(weight, partial_rows[row * kCausalPromptHeadDim + d], numerator);
            denominator += weight;
        }
    }
    out[index] = __float2bfloat16(denominator > 0.0f ? numerator / denominator : 0.0f);
}

} // namespace ninfer::ops
