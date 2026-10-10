#pragma once

// Tiled causal prompt kernel for the rk8v4 and rk4v4 caches (packed INT4 values; rk4v4 also packs
// Lloyd-Max keys) for the registered head geometries. The INT8-G64 cache runs the FA2-style kernel
// in prompt_i8_fa2.cuh instead. Q and cached K use the same fixed register-only D256 rotation
// before their private G64 encoders. QK stays INT8 through m16n8k32.s8 Tensor Cores; V alone is
// dequantized with packed FP16 arithmetic while producer warps execute QK. Sixteen warps split
// each 16-row FP16 PV output (FP32 accumulate) across four 64-dimension slices.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cstdint>

// Swept on sm_86: the shipped 120 comes from an SM120 tuning note, and shared memory (93,184 B of
// the 99 KiB budget) pins this kernel to one CTA per SM on either card, so registers cannot buy
// occupancy here -- only fewer spills. 512 threads x 128 registers is exactly the 65,536 file.
// The 2026-09-19 INT8-PV ablation that used to be recorded here is superseded. Its timing split
// (4096 tokens over 25.6K keys: PV MMA 27.6 %, V dequant 5.6 %, QK + softmax + staging 66.8 %)
// showed this kernel only a third tensor-bound, and concluded that only restructuring the other
// 66.8 % would pay. prompt_i8_fa2.cuh is that restructure for the INT8-G64 cache (register-resident
// P, no V worker warps, double-buffered pages, FP16-accumulated PV, longest-first issue, key
// splits); measured against this kernel on the RTX 3090 (2026-10-10, both geometries) it takes
// 0.55-0.64x the time on 4096-token chunks and 0.13-0.45x on follow-ups of up to 256 tokens over
// 32K-184K cached keys. The record's "int8 P improves perplexity" note is withdrawn: lower
// perplexity from dropping each tile's small weights is not more exact, and Infernix's KL study
// measured 2.2x the KL for 8-bit PV at long context. rk8v4 and rk4v4, and INT8-G64 over at most 256
// visible keys, stay on this kernel.

#ifndef NINFER_PROMPT_I8_MAXNREG
#define NINFER_PROMPT_I8_MAXNREG 120
#endif

namespace ninfer::ops {

inline constexpr int kCausalPromptI8Warps      = 16;
inline constexpr int kCausalPromptI8Threads    = kCausalPromptI8Warps * 32;
inline constexpr int kCausalPromptI8Br         = 64;
inline constexpr int kCausalPromptI8Bc         = 64;
inline constexpr int kCausalPromptI8Groups     = kCausalPromptHeadDim / kKVCacheInt8Group;
inline constexpr int kCausalPromptI8DB16       = kCausalPromptHeadDim / 2;
inline constexpr int kCausalPromptI8RowTiles   = kCausalPromptI8Br / 16;
inline constexpr int kCausalPromptI8DConsumers = kCausalPromptI8Warps / kCausalPromptI8RowTiles;

inline constexpr int kCausalPromptI8QBytes = kCausalPromptI8Br * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8QScaleBytes =
    kCausalPromptI8Br * kCausalPromptI8Groups * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptI8KBytes = kCausalPromptI8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8VBytes = kCausalPromptI8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8VStageBytes =
    kCausalPromptI8Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8PBytes =
    kCausalPromptI8Br * kCausalPromptI8Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8VGroups =
    kCausalPromptHeadDim / kKVCacheInt4ValueGroup;
// Key scale rows plus value scale rows. Value rows are always the wider packed-int4 stride so
// both codings share one arena layout; INT8 uses the leading kCausalPromptI8Groups entries. The
//512 extra bytes do not change occupancy, which is one CTA per SM either way.
inline constexpr int kCausalPromptI8ScaleBytes =
    kCausalPromptI8Bc * (kCausalPromptI8Groups + kCausalPromptI8VGroups) *
    static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8StatsBytes =
    2 * kCausalPromptI8Br * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptI8SmemBytes =
    kCausalPromptI8QBytes + kCausalPromptI8QScaleBytes + kCausalPromptI8KBytes +
    kCausalPromptI8VBytes + kCausalPromptI8VStageBytes + kCausalPromptI8PBytes +
    kCausalPromptI8ScaleBytes + kCausalPromptI8StatsBytes;

static_assert(kCausalPromptI8Groups == 4);
static_assert(kCausalPromptI8DConsumers == 4);
static_assert(kCausalPromptI8SmemBytes == 93184);

// The f16 value loaders this kernel needs now live in ops/kv_cache/int8_g64_codec.cuh, beside their
// bf16 counterparts: kv_cache_int8_dequant_f16x8_from and kv_cache_int4_dequant_f16x8_from. They
// used to be defined locally here, which is precisely why upstream's small-T kernel reached for the
// *bf16* loader when it needed f16 codes -- the shared codec looked like it had no f16 variant for
// the INT8 codings, and bf16 and f16 are the same width, so the mistake compiled and silently
// reinterpreted every value.
//
// The shared versions take a float scale and multiply in FP32 before the single rounding to f16,
// where these took a __half and multiplied in half2. That is the same result, not an approximation
// of it: an int8 code carries at most 8 significant bits and an f16 scale at most 11, so their
// product needs at most 19 and is exact in FP32's 24-bit mantissa. Both forms therefore round
// exactly once, to the same value. Verified empirically as well -- every OP_ERROR_STATS line in
// ninfer_softmax_attention_test is byte-identical across the change.

// PackedValues selects the rk8v4 half-width value plane. Packed bytes are staged into the leading
// half of the same V slot the INT8 coding uses, so the shared-memory footprint and therefore the
// occupancy of both instantiations are identical; only the global traffic halves.
//
// PackedKeys selects the rk4v4 key plane (4-bit Lloyd-Max indices). As in the small-T kernel, the
// next tile's packed keys are loaded into registers where the INT8 path issues its cp.async, stay
// in flight across the PV MMAs, and are expanded into the unchanged INT8 K tile before the tile
// barrier.
template <typename Geometry, typename Metadata, bool PackedValues = false, bool PackedKeys = false>
__global__ __maxnreg__(NINFER_PROMPT_I8_MAXNREG) void causal_attention_prompt_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::int8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kCausalPromptI8Br;
    constexpr int Bc            = kCausalPromptI8Bc;
    constexpr int DB16          = kCausalPromptI8DB16;
    constexpr int Groups        = kCausalPromptI8Groups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (kCausalPromptI8DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kCausalPromptI8RowTiles;
    constexpr int VWorkerWarps  = kCausalPromptI8Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + kCausalPromptI8QBytes);
    std::int8_t* k_i8 = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                       kCausalPromptI8QScaleBytes);
    std::int8_t* v_i8 = k_i8 + kCausalPromptI8KBytes;
    __half* v_f16     = reinterpret_cast<__half*>(v_i8 + kCausalPromptI8VBytes);
    __half* p_s       = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) +
                                                  kCausalPromptI8VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kCausalPromptI8PBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    constexpr int VGroups = kCausalPromptI8VGroups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * VGroups);
    float* final_l_s     = alpha_s + Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 kCausalPromptI8Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Quantize Q cooperatively. One full warp rotates and encodes one D256 row at a time.
    for (int row = warp; row < Br; row += kCausalPromptI8Warps) {
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = 0.0f;
            if (row < tile_rows) {
                q_values[r] =
                    __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)]);
            }
        }
        normalized_hadamard_d256_inplace(q_values, lane);

#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float absmax    = fmaxf(fabsf(x0), fabsf(x1));
            absmax          = warp_max(absmax, FullMask);
            const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_prompt_store_byte_swizzled(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_prompt_store_byte_swizzled(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    constexpr int KChunks          = Bc * (D / 16);
    constexpr int KChunksPerThread = (KChunks + kCausalPromptI8Threads - 1) / kCausalPromptI8Threads;
    uint2 k_packed[PackedKeys ? KChunksPerThread : 1];
    // Chunks past the causal limit keep their placeholder's expansion: their scales are staged as
    // zero and their scores masked, exactly as the zero-filled INT8 chunks are.
    auto commit_k_tile = [&]() {
        if constexpr (PackedKeys) {
#pragma unroll
            for (int i = 0; i < KChunksPerThread; ++i) {
                const int chunk = tid + i * kCausalPromptI8Threads;
                if (chunk < KChunks) {
                    const int key_l = chunk / (D / 16);
                    const int dc    = chunk - key_l * (D / 16);
                    store_vec(&k_i8[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2],
                              kv_cache_lloyd4_expand16(k_packed[i]));
                }
            }
        }
    };

    // A FullTile lies wholly at or below the first query row, so every key is visible and the
    // per-key causal test drops out of the staging.
    auto issue_kv_tile = [&]<bool FullTile>(int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kCausalPromptI8Threads) {
            const int key = tile_k0 + key_l;
            __half* kd    = &k_scale_s[key_l * Groups];
            __half* vd    = &v_scale_s[key_l * VGroups];
            if (FullTile || key <= max_query_abs) {
                const std::int64_t off =
                    kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
                if constexpr (PackedValues) {
                    const std::int64_t voff = kv_cache_int4_value_scale_index<Geometry>(
                        physical_page, kv_head, 0, key_l);
                    ninfer::ops::cp_async<16>(vd, &cache_v_scale[voff]);
                } else {
                    ninfer::ops::cp_async<8>(vd, &cache_v_scale[off]);
                }
            } else {
                store_vec(kd, make_int2(0, 0));
                if constexpr (PackedValues) {
                    store_vec(vd, make_int4(0, 0, 0, 0));
                } else {
                    store_vec(vd, make_int2(0, 0));
                }
            }
        }
        if constexpr (PackedKeys) {
            static_assert(PackedValues, "rk4v4 pairs packed keys with packed values");
#pragma unroll
            for (int i = 0; i < KChunksPerThread; ++i) {
                const int chunk = tid + i * kCausalPromptI8Threads;
                if (chunk < KChunks) {
                    const int key_l = chunk / (D / 16);
                    const int dc    = chunk - key_l * (D / 16);
                    const int d     = dc * 16;
                    const int key   = tile_k0 + key_l;
                    k_packed[i]     = make_uint2(0u, 0u);
                    if (FullTile || key <= max_query_abs) {
                        // Both planes are 128 bytes per row, so one offset addresses both.
                        const std::int64_t off = kv_cache_int4_value_code_index<Geometry>(
                            physical_page, kv_head, d >> 1, key_l);
                        k_packed[i] = *reinterpret_cast<const uint2*>(
                            reinterpret_cast<const std::uint8_t*>(cache_k) + off);
                        ninfer::ops::cp_async<8>(
                            &v_i8[key_l * D + (d >> 1)],
                            reinterpret_cast<const std::uint8_t*>(cache_v) + off);
                    } else {
                        store_vec(&v_i8[key_l * D + (d >> 1)], make_int2(0, 0));
                    }
                }
            }
        } else {
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * (D / 16); chunk += kCausalPromptI8Threads) {
                const int key_l = chunk / (D / 16);
                const int dc    = chunk - key_l * (D / 16);
                const int d     = dc * 16;
                const int key   = tile_k0 + key_l;
                std::int8_t* kd = &k_i8[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2];
                if (FullTile || key <= max_query_abs) {
                    const std::int64_t off =
                        kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d, key_l);
                    cp_async<16, Cache::cg>(kd, &cache_k[off]);
                    if constexpr (PackedValues) {
                        // Sixteen dimensions occupy eight packed bytes.
                        const std::int64_t voff = kv_cache_int4_value_code_index<Geometry>(
                            physical_page, kv_head, d >> 1, key_l);
                        // cp.async.cg is 16-byte only; the 8-byte form uses the default policy.
                        ninfer::ops::cp_async<8>(
                            &v_i8[key_l * D + (d >> 1)],
                            reinterpret_cast<const std::uint8_t*>(cache_v) + voff);
                    } else {
                        cp_async<16, Cache::cg>(&v_i8[key_l * D + d], &cache_v[off]);
                    }
                } else {
                    store_vec(kd, make_int4(0, 0, 0, 0));
                    if constexpr (PackedValues) {
                        store_vec(&v_i8[key_l * D + (d >> 1)], make_int2(0, 0));
                    } else {
                        store_vec(&v_i8[key_l * D + d], make_int4(0, 0, 0, 0));
                    }
                }
            }
        }
        ninfer::ops::cp_commit();
    };

    // Leading key blocks lie wholly at or below the first query row of a full query tile. They run a
    // FullTile body: no causal mask, no zero-selects in the softmax, and unconditional staging.
    // Only the blocks crossing the diagonal, and every block of a partial tail tile, keep the
    // masked path.
    const int full_blocks =
        q0 + Br <= tokens ? min(key_blocks, (base_pos + q0 + 1) / Bc) : 0;
    if (full_blocks > 0) {
        issue_kv_tile.template operator()<true>(0);
    } else {
        issue_kv_tile.template operator()<false>(0);
    }
    commit_k_tile();
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Keeping exactly two group scales live is the spill-free 120-register point on SM120.
    // Groups 2/3 reload per key tile; retaining all four creates an 8-byte stack frame.
    float q_scale_r0[Groups - 2];
    float q_scale_r1[Groups - 2];
    if (warp < ProducerWarps) {
        const int scale_row0 = warp * 16 + gid;
        const int scale_row1 = scale_row0 + 8;
#pragma unroll
        for (int grp = 0; grp < Groups - 2; ++grp) {
            float qs0       = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
            float qs1       = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
            q_scale_r0[grp] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[grp] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    const auto step      = [&]<bool FullTile>(int kb) {
        const int k0 = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < Groups - 2) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    const int scale_row0 = row_base + gid;
                    const int scale_row1 = scale_row0 + 8;
                    qs0                  = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
                    qs1                  = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
                    qs0                  = __shfl_sync(FullMask, qs0, gid * 4);
                    qs1                  = __shfl_sync(FullMask, qs1, gid * 4);
                }

                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = grp * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 causal_prompt_swz(row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = grp * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ks0      = 0.0f;
                    float ks1      = 0.0f;
                    if (gid == 0) {
                        ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                        ks1 = __half2float(k_scale_s[keyb * Groups + grp]);
                    }
                    ks0          = __shfl_sync(FullMask, ks0, lid);
                    ks1          = __shfl_sync(FullMask, ks1, lid);
                    score[nt][0] = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                    score[nt][1] = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                    score[nt][2] = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                    score[nt][3] = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
                }
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            float bm0 = -CUDART_INF_F;
            float bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if constexpr (!FullTile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                // A FullTile has no masked score, so no probability needs the zero-select.
                const float p00 = FullTile || score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = FullTile || score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = FullTile || score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = FullTile || score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col1)] = __float2half_rn(p11);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                if (FullTile || key <= max_query_abs) {
                    // dc == lane here, so a value group spans D/8/VGroups consecutive lanes:
                    // eight for the INT8 G64 coding and four for the packed G32 one. One lane per
                    // group loads the scale and broadcasts it to the rest of that group.
                    constexpr int VLanesPerGroup = PackedValues ? 4 : 8;
                    const int grp = PackedValues ? (d >> 5) : (d >> 6);
                    __half vs     = __float2half_rn(0.0f);
                    if ((lane & (VLanesPerGroup - 1)) == 0) {
                        vs = v_scale_s[key_l * VGroups + grp];
                    }
                    vs = __shfl_sync(FullMask, vs, lane & ~(VLanesPerGroup - 1));
                    const float vsf = __half2float(vs);
                    if constexpr (PackedValues) {
                        store_vec(dst, kv_cache_int4_dequant_f16x8_from(
                                           reinterpret_cast<const std::uint8_t*>(v_i8) +
                                               key_l * D + (d >> 1),
                                           vsf));
                    } else {
                        store_vec(dst,
                                  kv_cache_int8_dequant_f16x8_from(&v_i8[key_l * D + d], vsf));
                    }
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            if (kb + 1 < full_blocks) {
                issue_kv_tile.template operator()<true>((kb + 1) * Bc);
            } else {
                issue_kv_tile.template operator()<false>((kb + 1) * Bc);
            }
        }

        const int row_tile = warp % kCausalPromptI8RowTiles;
        const int d_slice  = warp / kCausalPromptI8RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       causal_prompt_p_swz<Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if (has_next) {
            commit_k_tile();
            ninfer::ops::cp_wait<0>();
        }
        __syncthreads();
    };
    for (int kb = 0; kb < full_blocks; ++kb) { step.template operator()<true>(kb); }
    for (int kb = full_blocks; kb < key_blocks; ++kb) { step.template operator()<false>(kb); }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kCausalPromptI8RowTiles;
    const int d_slice  = warp / kCausalPromptI8RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             kCausalPromptI8Threads);
}

} // namespace ninfer::ops
