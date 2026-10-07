#pragma once

// Signed int8, per-token G64 KV-cache codec shared by append and causal-attention kernels.
// This header owns index math, vectorized decode, and scalar encode; there is deliberately no
// standalone transcode kernel in the production path.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheInt8HeadDim = 256;
inline constexpr int kKVCacheInt8Group   = 64;
inline constexpr int kKVCacheInt8Groups  = kKVCacheInt8HeadDim / kKVCacheInt8Group;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_code_index(int physical_page, int kv_head, int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int8_quant_src_index(int kv_head, int d,
                                                                      int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheInt8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheInt8QuantParams {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary shared by standalone and fused append. The stored scale is
// FP16-RNE(absmax/127); codes always use the reciprocal of that represented FP16 value.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int8_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int8_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// --- packed signed int4 value codec (rk8v4) ---------------------------------------------------
// The rk8v4 profile keeps K as rotated INT8 and stores V as two signed 4-bit codes per byte, so a
// V plane is half as wide as its INT8 counterpart while the G64 scale plane is unchanged. The
// group encoding mirrors the INT8 one with 7 in place of 127, which is the largest magnitude a
// signed 4-bit code can represent.
inline constexpr int kKVCacheInt4Max = 7;

// Values use a finer group than keys. Four-bit codes resolve a group to 15 levels, so the group
// absmax sets the step for every member and one outlier costs the whole group precision. Halving
// the group to 32 halves the number of values a single outlier can degrade, at the price of four
// extra FP16 scales per row.
inline constexpr int kKVCacheInt4ValueGroup  = 32;
inline constexpr int kKVCacheInt4ValueGroups = kKVCacheInt8HeadDim / kKVCacheInt4ValueGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int4_value_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt4ValueGroups, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int4_value_code_index(int physical_page, int kv_head, int packed_d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim / 2, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, packed_d);
}

// Persistent group-scale boundary for packed int4 values: FP16-RNE(absmax/7). Codes always use
// the reciprocal of that represented FP16 value, exactly as the INT8 codec does.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int4_quant_params(float absmax) {
    const __half scale = __float2half_rn(
        absmax > 0.0f ? absmax / static_cast<float>(kKVCacheInt4Max) : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int4_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-kKVCacheInt4Max, min(kKVCacheInt4Max, q));
    return static_cast<std::int8_t>(q);
}

// Byte layout: dimension d occupies the low nibble when d is even and the high nibble when odd,
// so a packed byte holds the adjacent pair (2p, 2p+1).
__device__ __forceinline__ std::uint8_t kv_cache_int4_pack(std::int8_t low, std::int8_t high) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(low) & 0x0fu) |
                                     ((static_cast<unsigned>(high) & 0x0fu) << 4));
}

__device__ __forceinline__ std::int8_t kv_cache_int4_unpack(std::uint8_t packed, int high) {
    const unsigned nibble = high != 0 ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// --- rotated Lloyd-Max 4-bit key codec (rk4v4) ------------------------------------------------
// The rk4v4 profile stores each rotated key dimension as a 4-bit index into the symmetric
// 16-level Lloyd-Max quantizer for N(0,1), packed two per byte exactly like the rk8v4 value
// plane (low nibble = even dimension). Index = sign << 3 | magnitude, magnitude 0..7.
//
// Each level is represented by a fixed INT8 code, round(L_m / L_7 * 127), and the G64 scale plane
// keeps one FP16 scale per 64 dimensions, exactly as the INT8 key coding does. Expanding a nibble
// to its code therefore reproduces an ordinary rotated-INT8 key, and every INT8-family attention
// kernel consumes it through its unchanged m16n8k32.s8 QK path. The encoder picks each index
// against the group's RMS and then takes the least-squares scale for the chosen codes.
inline constexpr int kKVCacheLloyd4Levels = 8;

// Level boundaries in units of the group RMS: midpoints of the Lloyd-Max reconstruction points
// {0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326}.
__device__ __forceinline__ int kv_cache_lloyd4_magnitude(float z) {
    const float a = fabsf(z);
    int m         = 0;
    m += a >= 0.25825f;
    m += a >= 0.52245f;
    m += a >= 0.79960f;
    m += a >= 1.09930f;
    m += a >= 1.43715f;
    m += a >= 1.84355f;
    m += a >= 2.40080f;
    return m;
}

// The INT8 code of each magnitude, packed four per word for the byte-permute expander.
inline constexpr unsigned kKVCacheLloyd4PosLo = 0x2C1F1206u; // 6, 18, 31, 44
inline constexpr unsigned kKVCacheLloyd4PosHi = 0x7F604B3Au; // 58, 75, 96, 127
inline constexpr unsigned kKVCacheLloyd4NegLo = 0xD4E1EEFAu; // -6, -18, -31, -44
inline constexpr unsigned kKVCacheLloyd4NegHi = 0x81A0B5C6u; // -58, -75, -96, -127

__host__ __device__ constexpr int kv_cache_lloyd4_code(int index) {
    constexpr int codes[kKVCacheLloyd4Levels] = {6, 18, 31, 44, 58, 75, 96, 127};
    return (index & 8) != 0 ? -codes[index & 7] : codes[index & 7];
}

// Expand four nibbles (the low 16 bits of `nibbles`, dimension order) into four INT8 codes.
// prmt reads its selector's bit 3 as sign-replicate, so the magnitude selects from a positive and
// a negative table and the sign bit, spread to a byte mask by a third prmt, picks between them.
__device__ __forceinline__ unsigned kv_cache_lloyd4_expand4(unsigned nibbles) {
    const unsigned magnitude = nibbles & 0x7777u;
    const unsigned pos       = __byte_perm(kKVCacheLloyd4PosLo, kKVCacheLloyd4PosHi, magnitude);
    const unsigned neg       = __byte_perm(kKVCacheLloyd4NegLo, kKVCacheLloyd4NegHi, magnitude);
    const unsigned mask      = __byte_perm(0u, 0xFF00u, ((nibbles >> 3) & 0x1111u) | 0x4444u);
    return (neg & mask) | (pos & ~mask);
}

// Sixteen consecutive dimensions: eight packed bytes in, sixteen INT8 codes out.
__device__ __forceinline__ int4 kv_cache_lloyd4_expand16(uint2 packed) {
    return make_int4(static_cast<int>(kv_cache_lloyd4_expand4(packed.x)),
                     static_cast<int>(kv_cache_lloyd4_expand4(packed.x >> 16)),
                     static_cast<int>(kv_cache_lloyd4_expand4(packed.y)),
                     static_cast<int>(kv_cache_lloyd4_expand4(packed.y >> 16)));
}

struct KVCacheLloyd4Encoded {
    int index0;
    int index1;
    __half scale;
};

// Encode one G64 group held by a full warp, two dimensions per lane (the append kernels' d0/d1
// lane assignment). Every lane receives its two indices and the group's represented scale.
//
// Every operation is spelled with an explicit rounding intrinsic and the reductions are xor
// butterflies, so the result is a fixed function of the 64 inputs that a host oracle reproduces
// exactly: nvcc would otherwise be free to contract the products into FMAs.
__device__ __forceinline__ KVCacheLloyd4Encoded kv_cache_lloyd4_encode_group(float x0, float x1) {
    constexpr unsigned FullMask = 0xffffffffu;
    float sum_sq                = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(FullMask, sum_sq, offset));
    }
    const float rms     = __fsqrt_rn(__fmul_rn(sum_sq, 1.0f / 64.0f));
    const float inv_rms = rms > 0.0f ? __fdiv_rn(1.0f, rms) : 0.0f;
    const int index0    = (x0 < 0.0f ? 8 : 0) | kv_cache_lloyd4_magnitude(__fmul_rn(x0, inv_rms));
    const int index1    = (x1 < 0.0f ? 8 : 0) | kv_cache_lloyd4_magnitude(__fmul_rn(x1, inv_rms));
    const float c0      = static_cast<float>(kv_cache_lloyd4_code(index0));
    const float c1      = static_cast<float>(kv_cache_lloyd4_code(index1));
    float xc            = __fadd_rn(__fmul_rn(x0, c0), __fmul_rn(x1, c1));
    float cc            = __fadd_rn(__fmul_rn(c0, c0), __fmul_rn(c1, c1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        xc = __fadd_rn(xc, __shfl_xor_sync(FullMask, xc, offset));
        cc = __fadd_rn(cc, __shfl_xor_sync(FullMask, cc, offset));
    }
    return {index0, index1, __float2half_rn(rms > 0.0f ? __fdiv_rn(xc, cc) : 0.0f)};
}

// FP16 loaders for the INT4 and INT8 planes.
//
// A kernel that stages V for an f16 PV MMA needs f16 codes, and a bf16 loader cannot serve it:
// both plane types are sixteen bits wide, so storing a bf16 result through a __half* compiles,
// runs, and silently reinterprets every value. That is not hypothetical -- it is what upstream's
// int8 small-T kernel does today, and it is why the whole INT8 family (int8-g64 and rk8v4 alike,
// since both go through these helpers) comes out 3-11x the reference with a ratio that varies
// per value. Nothing else in the pipeline can catch it: the sizes match, so no allocation, no
// dtype validation and no compiler diagnostic fires.
//
// int8 codes carry at most eight significant bits and a scale, so f16's narrower exponent is not a
// constraint here while its wider mantissa is a small gain on the P*V product.
__device__ __forceinline__ int4 kv_cache_int4_dequant_f16x8_from(const std::uint8_t* packed4,
                                                                 float s) {
    const unsigned raw        = load_vec<unsigned>(packed4);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
    unsigned out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(kv_cache_int4_unpack(bytes[i], 0)) * s;
        const float x1 = static_cast<float>(kv_cache_int4_unpack(bytes[i], 1)) * s;
        out[i]         = pack_f16x2(x0, x1);
    }
    return make_int4(static_cast<int>(out[0]), static_cast<int>(out[1]), static_cast<int>(out[2]),
                     static_cast<int>(out[3]));
}

__device__ __forceinline__ int4 kv_cache_int8_dequant_f16x8_from(const std::int8_t* codes8,
                                                                 float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_f16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Write one rotated G64 key group held by a full warp (lane owns d0 = 64g+lane and d1 = d0+32),
// shared by the standalone append kernels and the small-T kernel's fused append. PackedKeys
// selects the rk4v4 Lloyd-Max coding: two 4-bit indices per byte, low nibble = even dimension, so
// a byte spans lanes l and l^1 exactly as the packed value plane does. Otherwise the key is INT8
// G64. Both keep one FP16 scale per group in the same scale plane.
template <typename Geometry, bool PackedKeys>
__device__ __forceinline__ void
kv_cache_i8_family_store_key_group(std::int8_t* cache_k, __half* scale_k, int page, int kv_head,
                                   int group, int page_off, int lane, float k0, float k1) {
    constexpr unsigned FullMask = 0xffffffffu;
    __half scale;
    if constexpr (PackedKeys) {
        const auto encoded = kv_cache_lloyd4_encode_group(k0, k1);
        const int partner0 = __shfl_xor_sync(FullMask, encoded.index0, 1);
        const int partner1 = __shfl_xor_sync(FullMask, encoded.index1, 1);
        if ((lane & 1) == 0) {
            auto* packed_k                 = reinterpret_cast<std::uint8_t*>(cache_k);
            const std::int64_t packed_base = kv_cache_int4_value_code_index<Geometry>(
                page, kv_head, group * (kKVCacheInt8Group / 2), page_off);
            packed_k[packed_base + (lane >> 1)] =
                static_cast<std::uint8_t>(encoded.index0 | (partner0 << 4));
            packed_k[packed_base + (lane >> 1) + 16] =
                static_cast<std::uint8_t>(encoded.index1 | (partner1 << 4));
        }
        scale = encoded.scale;
    } else {
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        scale                          = k_quant.scale;
    }
    if (lane == 0) {
        scale_k[kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off)] = scale;
    }
}

} // namespace ninfer::ops
