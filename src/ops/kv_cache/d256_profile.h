#pragma once

#include "core/dtype.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

// K and V are described separately because the rk8v4 profile pairs a rotated INT8 key plane with
// a packed signed int4 value plane. Every other profile stores both planes identically, so their
// key and value fields agree.
//
// A value plane of DType::U8 denotes two signed 4-bit codes per byte, low nibble first, which is
// why its leading extent is half the head dimension. The G64 scale planes are shared unchanged.
struct D256KVCacheProfile {
    DType key_code_dtype;
    DType value_code_dtype;
    std::int32_t value_leading_extent;
    std::int32_t quant_group;
    std::int32_t scale_leading_extent;
    // Values may use a finer group than keys, so their scale plane can be wider.
    std::int32_t value_scale_leading_extent;

    [[nodiscard]] bool packed_int4_values() const {
        return value_code_dtype == DType::U8;
    }
};

// value_code_dtype selects between the INT8 and packed int4 value codings inside the INT8 family;
// callers pass the value plane's own dtype. Other families ignore it, because their value coding
// is implied by the family.
inline D256KVCacheProfile d256_kv_cache_profile(DType dtype, DType value_code_dtype) {
    switch (dtype) {
    case DType::BF16:
        return {DType::BF16, DType::BF16, kD256KVCacheHeadDim, 0, 0, 0};
    case DType::I8:
        if (value_code_dtype == DType::U8) {
            return {DType::I8, DType::U8, kD256KVCacheHeadDim / 2, 64, 4, 8};
        }
        return {DType::I8, DType::I8, kD256KVCacheHeadDim, 64, 4, 4};
    case DType::FP8_E4M3FN:
        return {DType::FP8_E4M3FN, DType::FP8_E4M3FN, kD256KVCacheHeadDim, 256, 1, 1};
    default:
        throw std::invalid_argument("unsupported D256 KV-cache dtype");
    }
}

// Overload for call sites that describe a cache whose two planes share one coding.
inline D256KVCacheProfile d256_kv_cache_profile(DType dtype) {
    return d256_kv_cache_profile(dtype, dtype);
}

} // namespace ninfer::ops
