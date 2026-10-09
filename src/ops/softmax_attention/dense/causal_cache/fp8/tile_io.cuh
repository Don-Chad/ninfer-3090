#pragma once
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"

namespace ninfer::ops::detail {

__device__ __forceinline__ int4 fp8_kv_dequant_f16x8(const std::uint8_t* codes, __half scale) {
    const int2 raw         = load_vec<int2>(codes);
    const std::uint16_t* c = reinterpret_cast<const std::uint16_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 value2 = kv_cache_fp8_dequant_code2_to_half2(c[i], scale);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

struct Fp8KvTiledValues {
    using Scale                      = __half;
    static constexpr int kCodeBytes  = 256;
    static constexpr int kScaleItems = 1;

    __device__ __forceinline__ static int4 expand(const std::uint8_t* codes, Scale scale) {
        return fp8_kv_dequant_f16x8(codes, scale);
    }
};

} // namespace ninfer::ops::detail
