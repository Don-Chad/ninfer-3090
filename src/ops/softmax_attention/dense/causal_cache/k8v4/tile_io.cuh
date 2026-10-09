#pragma once

#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"

namespace ninfer::ops::detail {
struct K8V4KvTiledValues {
    using Scale                      = std::uint8_t;
    static constexpr int kCodeBytes  = 128;
    static constexpr int kScaleItems = 16;

    __device__ __forceinline__ static int4 expand(const std::uint8_t* codes, Scale scale) {
        return kv_cache_nvfp4_dequant_f16x8(codes, scale);
    }
};
} // namespace ninfer::ops::detail
