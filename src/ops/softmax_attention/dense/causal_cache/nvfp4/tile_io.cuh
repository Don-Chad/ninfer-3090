#pragma once
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

namespace ninfer::ops::detail {

__device__ __forceinline__ int nvfp4_kv_code_swizzle(int row, int byte) {
    return (((byte >> 4) ^ (row & 7)) << 4) | (byte & 15);
}

} // namespace ninfer::ops::detail
