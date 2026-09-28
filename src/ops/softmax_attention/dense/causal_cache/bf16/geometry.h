#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {

template <int HeadDim, int QueryHeads, int KVHeads>
struct Bf16KvGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using Bf16KvD256H24Kv4 = Bf16KvGeometry<256, 24, 4>;
using Bf16KvD256H16Kv2 = Bf16KvGeometry<256, 16, 2>;

} // namespace ninfer::ops::detail
