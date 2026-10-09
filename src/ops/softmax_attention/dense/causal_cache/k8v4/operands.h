#pragma once
#include "ops/softmax_attention/common/causal_operands.h"

namespace ninfer::ops::detail {
template <bool Writable>
using K8V4KvCacheView = QuantizedCausalCacheView<std::uint8_t, __half, std::uint8_t, Writable>;
using K8V4KvReadView  = K8V4KvCacheView<false>;

} // namespace ninfer::ops::detail
