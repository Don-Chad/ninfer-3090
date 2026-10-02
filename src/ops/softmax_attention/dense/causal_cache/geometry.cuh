#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue, int MaximumSplitsValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = MaximumSplitsValue * SmallTSplitScale;
};

// The 27B's four KV heads cap the small-T split count at 41: 4 x 41 = 164 partial CTAs, one full
// two-CTA-per-SM wave on the 3090's 82 SMs, where the shared 85 made 340 CTAs, 2.07 waves. Swept on
// sm_86 2026-10-02 (bench/ops/causal_softmax_attention_bench, rk4v4, four interleaved rounds,
// median us, cap 85 vs 41): T=1 16K 77 vs 65, 128K 447 vs 344; T=4 16K 115 vs 86, 128K 602 vs
// 468; T=8 32K 234 vs 182, 64K 350 vs 310, 128K 781 vs 572, 176K 1030 vs 773. T=8 at 16K is
// unchanged (120). Caps 82, 123 and 164 were each slower than 41 somewhere in 32K..128K.
using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1, 41>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2, 85>;

} // namespace ninfer::ops
