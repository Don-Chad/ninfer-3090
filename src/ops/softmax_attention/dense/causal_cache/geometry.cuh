#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue, int MaximumSplitsValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = MaximumSplitsValue * SmallTSplitScale;
};

// Both geometries cap the small-T grid at 164 partial CTAs, one full two-CTA-per-SM wave on the
// 3090's 82 SMs: the 27B's four KV heads at 41 splits, the 35B-A3B's two at 82 (41 x its split
// scale). The shared 85 made 340 CTAs, 2.07 waves. Swept on sm_86 2026-10-02
// (bench/ops/causal_softmax_attention_bench, rk4v4, four interleaved rounds, median us, old cap vs
// new):
//   27B  T=1 16K 77 vs 65, 128K 447 vs 344; T=4 16K 115 vs 86, 128K 602 vs 468;
//        T=8 32K 234 vs 182, 64K 350 vs 310, 128K 781 vs 572, 176K 1030 vs 773 (16K unchanged)
//   35B  T=1 16K 57 vs 43, 224K 421 vs 302; T=4 16K 106 vs 68, 64K 205 vs 154, 224K 658 vs 447;
//        T=8 16K 187 vs 136, 128K 603 vs 535, 224K 1221 vs 830
// For the 27B, caps of 82, 123 and 164 were each slower than 41 somewhere in 32K..128K.
using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1, 41>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2, 41>;

} // namespace ninfer::ops
