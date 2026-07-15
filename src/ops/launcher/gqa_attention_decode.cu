// ninfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Split-KV grid sizing keyed on the actual attention window (kv.pos + tokens),
// not the allocation ceiling. Over-splitting inflates the partial scratch that
// the reduce kernel must stream back, so the count must track the live context.
// Supplies an upper bound for the device-side default active-split tiers. The
// dtype-aware wrapper below replaces it only for measured int8 specializations.
template <typename Geometry>
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::DecodeSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::DecodeSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::DecodeSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::DecodeSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::DecodeSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::DecodeSplitScale); }

    return (splits < Geometry::DecodeSplits) ? splits : Geometry::DecodeSplits;
}

template <typename Geometry>
std::int32_t gqa_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    // A 64-key default split just above a 32-key boundary makes the partial
    // kernel execute a nearly empty second tile. These short ranges instead
    // launch one 32-key tile per split; the larger CTAs keep the small grid busy.
    if (kv_dtype == DType::I8 && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::DecodeSplitScale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 128 && window <= 160) {
        return div_up(window, 24 / Geometry::DecodeSplitScale);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (kv_dtype == DType::I8 && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::DecodeSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return gqa_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry, int TokenTile, int WarpsPerCta>
void launch_tc_partial_bf16(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                            float scale, KVCache& kv, int layer, std::int32_t padded_context,
                            std::int32_t max_context, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const int tokens     = q.ne[2];
    const dim3 grid(Geometry::KVHeads, splits, 1);
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];
    // bf16 kernel uses only static smem (no dynamic staging).
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
            static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
            tokens, padded_context, max_context, scale,
            static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile>
void launch_tc_partial_i8(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                          float scale, KVCache& kv, int layer, std::int32_t padded_context,
                          std::int32_t max_context, std::int32_t window, std::int32_t splits,
                          Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                          cudaStream_t stream) {
    Tensor& cache_k       = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v       = kv.v[static_cast<std::uint32_t>(layer)];
    Tensor& cache_k_scale = kv.k_scale[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v_scale = kv.v_scale[static_cast<std::uint32_t>(layer)];
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, 1);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kGqaHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                             KeyBlock, DynamicArena>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data), padded_context, max_context, scale,
                static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
                static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (window > 128 && window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (window > 128 && window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (window > 128 && window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_decode_splits(std::int32_t q_heads, std::int32_t kv_heads) {
    if (q_heads == Gqa27Geometry::QHeads && kv_heads == Gqa27Geometry::KVHeads) {
        return Gqa27Geometry::DecodeSplits;
    }
    if (q_heads == Gqa35Geometry::QHeads && kv_heads == Gqa35Geometry::KVHeads) {
        return Gqa35Geometry::DecodeSplits;
    }
    throw std::invalid_argument("gqa_attention_decode_splits: unsupported head geometry");
}

template <typename Geometry>
void gqa_attention_small_t_launch_for(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& pos, float scale, KVCache& kv, int layer,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      Tensor& out, cudaStream_t stream) {
    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    const auto max_context    = static_cast<std::int32_t>(kv.max_context);
    // Split count tracks the live attention window (kv.pos is the pre-round base
    // position; positions run [base, base+T)), so decode/verify at a short context
    // inside a large allocation is not over-split. The kernel still receives the
    // real max_context for cache-bounds checks.
    const auto window = static_cast<std::int32_t>(kv.pos) + q.ne[2];
    const auto splits = gqa_small_t_split_count<Geometry>(window, q.ne[2], kv.dtype);

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define NINFER_GQA_SMALL_T_DISPATCH(TOKENS, WARPS)                                                 \
    do {                                                                                           \
        if (kv.dtype == DType::I8) {                                                               \
            launch_tc_partial_i8<Geometry, (TOKENS)>(q, k, v, pos, scale, kv, layer,               \
                                                     padded_context, max_context, window, splits,  \
                                                     partial_acc, partial_m, partial_l, stream);   \
        } else {                                                                                   \
            launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS)>(                                   \
                q, k, v, pos, scale, kv, layer, padded_context, max_context, splits, partial_acc,  \
                partial_m, partial_l, stream);                                                     \
        }                                                                                          \
    } while (0)

    switch (q.ne[2]) {
    case 1:
        NINFER_GQA_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        NINFER_GQA_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        NINFER_GQA_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        NINFER_GQA_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        NINFER_GQA_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        NINFER_GQA_SMALL_T_DISPATCH(6, 4);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef NINFER_GQA_SMALL_T_DISPATCH

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(kGqaHeadDim, kDChunk), q.ne[2]);
    gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk>
        <<<reduce_grid, kReduceBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(partial_acc.data),
            static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
            static_cast<const std::int32_t*>(pos.data), q.ne[2], splits,
            static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, float scale, KVCache& kv, int layer,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, k, v, pos, scale, kv, layer, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, k, v, pos, scale, kv, layer, partial_acc,
                                                    partial_m, partial_l, out, stream);
}

void gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& positions, float scale, KVCache& kv, int layer,
                          Tensor* partial_acc, Tensor* partial_m, Tensor* partial_l, Tensor& out,
                          cudaStream_t stream) {
    if (gqa_attention_uses_small_t(q.ne[2])) {
        if (partial_acc == nullptr || partial_m == nullptr || partial_l == nullptr) {
            throw std::invalid_argument("gqa_attention: small-T route requires workspace");
        }
        gqa_attention_small_t_launch(q, k, v, positions, scale, kv, layer, *partial_acc, *partial_m,
                                     *partial_l, out, stream);
        return;
    }
    gqa_attention_prompt_launch(q, k, v, positions, scale, kv, layer, out, stream);
}

} // namespace ninfer::ops::detail
