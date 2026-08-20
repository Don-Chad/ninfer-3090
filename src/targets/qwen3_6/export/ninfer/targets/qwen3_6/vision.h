#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>

namespace ninfer {
class EvictableWeightPool;
}

namespace ninfer::artifact {
class MaterializedArtifact;
struct MaterializationPlan;
}

namespace ninfer::targets::qwen3_6 {

struct VisionBackboneConfig {
    static constexpr int layers              = 27;
    static constexpr int hidden              = 1152;
    static constexpr int intermediate        = 4304;
    static constexpr int heads               = 16;
    static constexpr int head_dim            = hidden / heads;
    static constexpr int patch_dim           = 3 * 2 * 16 * 16;
    static constexpr int merge               = 2;
    static constexpr int merge_unit          = merge * merge;
    static constexpr int merger_hidden       = hidden * merge_unit;
    static constexpr int position_embeddings = 48 * 48;
    static constexpr int rotary_dim          = head_dim;
    static constexpr float rope_theta        = 10'000.0F;
    static constexpr float norm_epsilon      = 1.0e-6F;
};

struct VisionLayerPlan {
    artifact::ObjectHandle qkv;
    artifact::ObjectHandle qkv_bias;
    artifact::ObjectHandle output;
    artifact::ObjectHandle output_bias;
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
    artifact::ObjectHandle fc2;
    artifact::ObjectHandle fc2_bias;
    artifact::ObjectHandle norm1_weight;
    artifact::ObjectHandle norm1_bias;
    artifact::ObjectHandle norm2_weight;
    artifact::ObjectHandle norm2_bias;
};

struct VisionBackbonePlan {
    artifact::ObjectHandle patch_embedding;
    artifact::ObjectHandle patch_embedding_bias;
    artifact::ObjectHandle position_embedding;
    std::array<VisionLayerPlan, VisionBackboneConfig::layers> layers;
};

struct VisionMergerInputPlan {
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
};

struct VisionMergerNormPlan {
    artifact::ObjectHandle weight;
    artifact::ObjectHandle bias;
};

struct VisionLayerWeights {
    Weight qkv;
    Tensor qkv_bias;
    Weight output;
    Tensor output_bias;
    Weight fc1;
    Tensor fc1_bias;
    Weight fc2;
    Tensor fc2_bias;
    Tensor norm1_weight;
    Tensor norm1_bias;
    Tensor norm2_weight;
    Tensor norm2_bias;
};

struct VisionCommonWeights {
    Weight patch_embedding;
    Tensor patch_embedding_bias;
    Tensor position_embedding;
    std::array<VisionLayerWeights, VisionBackboneConfig::layers> layers;
    Weight merger_fc1;
    Tensor merger_fc1_bias;
    Tensor merger_norm_weight;
    Tensor merger_norm_bias;
};

struct VisionWeights {
    VisionCommonWeights common;
    Weight merger_fc2;
    Tensor merger_fc2_bias;
};

// Byte ranges of the vision groups inside the pinned weight block, used by the
// overlay window to stage weights through borrowed device memory. Ranges are
// contiguous by binding order; slot_bytes is the largest single layer range.
struct VisionOverlayLayout {
    std::size_t prelude_begin = 0;   // patch embedding .. position embedding
    std::size_t prelude_bytes = 0;
    std::array<std::size_t, VisionBackboneConfig::layers> layer_begin{};
    std::array<std::size_t, VisionBackboneConfig::layers> layer_bytes{};
    std::size_t merger_begin = 0;    // merger fc1 .. merger norm bias
    std::size_t merger_bytes = 0;
    std::size_t slot_bytes    = 0;
    std::size_t staging_bytes = 0;   // prelude + merger + two layer slots, aligned
};

// Runtime assets the overlay window needs, published on the model view when the
// engine runs with VisionResidency::Overlay.
struct VisionOverlayAssets {
    ninfer::EvictableWeightPool* pool  = nullptr;
    const std::byte* pinned_block      = nullptr;
    std::size_t pinned_bytes           = 0;
    std::size_t ladder_bytes           = 0;   // evictable tail a window may borrow
    VisionOverlayLayout layout;
};

[[nodiscard]] VisionOverlayLayout compute_vision_overlay_layout(
    const VisionBackbonePlan& backbone, const VisionMergerInputPlan& merger_input,
    artifact::ObjectHandle merger_fc2, artifact::ObjectHandle merger_fc2_bias,
    const VisionMergerNormPlan& merger_norm, const artifact::MaterializationPlan& plan);

[[nodiscard]] VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                                      artifact::TensorPlacement placement);
[[nodiscard]] VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                                             artifact::TensorPlacement placement);
[[nodiscard]] VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                                           artifact::TensorPlacement placement);

[[nodiscard]] VisionCommonWeights materialize_vision_common(
    const artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionMergerInputPlan& merger_input, const VisionMergerNormPlan& merger_norm);

} // namespace ninfer::targets::qwen3_6
