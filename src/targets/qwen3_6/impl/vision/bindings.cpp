#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

struct PinnedRangeIndex {
    std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>> by_object;

    explicit PinnedRangeIndex(const artifact::MaterializationPlan& plan) {
        by_object.reserve(plan.pinned_objects.size());
        for (const artifact::PinnedMaterialization& placement : plan.pinned_objects) {
            by_object.emplace(placement.object.index,
                              std::make_pair(static_cast<std::size_t>(placement.offset),
                                             static_cast<std::size_t>(placement.bytes)));
        }
    }

    // Contiguous extent covering every listed object; throws when the objects
    // are not pinned or leave a gap larger than alignment padding.
    template <class Handles>
    std::pair<std::size_t, std::size_t> extent(const Handles& handles) const {
        std::size_t begin = SIZE_MAX;
        std::size_t end   = 0;
        std::size_t sum   = 0;
        for (const artifact::ObjectHandle handle : handles) {
            const auto it = by_object.find(handle.index);
            if (it == by_object.end()) {
                throw std::logic_error("vision overlay group tensor is not pinned");
            }
            begin = std::min(begin, it->second.first);
            end   = std::max(end, it->second.first + it->second.second);
            sum += it->second.second;
        }
        if (begin == SIZE_MAX || end <= begin || end - begin > sum + handles.size() * 4096) {
            throw std::logic_error("vision overlay group is not contiguous in the pinned block");
        }
        return {begin, end - begin};
    }
};

constexpr std::size_t kStagingAlignment = 256;

std::size_t staging_align(std::size_t bytes) {
    return (bytes + kStagingAlignment - 1) / kStagingAlignment * kStagingAlignment;
}

} // namespace

VisionOverlayLayout compute_vision_overlay_layout(const VisionBackbonePlan& backbone,
                                                  const VisionMergerInputPlan& merger_input,
                                                  artifact::ObjectHandle merger_fc2,
                                                  artifact::ObjectHandle merger_fc2_bias,
                                                  const VisionMergerNormPlan& merger_norm,
                                                  const artifact::MaterializationPlan& plan) {
    const PinnedRangeIndex index(plan);
    VisionOverlayLayout out;

    const std::array<artifact::ObjectHandle, 3> prelude = {
        backbone.patch_embedding, backbone.patch_embedding_bias, backbone.position_embedding};
    std::tie(out.prelude_begin, out.prelude_bytes) = index.extent(prelude);

    for (std::size_t layer = 0; layer < backbone.layers.size(); ++layer) {
        const VisionLayerPlan& source                        = backbone.layers[layer];
        const std::array<artifact::ObjectHandle, 12> handles = {
            source.qkv,      source.qkv_bias,     source.output,       source.output_bias,
            source.fc1,      source.fc1_bias,     source.fc2,          source.fc2_bias,
            source.norm1_weight, source.norm1_bias, source.norm2_weight, source.norm2_bias};
        std::tie(out.layer_begin[layer], out.layer_bytes[layer]) = index.extent(handles);
        out.slot_bytes = std::max(out.slot_bytes, out.layer_bytes[layer]);
    }

    const std::array<artifact::ObjectHandle, 6> merger = {
        merger_input.fc1, merger_input.fc1_bias, merger_fc2,
        merger_fc2_bias,  merger_norm.weight,    merger_norm.bias};
    std::tie(out.merger_begin, out.merger_bytes) = index.extent(merger);

    out.staging_bytes = staging_align(out.prelude_bytes) + staging_align(out.merger_bytes) +
                        2 * staging_align(out.slot_bytes);
    return out;
}

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    out.patch_embedding = bind("vision/patch_embedding", NumericFormat::Q6G64_F16S,
                               {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", NumericFormat::Q4G64_F16S,
                                        {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * VisionBackboneConfig::hidden});
        target.output            = bind(prefix + "attention/output", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", NumericFormat::Q4G64_F16S,
                                        {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                               artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlan{
        .fc1      = bind("vision/merger/fc1", NumericFormat::W8G32_F16S,
                         {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {VisionBackboneConfig::merger_hidden}),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                             artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
    };
}

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm) {
    using artifact::NumericFormat;

    VisionCommonWeights out;
    out.patch_embedding = artifact::materialized_weight(
        materialized, backbone.patch_embedding, NumericFormat::Q6G64_F16S,
        VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim);
    out.patch_embedding_bias =
        artifact::materialized_tensor(materialized, backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding = artifact::materialized_tensor(
        materialized, backbone.position_embedding, NumericFormat::BF16,
        {VisionBackboneConfig::hidden, VisionBackboneConfig::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv                    = artifact::materialized_weight(
            materialized, source.qkv, NumericFormat::Q4G64_F16S, 3 * VisionBackboneConfig::hidden,
            VisionBackboneConfig::hidden);
        target.qkv_bias = artifact::materialized_tensor(
            materialized, source.qkv_bias, NumericFormat::BF16, {3 * VisionBackboneConfig::hidden});
        target.output = artifact::materialized_weight(
            materialized, source.output, NumericFormat::Q5G64_F16S, VisionBackboneConfig::hidden,
            VisionBackboneConfig::hidden);
        target.output_bias = artifact::materialized_tensor(
            materialized, source.output_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.fc1 = artifact::materialized_weight(
            materialized, source.fc1, NumericFormat::Q4G64_F16S, VisionBackboneConfig::intermediate,
            VisionBackboneConfig::hidden);
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {VisionBackboneConfig::intermediate});
        target.fc2 = artifact::materialized_weight(
            materialized, source.fc2, NumericFormat::Q5G64_F16S, VisionBackboneConfig::hidden,
            VisionBackboneConfig::intermediate);
        target.fc2_bias = artifact::materialized_tensor(
            materialized, source.fc2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight = artifact::materialized_tensor(
            materialized, source.norm1_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias = artifact::materialized_tensor(
            materialized, source.norm1_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight = artifact::materialized_tensor(
            materialized, source.norm2_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias = artifact::materialized_tensor(
            materialized, source.norm2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }

    out.merger_fc1 = artifact::materialized_weight(
        materialized, merger_input.fc1, NumericFormat::W8G32_F16S,
        VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden);
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {VisionBackboneConfig::merger_hidden});
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    return out;
}

} // namespace ninfer::targets::qwen3_6
