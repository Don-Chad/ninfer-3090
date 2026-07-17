#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <string>

namespace ninfer::targets::qwen3_6 {

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder) {
    using artifact::NumericFormat;

    VisionBackbonePlan out;
    out.patch_embedding = artifact::bind_device_tensor(
        binder, "vision/patch_embedding", NumericFormat::Q6G64_F16S,
        {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias = artifact::bind_device_tensor(
        binder, "vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding = artifact::bind_device_tensor(
        binder, "vision/position_embedding", NumericFormat::BF16,
        {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = artifact::bind_device_tensor(
            binder, prefix + "attention/qkv", NumericFormat::Q4G64_F16S,
            {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias =
            artifact::bind_device_tensor(binder, prefix + "attention/qkv_bias", NumericFormat::BF16,
                                         {3 * VisionBackboneConfig::hidden});
        target.output = artifact::bind_device_tensor(
            binder, prefix + "attention/output", NumericFormat::Q5G64_F16S,
            {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias =
            artifact::bind_device_tensor(binder, prefix + "attention/output_bias",
                                         NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.fc1 = artifact::bind_device_tensor(
            binder, prefix + "mlp/fc1", NumericFormat::Q4G64_F16S,
            {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias =
            artifact::bind_device_tensor(binder, prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                         {VisionBackboneConfig::intermediate});
        target.fc2 = artifact::bind_device_tensor(
            binder, prefix + "mlp/fc2", NumericFormat::Q5G64_F16S,
            {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias = artifact::bind_device_tensor(
            binder, prefix + "mlp/fc2_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight = artifact::bind_device_tensor(
            binder, prefix + "norm1/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias = artifact::bind_device_tensor(
            binder, prefix + "norm1/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight = artifact::bind_device_tensor(
            binder, prefix + "norm2/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias = artifact::bind_device_tensor(
            binder, prefix + "norm2/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder) {
    using artifact::NumericFormat;
    return VisionMergerInputPlan{
        .fc1 = artifact::bind_device_tensor(
            binder, "vision/merger/fc1", NumericFormat::W8G32_F16S,
            {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias =
            artifact::bind_device_tensor(binder, "vision/merger/fc1_bias", NumericFormat::BF16,
                                         {VisionBackboneConfig::merger_hidden}),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder) {
    using artifact::NumericFormat;
    return VisionMergerNormPlan{
        .weight = artifact::bind_device_tensor(binder, "vision/merger/norm/weight",
                                               NumericFormat::BF16, {VisionBackboneConfig::hidden}),
        .bias = artifact::bind_device_tensor(binder, "vision/merger/norm/bias", NumericFormat::BF16,
                                             {VisionBackboneConfig::hidden}),
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
