#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "runtime/contract/transient_region.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

struct VisionItemView {
    std::span<const float> patches;
    const qwen3_6::VisionItemControl* control = nullptr;
};

struct VisionScheduleConfig {
    static constexpr int layers              = VisionConfig::layers;
    static constexpr int hidden              = VisionConfig::hidden;
    static constexpr int intermediate        = VisionConfig::intermediate;
    static constexpr int out_hidden          = VisionConfig::output_hidden;
    static constexpr int heads               = VisionConfig::heads;
    static constexpr int head_dim            = VisionConfig::head_dim;
    static constexpr int patch_dim           = VisionConfig::patch_dim;
    static constexpr int merge_unit          = VisionConfig::merge_unit;
    static constexpr int merger_hidden       = VisionConfig::merger_hidden;
    static constexpr int position_embeddings = VisionConfig::position_embeddings;
    static constexpr int rotary_dim          = VisionConfig::rotary_dim;
    static constexpr float rope_theta        = VisionConfig::rope_theta;
    static constexpr float norm_eps          = VisionConfig::norm_epsilon;
};

class VisionWeightStream;

// One vision item's merged embeddings, parked in pinned host memory by an
// overlay window and re-uploaded into the request transient when prefill
// consumes the item.
struct PinnedVisionResult {
    PinnedHostBuffer buffer;
    std::size_t bytes = 0;
};

struct VisionOverlayWindowStats {
    double window_seconds  = 0.0;
    double evict_seconds   = 0.0;
    double restore_seconds = 0.0;
    std::size_t evicted_bytes = 0;
    std::size_t staged_bytes  = 0;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const LoadedModelData& model);
    // Overlay windows bind against a rebased per-window weight view.
    VisionContext(DeviceContext& device, const qwen3_6::VisionWeights& weights);

    [[nodiscard]] static std::size_t output_transient_bytes(std::size_t merged_tokens);
    [[nodiscard]] static std::size_t workspace_bytes(const qwen3_6::VisionItemControl& item);
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                              std::uint32_t max_segments);
    // weight_stream, when set, gates each stage on the streamed uploads of an
    // overlay window; resident execution passes nullptr.
    void encode(const VisionItemView& item, Tensor& output, WorkspaceArena& workspace,
                VisionWeightStream* weight_stream = nullptr) const;

private:
    struct BlockW {
        const Tensor* norm1_weight    = nullptr;
        const Tensor* norm1_bias      = nullptr;
        const Weight* qkv             = nullptr;
        const Tensor* qkv_bias        = nullptr;
        const Weight* projection      = nullptr;
        const Tensor* projection_bias = nullptr;
        const Tensor* norm2_weight    = nullptr;
        const Tensor* norm2_bias      = nullptr;
        const Weight* fc1             = nullptr;
        const Tensor* fc1_bias        = nullptr;
        const Weight* fc2             = nullptr;
        const Tensor* fc2_bias        = nullptr;
    };

    struct MergerW {
        const Tensor* norm_weight = nullptr;
        const Tensor* norm_bias   = nullptr;
        const Weight* fc1         = nullptr;
        const Tensor* fc1_bias    = nullptr;
        const Weight* fc2         = nullptr;
        const Tensor* fc2_bias    = nullptr;
    };

    DeviceContext& ctx_;
    const Weight* patch_embed_      = nullptr;
    const Tensor* patch_embed_bias_ = nullptr;
    const Tensor* position_embed_   = nullptr;
    std::array<BlockW, VisionScheduleConfig::layers> blocks_{};
    MergerW merger_{};
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_6::VisionItemControl* control = nullptr;
    Tensor embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                         WorkspaceArena& workspace, qwen3_6::PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, runtime::TransientRegion transient);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    [[nodiscard]] bool release_consumed_media_payload() noexcept;
    [[nodiscard]] double elapsed_seconds() const;
    // Install embeddings produced by an overlay window: prepare_chunk then
    // uploads them instead of encoding, and the media payload becomes
    // releasable immediately.
    void set_preencoded(std::vector<PinnedVisionResult> results,
                        const VisionOverlayWindowStats& stats);
    [[nodiscard]] const std::optional<VisionOverlayWindowStats>& overlay_stats() const noexcept {
        return overlay_stats_;
    }

private:
    DeviceContext& device_;
    WorkspaceArena& workspace_;
    qwen3_6::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    runtime::TransientRegion transient_;
    VisionContext context_;
    std::optional<std::uint32_t> active_item_;
    std::uint32_t final_item_ = 0;
    bool final_item_encoded_  = false;
    bool payload_released_    = false;
    std::vector<CudaEventTimer> timers_;
    std::vector<PinnedVisionResult> preencoded_;
    std::optional<VisionOverlayWindowStats> overlay_stats_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
