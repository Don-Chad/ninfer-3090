#pragma once

#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/execution/vision_overlay.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using detail::VisionWorkspacePlan;
using detail::VisionPrefillPlan;
using detail::VisionUseSpan;

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_5::VisionItemControl* control = nullptr;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const execution::Parameters& parameters);
    // Explicit operands on an explicit compute stream: an overlay window binds parameters rebased
    // onto its borrowed staging and may encode on the Vision stream beside other lanes' decode.
    VisionContext(DeviceContext& device, const VisionConfig& config,
                  const VisionParameters& parameters, cudaStream_t stream);

    [[nodiscard]] static std::size_t workspace_bytes(const VisionConfig& config,
                                                     const VisionParameters& parameters,
                                                     std::size_t patches, std::size_t merged_tokens,
                                                     const VisionWorkspacePlan& plan);
    [[nodiscard]] static VisionWorkspacePlan plan_workspace(const VisionConfig& config,
                                                            const VisionParameters& parameters,
                                                            std::uint32_t max_merged_tokens,
                                                            std::size_t general_capacity_bytes);

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }

    [[nodiscard]] static Tensor bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                            std::size_t merged_tokens);
    // weight_stream, when given, gates each stage on its streamed upload (overlay window).
    void encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                const VisionWorkspacePlan& plan, VisionWeightStream* weight_stream = nullptr) const;

private:
    DeviceContext& ctx_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
    cudaStream_t stream_ = nullptr;
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_5::VisionItemControl* control = nullptr;
    // Resident residency: the item's device handoff [output_hidden, merged].
    Tensor embeddings;
    // Overlay residency: the item's pinned BF16 embeddings; prefill stages the columns it uses.
    std::span<const std::byte> host_embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         DeviceSpan workspace, const VisionWorkspacePlan& workspace_plan,
                         qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes);
    // Overlay residency: items are encoded inside windows brokered by the Program. The bridge
    // staging holds the one visual column an MTP bridge composes outside a prefill chunk.
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         const VisionWorkspacePlan& window_plan,
                         qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes, VisionResidencyBroker& broker,
                         PinnedResultPool::Handle result, DeviceSpan bridge_staging);
    ~VisionPrefillSession();

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    // One visual column of an encoded chunk on the device, for an MTP bridge.
    [[nodiscard]] Tensor bridge_column(const VisionChunk& chunk, std::int32_t column);
    // Overlay residency: starts the encode of the next item ahead of its prefill unit so its window
    // overlaps other lanes' decode. A no-op when a window is open, the item is already active, free
    // KV cannot fund the window or the residency is resident; the synchronous path then stands.
    void submit_next_item();
    // True while a submitted item is still encoding: the lane must not be given a prefill unit.
    [[nodiscard]] bool vision_pending() const;
    // True from the submit of an item until its window is closed by completion, including the span
    // where the encode has finished but the item is not yet consumed. Only one window can be open,
    // so another lane must not start its own encode while this holds.
    [[nodiscard]] bool overlay_window_open() const;
    [[nodiscard]] VisionOverlayWindowStats overlay_stats() const noexcept;
    void release_encoded_media_payloads() noexcept;
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return active_handoff_bytes_;
    }

private:
    void validate_plan() const;

    DeviceContext& device_;
    const execution::Parameters& parameters_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    std::size_t& handoff_peak_bytes_;
    std::optional<VisionContext> context_;
    std::unique_ptr<VisionOverlaySession> overlay_;
    DeviceSpan bridge_staging_;
    std::span<const std::byte> host_result_;
    std::size_t next_use_ = 0;
    std::optional<std::uint32_t> active_item_;
    std::optional<std::uint32_t> submitted_item_;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::models::qwen3_5::execution
