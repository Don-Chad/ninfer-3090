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
using detail::VisionHandoffState;

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
    // Overlay residency: the item's pinned BF16 embeddings, from which a prefill chunk stages the
    // columns it uses.
    std::span<const std::byte> host_embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         DeviceSpan workspace, const VisionWorkspacePlan& workspace_plan,
                         const qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         VisionHandoffState& handoff, std::size_t& handoff_peak_bytes);
    // Overlay residency: items are encoded inside windows brokered by the Program, into this
    // session's pinned result slot. The bridge staging holds the one visual column an MTP bridge
    // composes outside a prefill chunk.
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         const VisionWorkspacePlan& window_plan,
                         const qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         VisionResidencyBroker& broker, PinnedResultPool::Handle result,
                         DeviceSpan bridge_staging);
    ~VisionPrefillSession();

    VisionPrefillSession(const VisionPrefillSession&)            = delete;
    VisionPrefillSession& operator=(const VisionPrefillSession&) = delete;

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    // One visual column of an encoded chunk on the device, for an MTP bridge.
    [[nodiscard]] Tensor bridge_column(const VisionChunk& chunk, std::int32_t column);

    // Overlay residency: device bytes the window of the item a chunk of `chunk` tokens starting at
    // `cursor` consumes, or nothing when that chunk needs no new encode (no item, already encoded
    // or submitted).
    [[nodiscard]] std::optional<std::size_t> pending_window_bytes(std::uint32_t cursor,
                                                                  std::uint32_t chunk) const;
    // Overlay residency: opens a KV-funded window for that item and starts its encode on the Vision
    // stream, so it runs beside other lanes' units. False, leaving nothing open, when the window
    // cannot be funded from free KV now; the prefill unit then encodes synchronously.
    [[nodiscard]] bool submit_item(std::uint32_t cursor, std::uint32_t chunk);
    // True while a submitted item is still encoding: the lane must not be given a prefill unit.
    [[nodiscard]] bool vision_pending() const;
    // Closes the window of a submitted item whose encode has finished. Returns whether it did.
    bool poll();
    [[nodiscard]] VisionOverlayWindowStats overlay_stats() const noexcept;

    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return owns_handoff() ? active_handoff_bytes_ : 0;
    }

private:
    void validate_plan() const;
    [[nodiscard]] const VisionUseSpan* use_at(std::uint32_t cursor) const;

    [[nodiscard]] bool owns_handoff() const noexcept {
        return handoff_ != nullptr && handoff_->owner_ == this &&
               handoff_->generation_ == active_generation_;
    }

    DeviceContext& device_;
    const execution::Parameters& parameters_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    const qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    VisionHandoffState* handoff_      = nullptr;
    std::size_t* handoff_peak_bytes_  = nullptr;
    std::optional<VisionContext> context_;
    std::optional<std::uint32_t> active_item_;
    std::uint64_t active_generation_  = 0;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<CudaEventTimer> timers_;
    // Overlay residency.
    std::unique_ptr<VisionOverlaySession> overlay_;
    DeviceSpan bridge_staging_;
    std::span<const std::byte> host_result_;
    std::optional<std::uint32_t> submitted_item_;
};

} // namespace ninfer::models::qwen3_5::execution
