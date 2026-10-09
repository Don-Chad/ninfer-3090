#pragma once

#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
    void encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                const VisionWorkspacePlan& plan) const;

private:
    DeviceContext& ctx_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_5::VisionItemControl* control = nullptr;
    // The item's device handoff [output_hidden, merged].
    Tensor embeddings;
    // Overlay Vision residency (parked; always empty on this build): the item's pinned BF16
    // embeddings, from which a prefill chunk stages the columns it uses.
    std::span<const std::byte> host_embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         DeviceSpan workspace, const VisionWorkspacePlan& workspace_plan,
                         const qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         VisionHandoffState& handoff, std::size_t& handoff_peak_bytes);
    ~VisionPrefillSession();

    VisionPrefillSession(const VisionPrefillSession&)            = delete;
    VisionPrefillSession& operator=(const VisionPrefillSession&) = delete;

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return owns_handoff() ? active_handoff_bytes_ : 0;
    }

private:
    [[nodiscard]] bool owns_handoff() const noexcept {
        return handoff_.owner_ == this && handoff_.generation_ == active_generation_;
    }

    DeviceContext& device_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    const qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    VisionHandoffState& handoff_;
    std::size_t& handoff_peak_bytes_;
    VisionContext context_;
    std::optional<std::uint32_t> active_item_;
    std::uint64_t active_generation_  = 0;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::models::qwen3_5::execution
