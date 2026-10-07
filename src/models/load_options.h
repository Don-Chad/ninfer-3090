#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <vector>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    // Device-memory trades that store a stored Q8 vocabulary matrix (or the MoE MTP layer's routed
    // experts) at a narrower group-64 width while the weights load; the artifact is unchanged.
    bool lm_head_q4     = false;
    bool lm_head_q6     = false;
    bool embedding_q4   = false;
    bool embedding_q6   = false;
    bool mtp_experts_q4 = false;
    // Execution selection carried with the instance: admit the integer small-T gate_up route at
    // verify widths (see execution::DenseParameters::verify_gate_up_policy).
    bool mlp_a8_decode = false;
    // Clearing this returns full prefill tiles to their A16 routes; see EngineOptions::prefill_a8.
    bool prefill_a8 = true;
    // See EngineOptions::prefill_cublas: hands wide prefill GEMMs to cuBLAS, a further quality
    // trade on top of prefill_a8 and therefore off unless asked for.
    bool prefill_cublas = false;
    // See EngineOptions::prefill_cublas_projections.
    bool prefill_cublas_projections = true;
    // Store the GDN recurrent state in FP16 between steps; the recurrence still computes in FP32.
    // Free within measurement noise for a ~2% C8 decode gain and a halved per-slot host state
    // image (docs/maintainer/quality-trade-experiments.md).
    bool gdn_state_fp16 = false;
    // Largest merged-token count one media item may occupy; larger media is downscaled at
    // preprocessing, and the Vision workspace is planned for this bound.
    std::uint32_t vision_max_merged_tokens = 16384;
    // Overlay keeps the Vision tower in pinned Host memory and places the output head, token
    // embedding, proposal head and MTP layer in an evictable device tail that one encode window
    // may borrow. Resident whenever Vision is disabled.
    VisionResidency vision_residency = VisionResidency::Resident;
    // How many devices the instance spans (`--devices`). One is the single-GPU route; more split
    // the layers into that many stages, each owning its layers' weights, KV cache and recurrent
    // state. The embedding, head and round state stay on the first device.
    std::size_t ranks = 1;
    // Layers per stage (`--stage-layers`), one count per device. Empty lets the engine choose.
    std::vector<std::uint32_t> stage_layers;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool overlay_vision() const noexcept {
        return vision && vision_residency == VisionResidency::Overlay;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose        = options.purpose,
            .vision         = options.enable_vision,
            .speculative    = options.speculative.backend,
            .proposal_head  = options.speculative.proposal_head,
            .lm_head_q4     = options.lm_head_q4,
            .lm_head_q6     = options.lm_head_q6,
            .embedding_q4   = options.embedding_q4,
            .embedding_q6   = options.embedding_q6,
            .mtp_experts_q4 = options.mtp_experts_q4,
            .mlp_a8_decode  = options.mlp_a8_decode,
            .prefill_a8     = options.prefill_a8,
            .prefill_cublas = options.prefill_cublas,
            .prefill_cublas_projections = options.prefill_cublas_projections,
            .gdn_state_fp16 = options.gdn_state_fp16,
            .vision_max_merged_tokens = options.vision_max_merged_tokens,
            .vision_residency         = options.enable_vision ? options.vision_residency
                                                              : VisionResidency::Resident,
            .ranks = std::max<std::size_t>(options.devices.size(), 1),
            .stage_layers = options.stage_layers};
}

} // namespace ninfer::models
