#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    std::optional<qwen3_5::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    // ReplaySSM records, one layout per state shard: the first here, the rest in `extra`.
    std::optional<GdnReplayRecordLayout> replay_records;
    std::vector<GdnReplayRecordLayout> extra_replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_5::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    std::optional<TensorLayout> grammar_masks;
    std::size_t bytes            = 0;
    // Persistent bytes on each further device (device 1 first): the KV planes, block-table copy and
    // recurrent state of the layers that stage owns. Empty on one device.
    std::vector<std::size_t> extra_rank_bytes;
    std::size_t kv_payload_bytes = 0;
};

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    std::uint32_t lookup_ngram              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
};

struct SequencePlanImpl {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t main_page_groups          = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    std::uint32_t lookup_ngram              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    PersistentLayout persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

// The widest forward pass a pipeline stage boundary carries: prefill columns, or every lane's
// verification columns. Sizes the boundary links and the scratch each stage needs around its layers.
[[nodiscard]] inline std::uint64_t stage_boundary_columns(const SequencePlanImpl& plan) noexcept {
    return std::max<std::uint64_t>(
        std::min(plan.prefill_chunk, plan.capacity),
        static_cast<std::uint64_t>(plan.max_concurrency) * (plan.draft_window + 1U));
}

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl> minimum;
};

// Largest merged-token count one media item may occupy under these startup options.
[[nodiscard]] std::uint32_t vision_item_token_bound(std::uint32_t capacity,
                                                    const models::LoadOptions& features);

[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::models::qwen3_5::detail
