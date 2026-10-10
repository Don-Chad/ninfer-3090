#pragma once
#include "models/qwen3_5/program/program.h"
#include "ninfer/ops/attention_geometry.h"

namespace ninfer::models::qwen3_5::execution {
struct MtpCausalAttentionEnvelopes;
struct DFlashEnvelopes;
} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
// The target's full-attention geometry and KV storage, which select the attention routes an MTP
// round records and therefore where its graph topology classes break.
struct MtpGraphAttention {
    ops::AttentionHeadGeometry geometry;
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    // The executing device's SM count, which the INT8-G64 prompt route plans its launch from.
    std::int32_t multiprocessor_count = 0;
};

[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(
    std::uint32_t capacity, std::uint32_t draft_window, const MtpGraphAttention& attention);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window,
                                                                       std::uint32_t batch_size);

[[nodiscard]] execution::MtpCausalAttentionEnvelopes
mtp_causal_attention_envelopes(std::uint32_t max_frontier, std::uint32_t k, std::uint32_t capacity);
[[nodiscard]] execution::DFlashEnvelopes dflash_envelopes(std::uint32_t max_frontier,
                                                          std::uint32_t k);

} // namespace ninfer::models::qwen3_5::detail
