#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::filesystem::path grammar_path;
    std::filesystem::path json_schema_path;
    std::optional<std::string> regex;
    std::vector<std::string> choices;
    bool json_object = false;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 1024;
    int device                   = 0;
    // Several ids split the model's layers into one pipeline stage per device, the first also
    // holding the embedding, head and round state. Empty means use `device`.
    std::vector<int> devices;
    // Layers per pipeline stage, one count per entry of `devices`. Empty lets the engine choose.
    std::vector<std::uint32_t> stage_layers;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    std::uint32_t vision_max_merged_tokens = 16384;
    bool use_cuda_graph = true;
    bool lm_head_q4     = false;
    bool lm_head_q6     = false;
    bool embedding_q4   = false;
    bool embedding_q6   = false;
    bool mtp_experts_q4 = false;
    bool gdn_state_fp16 = false;
    bool mlp_a8_decode  = false;
    bool prefill_a8     = true;
    bool prefill_cublas = false;
    bool prefill_cublas_projections = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    ninfer::ReasoningLoopAction reasoning_loop = ninfer::ReasoningLoopAction::Off;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy                 = false;
    product::LogLevel log_level = product::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
