#include "guarded_main.h"
#include "speculative_graft.h"
#include "ninfer/engine.h"
#include "runtime/engine/worker_fault.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 1024;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.enable_vision                    = true;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 4;
    return options;
}

ninfer::EngineOptions host_restore_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions shared_replacement_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 4;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions anthropic_prefix_regression_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 2048;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk                    = 512;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 1;
    options.context_cache.host_state_slots   = 4;
    options.context_cache.host_kv_capacity_bytes            = 512ULL << 20;
    options.context_cache.max_private_continuations         = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions shared_rewrite_materialization_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 100000;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(100000);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 2;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions private_long_anchor_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::None;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 4;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 1;
    return options;
}

ninfer::EngineOptions last_alias_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 3;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions concurrent_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 512;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 8;
    options.max_pending_requests             = 8;
    options.context_cache.device_state_slots = 16;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 8;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions pressure_resume_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 8192;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 8ULL << 30;
    options.context_cache.max_private_continuations         = 4;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions private_checkpoint_pressure_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 8192;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(16384);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 4;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

// A request's result is delivered before the worker releases what it held (its shared-prefix
// reference, its lane), so a snapshot taken the moment generate() returns can still show the
// request's references. Wait for the worker to settle so a persisting reference is a real leak.
ninfer::RuntimeStats settled_runtime_stats(const ninfer::Engine& engine) {
    ninfer::RuntimeStats stats = engine.runtime_stats();
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (stats.running_requests == 0 && stats.terminal_pending_requests == 0 &&
            stats.shared_active_references == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        stats = engine.runtime_stats();
    }
    return stats;
}

std::vector<std::uint8_t> gradient_ppm(int width = 64, int height = 64) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput chinese_chat(bool enable_thinking) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "你好，简单介绍一下你自己。", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

// Token counts of one short Chinese chat under each registered chat template. Qwen3.8's template
// states the default xhigh reasoning effort in thinking mode, which Qwen3.6's does not, so the
// thinking count differs by family while the non-thinking count does not.
int exercise_registered_frontend(const ninfer::Engine& engine) {
    struct Golden {
        std::string_view family;
        std::uint32_t thinking;
        std::uint32_t non_thinking;
    };
    constexpr Golden kGoldens[] = {
        {"qwen3.6", 16, 18},
        {"qwen3.8", 58, 18},
    };
    const std::string model       = engine.load_summary().model_name;
    const std::uint32_t thinking     = engine.count_tokens(chinese_chat(true));
    const std::uint32_t non_thinking = engine.count_tokens(chinese_chat(false));
    for (const Golden& golden : kGoldens) {
        if (!model.starts_with(golden.family)) { continue; }
        if (thinking != golden.thinking || non_thinking != golden.non_thinking) {
            std::cerr << "registered tokenizer/chat template changed the prompt golden for " << model
                      << ": thinking " << thinking << " (expected " << golden.thinking
                      << "), non-thinking " << non_thinking << " (expected "
                      << golden.non_thinking << ")\n";
            return 1;
        }
        return 0;
    }
    std::cerr << "no registered prompt golden for model " << model << ": thinking " << thinking
              << ", non-thinking " << non_thinking << "\n";
    return 1;
}

class ObservationSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart start) override {
        if (started_) { valid_ = false; }
        started_ = true;
        start_   = start;
    }

    void progress(ninfer::PromptProgress progress) override {
        if (!started_ || timing_seen_ ||
            progress.total_prompt_tokens != start_.prompt.prompt_tokens ||
            progress.reused_prompt_tokens != start_.reused_prompt_tokens ||
            progress.processed_prompt_tokens < last_processed_ ||
            progress.processed_prompt_tokens > progress.total_prompt_tokens ||
            progress.elapsed_ns < last_progress_elapsed_ns_) {
            valid_ = false;
        }
        last_processed_           = progress.processed_prompt_tokens;
        last_progress_elapsed_ns_ = progress.elapsed_ns;
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (!started_ || last_processed_ != start_.prompt.prompt_tokens ||
            (timing_seen_ && (timing.generated_tokens < last_timing_.generated_tokens ||
                              timing.prompt_elapsed_ns != last_timing_.prompt_elapsed_ns ||
                              timing.generation_elapsed_ns < last_timing_.generation_elapsed_ns))) {
            valid_ = false;
        }
        timing_seen_ = true;
        last_timing_ = timing;
    }

    void publish(ninfer::OutputDelta) override {
        if (!timing_seen_) { valid_ = false; }
    }

    [[nodiscard]] bool valid_for(const ninfer::GenerationResult& result) const {
        return valid_ && started_ && timing_seen_ && start_.reused_prompt_tokens == 0 &&
               last_processed_ == start_.prompt.prompt_tokens &&
               last_timing_.generated_tokens == result.generated_token_ids.size() &&
               result.timings.prompt_wall_seconds > 0.0 &&
               result.timings.generation_wall_seconds >= 0.0;
    }

private:
    ninfer::GenerationStart start_;
    ninfer::GenerationTimingObservation last_timing_;
    std::uint32_t last_processed_           = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    bool started_                           = false;
    bool timing_seen_                       = false;
    bool valid_                             = true;
};

int exercise_stream_observations(ninfer::Engine& engine) {
    std::vector<ninfer::TokenId> prompt(2050, 198);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const ninfer::GenerationObservationOptions observation{
        .phase_timings = true, .live_timings = true, .prompt_progress = true};

    ObservationSink sink;
    ninfer::GenerationHandle generation =
        engine.submit(engine.prepare_tokens(std::move(prompt)), std::move(request),
                      ninfer::OutputConsumerMode::Streaming, observation);
    const ninfer::GenerationResult result = generation.wait(&sink);
    if (result.generated_token_ids.size() != 3 || !sink.valid_for(result)) {
        std::cerr
            << "stream observations lost prompt progress, commit timing, or publication order\n";
        return 1;
    }
    return 0;
}

int exercise_full_prefill_chunk(ninfer::Engine& engine) {
    constexpr std::size_t kChunkTokens = 1024;
    std::vector<ninfer::TokenId> prompt(kChunkTokens, 198);
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(std::move(prompt)), options);
    if (result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "full-chunk prefill did not complete through the planned workspace\n";
        return 1;
    }
    return 0;
}

int exercise_abandoned_handle_capacity(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    {
        auto abandoned = engine.submit(engine.prepare_tokens(prompt), request);
        if (!abandoned) {
            std::cerr << "abandonment fixture did not create a generation handle\n";
            return 1;
        }
    }
    const auto crossed = engine.generate(engine.prepare_tokens(prompt), request);
    if (crossed.generated_token_ids.size() != 1) {
        std::cerr << "request after an abandoned handle did not complete\n";
        return 1;
    }

    auto first      = engine.submit(engine.prepare_tokens(prompt), request);
    auto second     = engine.submit(engine.prepare_tokens(prompt), request);
    bool overloaded = false;
    try {
        auto third = engine.submit(engine.prepare_tokens(prompt), request);
        (void)third;
    } catch (const ninfer::RequestError& error) {
        overloaded = error.kind() == ninfer::RequestErrorKind::Overloaded;
    }
    if (!overloaded) {
        std::cerr << "outstanding capacity was released twice or not enforced\n";
        return 1;
    }
    if (first.wait().generated_token_ids.size() != 1 ||
        second.wait().generated_token_ids.size() != 1) {
        std::cerr << "requests retained after the overload check did not complete\n";
        return 1;
    }
    return 0;
}

int exercise_zero_suffix_reuse(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    ninfer::RequestOptions baseline_options;
    baseline_options.execution.requested_output_tokens = 8;
    baseline_options.execution.sampling.temperature    = 0.0F;
    baseline_options.execution.allow_prefix_reuse      = true;
    baseline_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), baseline_options);
    if (baseline.generated_token_ids.size() != 8) {
        std::cerr << "zero-suffix baseline did not generate eight tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> exact_frontier = prompt;
    exact_frontier.insert(exact_frontier.end(), baseline.generated_token_ids.begin(),
                          baseline.generated_token_ids.end() - 1);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 2;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(exact_frontier), reuse_options);
    if (reused.reused_prompt_tokens != exact_frontier.size()) {
        std::cerr << "zero-suffix reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << exact_frontier.size() << '\n';
        return 1;
    }
    if (reused.generated_token_ids.size() != 2 ||
        reused.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "zero-suffix reuse did not resume from the retained target frontier\n";
        return 1;
    }
    return 0;
}

int exercise_prefix(ninfer::Engine& engine) {
    ninfer::RequestOptions first_options;
    first_options.execution.requested_output_tokens = 5;
    first_options.execution.sampling.temperature    = 0.0F;
    first_options.stop.include_model_defaults       = false;

    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), first_options);
    if (first.generated_token_ids.size() != 5) {
        std::cerr << "first request did not generate five tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 5;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), reuse_options);

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "append reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }

    if (const int result = exercise_zero_suffix_reuse(engine, prompt); result != 0) {
        return result;
    }

    return 0;
}

int exercise_host_restore(const char* artifact) {
    ninfer::Engine engine(host_restore_engine_options(artifact));
    auto options = [](std::uint32_t outputs, bool reuse) {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = outputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = reuse;
        request.stop.include_model_defaults       = false;
        return request;
    };

    const auto retained_input = [] {
        std::string text;
        text.reserve(6U * 300U);
        for (std::uint32_t index = 0; index < 300; ++index) { text += "alpha "; }
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "host-restore-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };

    const ninfer::GenerationResult retained =
        engine.generate(engine.prepare(retained_input()), options(5, true));
    if (retained.prompt.prompt_tokens <= 256 || retained.generated_token_ids.size() != 5) {
        std::cerr << "Host-restore source request did not complete\n";
        return 1;
    }

    ninfer::PromptInput continuation = retained_input();
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = retained.reasoning;
    // The replayed reply diverges from its first generated token on purpose. Qwen3.8's template
    // re-renders a non-thinking reply token for token, so a replay that keeps the generated tokens
    // as its prefix correctly reuses the whole endpoint and never reaches the turn-closure
    // checkpoint this scenario restores from Host.
    assistant.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                  .text  = "Edited reply. " + retained.content,
                                                  .media = {}});
    continuation.messages.push_back(std::move(assistant));
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    continuation.messages.push_back(std::move(followup));

    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult pressure_result =
        engine.generate(engine.prepare(continuation), options(2, false));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    if (pressure_result.generated_token_ids.size() != 2 ||
        after_pressure.state_d2h_count <= before_pressure.state_d2h_count ||
        after_pressure.main_kv_d2h_pages <= before_pressure.main_kv_d2h_pages ||
        after_pressure.backend_kv_d2h_pages <= before_pressure.backend_kv_d2h_pages) {
        std::cerr << "Host pressure did not demote the complete MTP checkpoint: state="
                  << after_pressure.state_d2h_count << " main=" << after_pressure.main_kv_d2h_pages
                  << " backend=" << after_pressure.backend_kv_d2h_pages
                  << " degraded=" << after_pressure.pressure_private_owners_degraded
                  << " evicted=" << after_pressure.pressure_private_owners_evicted << '\n';
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(std::move(continuation)), options(2, true));
    const ninfer::RuntimeStats after_restore = engine.runtime_stats();
    if (restored.generated_token_ids.size() != 2 ||
        restored.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
        restored.reused_prompt_tokens == 0 ||
        after_restore.state_h2d_count <= after_pressure.state_h2d_count ||
        after_restore.main_kv_h2d_pages <= after_pressure.main_kv_h2d_pages ||
        after_restore.backend_kv_h2d_pages <= after_pressure.backend_kv_h2d_pages) {
        std::cerr << "Complete MTP checkpoint was not materialized from Host: path="
                  << static_cast<int>(restored.prefix_reuse_path)
                  << " reused=" << restored.reused_prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size()
                  << " state=" << after_restore.state_h2d_count
                  << " main=" << after_restore.main_kv_h2d_pages
                  << " backend=" << after_restore.backend_kv_h2d_pages
                  << " degraded=" << after_restore.pressure_private_owners_degraded
                  << " evicted=" << after_restore.pressure_private_owners_evicted << '\n';
        return 1;
    }

    // The uncached pressure request and checkpoint resume use different valid prefill splits, so
    // the pressure result is a completion and transfer trigger rather than an exact-token oracle.
    return 0;
}

int exercise_shared_replacement_and_full_capacity_reuse(const char* artifact) {
    ninfer::EngineOptions engine_options = shared_replacement_engine_options(artifact);
    engine_options.max_context           = 1024;
    engine_options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    engine_options.context_cache.max_private_continuations = 1;
    ninfer::Engine engine(std::move(engine_options));
    ninfer::RequestOptions capture_request;
    capture_request.execution.requested_output_tokens = 1;
    capture_request.execution.sampling.temperature    = 0.0F;
    capture_request.execution.allow_prefix_reuse      = true;
    capture_request.stop.include_model_defaults       = false;

    const auto plain_prompt = [](std::string text) {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        return input;
    };
    const auto tool_prompt = [](std::string tool_json, std::string question) {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.options.tool_jsons.push_back(std::move(tool_json));
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = 1,
        });
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        return input;
    };

    std::string observed_text;
    for (std::uint32_t index = 0; index < 4; ++index) { observed_text += "observed-prefix "; }
    const ninfer::GenerationResult observed_first =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    const ninfer::RuntimeStats after_observed_first = engine.runtime_stats();
    const ninfer::GenerationResult observed_second =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    const ninfer::RuntimeStats after_observed_second = engine.runtime_stats();
    if (observed_first.generated_token_ids.size() != 1 ||
        observed_second.generated_token_ids.size() != 1 ||
        after_observed_first.active_captures_completed != 1 ||
        after_observed_second.active_captures_completed !=
            after_observed_first.active_captures_completed + 1U) {
        std::cerr << "observed-prefix private/shared capture sequence changed: "
                  << after_observed_first.active_captures_completed << '/'
                  << after_observed_second.active_captures_completed << '\n';
        return 1;
    }

    const ninfer::GenerationResult observed_filler = engine.generate(
        engine.prepare(plain_prompt("Unrelated private endpoint.")), capture_request);
    const ninfer::GenerationResult observed_reuse =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    if (observed_filler.generated_token_ids.size() != 1 ||
        observed_reuse.generated_token_ids.size() != 1 ||
        observed_reuse.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        observed_reuse.reused_prompt_tokens == 0) {
        std::cerr << "promoted shared prefix was not reusable after private eviction: path="
                  << static_cast<int>(observed_reuse.prefix_reuse_path)
                  << " reused=" << observed_reuse.reused_prompt_tokens << '\n';
        return 1;
    }

    std::string long_description;
    for (std::uint32_t index = 0; index < 240; ++index) { long_description += "stable-schema "; }
    const std::string bravo_tool =
        std::string(R"({"type":"function","function":{"name":"bravo","description":")") +
        long_description +
        R"(","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})";
    const ninfer::GenerationResult replacement = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), capture_request);
    const ninfer::RuntimeStats after_replacement = settled_runtime_stats(engine);
    if (replacement.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not produce its deterministic stop token\n";
        return 1;
    }
    // Remove the exact private endpoint without publishing another shared marker. The following
    // identical Bravo prompt must therefore materialize from the retained shared prefix.
    const ninfer::GenerationResult filled =
        engine.generate(engine.prepare(plain_prompt("Another private endpoint.")), capture_request);
    const ninfer::RuntimeStats after_filler = settled_runtime_stats(engine);
    if (filled.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not displace the exact private endpoint\n";
        return 1;
    }
    ninfer::RequestOptions full_capacity_request = capture_request;
    full_capacity_request.execution.requested_output_tokens =
        std::numeric_limits<std::uint32_t>::max();
    full_capacity_request.stop.token_ids          = {replacement.generated_token_ids.front()};
    full_capacity_request.stop.publish_stop_token = true;
    const ninfer::GenerationResult reused         = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), full_capacity_request);
    const ninfer::RuntimeStats after_reuse = settled_runtime_stats(engine);

    if (reused.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not complete all requests\n";
        return 1;
    }
    if (reused.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        reused.reused_prompt_tokens == 0 || reused.reused_prompt_tokens % 64U == 0) {
        std::cerr << "single-slot shared replacement was not reusable at a non-aligned frontier: "
                  << "path=" << static_cast<int>(reused.prefix_reuse_path)
                  << " reused=" << reused.reused_prompt_tokens
                  << " captures=" << after_replacement.active_captures_completed << '/'
                  << after_filler.active_captures_completed << '/'
                  << after_reuse.active_captures_completed
                  << " shared_evicted=" << after_replacement.pressure_shared_owners_evicted << '/'
                  << after_filler.pressure_shared_owners_evicted << '/'
                  << after_reuse.pressure_shared_owners_evicted
                  << " shared_degraded=" << after_replacement.pressure_shared_owners_degraded << '/'
                  << after_filler.pressure_shared_owners_degraded << '/'
                  << after_reuse.pressure_shared_owners_degraded
                  << " private_evicted=" << after_replacement.pressure_private_owners_evicted << '/'
                  << after_filler.pressure_private_owners_evicted << '/'
                  << after_reuse.pressure_private_owners_evicted
                  << " refs=" << after_replacement.shared_active_references << '/'
                  << after_filler.shared_active_references << '/'
                  << after_reuse.shared_active_references
                  << " targets=" << reused.materialization.targets_evaluated
                  << " degradation=" << reused.materialization.selected_degradation_units
                  << " maximal=" << reused.materialization.selected_maximal_fallback << " stop="
                  << ninfer::materialization_stop_reason_name(reused.materialization.stop_reason)
                  << '\n';
        return 1;
    }
    if (after_replacement.active_captures_completed < 2 ||
        after_reuse.active_captures_completed < after_replacement.active_captures_completed) {
        std::cerr << "shared replacement capture was skipped under full State/KV capacity: "
                  << after_replacement.active_captures_completed << '/'
                  << after_reuse.active_captures_completed << '\n';
        return 1;
    }
    if (after_replacement.shared_active_references != 0 ||
        after_filler.shared_active_references != 0 || after_reuse.shared_active_references != 0) {
        std::cerr << "completed shared-prefix requests leaked active references: "
                  << after_replacement.shared_active_references << '/'
                  << after_filler.shared_active_references << '/'
                  << after_reuse.shared_active_references << '\n';
        return 1;
    }
    return 0;
}

int exercise_anthropic_prefix_regression(const char* artifact) {
    ninfer::Engine engine(anthropic_prefix_regression_engine_options(artifact));
    if (!engine.options().context_cache.max_shared_prefixes ||
        *engine.options().context_cache.max_shared_prefixes !=
            ninfer::kMaximumExplicitPromptCacheMarkers) {
        std::cerr << "single-concurrency Engine did not expose four default shared prefixes\n";
        return 1;
    }

    const auto conversation = [](bool followup, const ninfer::GenerationResult* first = nullptr) {
        ninfer::PromptInput input;
        input.options.enable_thinking   = true;
        input.options.preserve_thinking = true;
        input.context_cache.session_key = "anthropic-prefix-regression";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;

        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                 .text  = "Reply with exactly the word blue.",
                                                 .media = {}});
        input.messages.push_back(std::move(user));
        if (!followup) { return input; }
        if (first == nullptr) { throw std::logic_error("followup fixture has no source result"); }

        ninfer::ChatMessage assistant;
        assistant.role              = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = first->reasoning;
        if (!first->content.empty()) {
            assistant.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = first->content, .media = {}});
        }
        input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage next;
        next.role = ninfer::ChatRole::User;
        next.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                 .text  = "Now reply with exactly the word green.",
                                                 .media = {}});
        input.messages.push_back(std::move(next));
        return input;
    };
    const auto generation_options = [](std::uint32_t outputs, bool model_stops) {
        ninfer::RequestOptions options;
        options.execution.requested_output_tokens = outputs;
        options.execution.sampling.temperature    = 0.0F;
        options.execution.allow_prefix_reuse      = true;
        options.stop.include_model_defaults       = model_stops;
        return options;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation(false)), generation_options(192, true));
    if (first.reasoning.empty() || first.content.empty() ||
        first.finish_reason != ninfer::FinishReason::StopToken) {
        std::cerr << "endpoint regression fixture did not produce a closed reasoning response: "
                  << "reasoning=" << first.reasoning.size() << " content=" << first.content.size()
                  << " finish=" << static_cast<int>(first.finish_reason) << '\n';
        return 1;
    }
    const ninfer::RuntimeStats before_followup = engine.runtime_stats();
    const ninfer::GenerationResult followup =
        engine.generate(engine.prepare(conversation(true, &first)), generation_options(1, false));
    const ninfer::RuntimeStats after_followup = engine.runtime_stats();
    const std::uint64_t followup_prefill =
        after_followup.computed_prefill_tokens - before_followup.computed_prefill_tokens;
    if (followup.generated_token_ids.size() != 1 ||
        followup.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint ||
        followup.reused_prompt_tokens == 0 ||
        followup_prefill != followup.prompt.prompt_tokens - followup.reused_prompt_tokens) {
        std::cerr << "model-output reasoning frontier did not resume from PrivateEndpoint: path="
                  << static_cast<int>(followup.prefix_reuse_path)
                  << " reused=" << followup.reused_prompt_tokens
                  << " prompt=" << followup.prompt.prompt_tokens << " computed=" << followup_prefill
                  << '\n';
        return 1;
    }

    const auto tool_definition = [](std::string name, std::string word) {
        std::string description;
        for (std::uint32_t index = 0; index < 160; ++index) {
            description += word;
            description.push_back(' ');
        }
        return std::string(R"({"type":"function","function":{"name":")") + name +
               R"(","description":")" + description +
               R"(","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}})";
    };
    const std::string alpha   = tool_definition("alpha", "stable-alpha");
    const std::string bravo   = tool_definition("bravo", "stable-bravo");
    const std::string charlie = tool_definition("charlie", "branch-charlie");
    const auto tool_prompt    = [](std::vector<std::string> tools, std::string question) {
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.options.tool_jsons      = std::move(tools);
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        for (std::uint32_t count = 1; count <= input.options.tool_jsons.size(); ++count) {
            input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                   .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                   .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                   .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
                   .after_tool_count = count,
            });
        }
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
               .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
        input.messages.push_back(std::move(user));
        return input;
    };
    const auto filler_prompt = [](std::string text) {
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        return input;
    };

    const ninfer::RequestOptions one_token = generation_options(1, false);
    const ninfer::GenerationResult seed    = engine.generate(
        engine.prepare(tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninfer::GenerationResult first_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private seed continuation.")), one_token);
    if (seed.generated_token_ids.size() != 1 || first_filler.generated_token_ids.size() != 1) {
        std::cerr << "shared compact fixture did not establish its seed and filler\n";
        return 1;
    }

    const ninfer::RuntimeStats before_late = engine.runtime_stats();
    const ninfer::GenerationResult late    = engine.generate(
        engine.prepare(tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninfer::RuntimeStats after_late = engine.runtime_stats();
    const std::uint64_t late_prefill =
        after_late.computed_prefill_tokens - before_late.computed_prefill_tokens;
    const ninfer::GenerationResult second_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private late continuation.")), one_token);
    const ninfer::RuntimeStats before_branch = engine.runtime_stats();
    const ninfer::GenerationResult branch    = engine.generate(
        engine.prepare(tool_prompt({alpha, charlie}, "Use one listed function.")), one_token);
    const ninfer::RuntimeStats after_branch = engine.runtime_stats();
    const std::uint64_t branch_prefill =
        after_branch.computed_prefill_tokens - before_branch.computed_prefill_tokens;

    if (late.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        branch.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        late.generated_token_ids.size() != 1 || branch.generated_token_ids.size() != 1 ||
        late.reused_prompt_tokens <= branch.reused_prompt_tokens ||
        branch.reused_prompt_tokens == 0 || second_filler.generated_token_ids.size() != 1 ||
        late_prefill != late.prompt.prompt_tokens - late.reused_prompt_tokens ||
        branch_prefill != branch.prompt.prompt_tokens - branch.reused_prompt_tokens) {
        std::cerr << "default shared catalog did not retain nested compact frontiers: late_path="
                  << static_cast<int>(late.prefix_reuse_path)
                  << " late_reused=" << late.reused_prompt_tokens
                  << " late_prompt=" << late.prompt.prompt_tokens
                  << " late_computed=" << late_prefill
                  << " branch_path=" << static_cast<int>(branch.prefix_reuse_path)
                  << " branch_reused=" << branch.reused_prompt_tokens
                  << " branch_prompt=" << branch.prompt.prompt_tokens
                  << " branch_computed=" << branch_prefill << '\n';
        return 1;
    }
    return 0;
}

int exercise_shared_rewrite_materialization(const char* artifact) {
    ninfer::Engine engine(shared_rewrite_materialization_engine_options(artifact));

    std::vector<std::string> tools;
    tools.reserve(40);
    tools.push_back(
        R"({"type":"function","function":{"name":"read_chunk","description":"Read the next diagnostic chunk. Always use this tool until told done.","parameters":{"type":"object","properties":{"chunk":{"type":"integer"}},"required":["chunk"]}}})");
    for (std::uint32_t index = 1; index < 40; ++index) {
        tools.push_back(
            std::string(R"({"type":"function","function":{"name":"unused_tool_)") +
            std::to_string(index) +
            R"(","description":"Unused diagnostic tool.","parameters":{"type":"object","properties":{"value":{"type":"string"}}}}})");
    }

    constexpr std::string_view system_text =
        "You are testing a tool loop. On every turn call read_chunk exactly once with the next "
        "integer chunk number. Do not finish or answer in prose.";
    std::vector<ninfer::ChatMessage> messages;
    ninfer::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::string(system_text), .media = {}});
    messages.push_back(std::move(system));
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Start by calling read_chunk with chunk 1.",
        .media = {},
    });
    messages.push_back(std::move(user));

    const auto input = [&](bool latest_is_tool) {
        ninfer::PromptInput prompt;
        prompt.messages                  = messages;
        prompt.options.enable_thinking   = true;
        prompt.options.preserve_thinking = true;
        prompt.options.tool_jsons        = tools;
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
            .leading_instruction_bytes = static_cast<std::uint32_t>(system_text.size()),
        });
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(tools.size()),
        });
        if (latest_is_tool) {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(messages.size()),
                .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence            = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        } else {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count      = static_cast<std::uint32_t>(messages.size()),
                .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence                 = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                .after_message_part_count = 1,
            });
        }
        return prompt;
    };
    const auto request_options = [] {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 16384;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.thinking.budget         = 1024;
        request.execution.allow_prefix_reuse      = true;
        return request;
    };
    const auto append_result = [&](const ninfer::GenerationResult& result, std::uint32_t turn) {
        if (result.tool_calls.size() != 1) {
            throw std::logic_error("shared rewrite fixture did not produce one tool call");
        }
        ninfer::ChatMessage assistant;
        assistant.role              = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = result.reasoning + "\nHistory normalized by the client.";
        if (!result.content.empty()) {
            assistant.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        }
        const std::string call_id = "call_" + std::to_string(turn);
        assistant.tool_calls.push_back(ninfer::ToolCall{
            .id             = call_id,
            .name           = result.tool_calls.front().name,
            .arguments_json = result.tool_calls.front().arguments_json,
        });
        messages.push_back(std::move(assistant));

        std::string diagnostic;
        diagnostic.reserve(64000);
        for (std::uint32_t line = 0; line < 500; ++line) {
            diagnostic += "chunk=" + std::to_string(turn) + " line=" + std::to_string(line) +
                          " key=value abcdefghijklmnopqrstuvwxyz0123456789 "
                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ9876543210\n";
        }
        ninfer::ChatMessage tool_result;
        tool_result.role         = ninfer::ChatRole::Tool;
        tool_result.tool_call_id = call_id;
        tool_result.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(diagnostic), .media = {}});
        messages.push_back(std::move(tool_result));
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input(false)), request_options());
    append_result(first, 1);
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(input(true)), request_options());
    append_result(second, 2);
    const ninfer::RuntimeStats before_third = engine.runtime_stats();
    const ninfer::GenerationResult third =
        engine.generate(engine.prepare(input(true)), request_options());
    const ninfer::RuntimeStats after_third = engine.runtime_stats();

    if (first.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        second.reused_prompt_tokens == 0 ||
        third.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        third.reused_prompt_tokens == 0 || third.generated_token_ids.empty() ||
        after_third.historical_fork_hits <= before_third.historical_fork_hits) {
        std::cerr << "shared/private rewrite alias did not materialize through its active Fork: "
                  << "first_path=" << static_cast<int>(first.prefix_reuse_path)
                  << " second_path=" << static_cast<int>(second.prefix_reuse_path)
                  << " second_reused=" << second.reused_prompt_tokens
                  << " third_path=" << static_cast<int>(third.prefix_reuse_path)
                  << " third_reused=" << third.reused_prompt_tokens
                  << " third_outputs=" << third.generated_token_ids.size()
                  << " forks=" << before_third.historical_fork_hits << '/'
                  << after_third.historical_fork_hits << '\n';
        return 1;
    }
    return 0;
}

// A consumed private endpoint whose long anchor aliases a StateImage another owner also holds
// must not count that image in its active entitlement. Two conversations share a system prompt and
// each places an explicit long anchor on the system boundary, where it aliases the structural
// shared-prefix image; one lane with one spare Device state slot forces anchors onto the Host.
// Counting the shared anchor made start_request's actual-vs-expected invariant throw (or the
// state reservation raise bad_alloc) and latched the Engine within a few turns.
int exercise_shared_anchor_entitlement(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 1024;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 4;
    options.context_cache.host_kv_capacity_bytes = std::size_t{64} << 20U;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 4;
    options.context_cache.max_long_anchors_per_continuation = 2;
    ninfer::Engine engine(std::move(options));

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    std::string system =
        "You are a careful engineering assistant. Follow the house style: short sentences, units "
        "on every number, and name the assumption behind every estimate.";
    for (int line = 0; line < 12; ++line) {
        system += " Rule " + std::to_string(line) +
                  ": prefer measured figures to recalled ones, and say which you used.";
    }
    const auto conversation = [&](const std::vector<std::string>& turns) {
        ninfer::PromptInput prompt;
        ninfer::ChatMessage head;
        head.role = ninfer::ChatRole::System;
        head.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = system, .media = {}});
        prompt.messages.push_back(std::move(head));
        for (std::size_t index = 0; index < turns.size(); ++index) {
            ninfer::ChatMessage message;
            message.role = index % 2 == 0 ? ninfer::ChatRole::User : ninfer::ChatRole::Assistant;
            message.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = turns[index], .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking = false;
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = 1,
            .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
            .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
        });
        return prompt;
    };

    std::array<std::vector<std::string>, 2> turns{
        std::vector<std::string>{"Plan a three-day walking route through the lake district."},
        std::vector<std::string>{"Summarise the trade-offs between paged and contiguous KV."}};
    std::uint32_t reused_requests = 0;
    for (std::uint32_t round = 0; round < 5; ++round) {
        for (std::size_t lineage = 0; lineage < turns.size(); ++lineage) {
            const ninfer::GenerationResult result =
                engine.generate(engine.prepare(conversation(turns[lineage])), request);
            if (result.generated_token_ids.size() != 1) {
                std::cerr << "shared-anchor request generated no token: round=" << round
                          << " lineage=" << lineage << "\n";
                return 1;
            }
            if (result.reused_prompt_tokens != 0) { ++reused_requests; }
            turns[lineage].push_back("Answer " + std::to_string(round) + " for lineage " +
                                     std::to_string(lineage) + ", with enough detail to fill a page.");
            turns[lineage].push_back("Continue with part " + std::to_string(round + 1) + ".");
        }
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    if (!engine.is_available() || reused_requests == 0 || stats.state_d2h_count == 0) {
        std::cerr << "shared-anchor scenario did not exercise reuse under Host pressure: reused="
                  << reused_requests << " state_d2h=" << stats.state_d2h_count << "\n";
        return 1;
    }
    return 0;
}

ninfer::EngineOptions automatic_anchor_engine_options(const char* artifact,
                                                      std::uint32_t device_state_slots,
                                                      std::uint32_t host_state_slots,
                                                      std::uint32_t shared_prefixes) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 1024;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = device_state_slots;
    options.context_cache.host_state_slots       = host_state_slots;
    options.context_cache.host_kv_capacity_bytes = std::size_t{64} << 20U;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = shared_prefixes;
    options.context_cache.max_long_anchors_per_continuation = 2;
    return options;
}

// Turns alternate User/Assistant; with `leading_system` the first turn is the System message and
// the alternation starts after it.
ninfer::PromptInput automatic_anchor_conversation(const std::vector<std::string>& turns,
                                                  std::uint32_t automatic_anchors,
                                                  bool leading_system = false) {
    ninfer::PromptInput prompt;
    for (std::size_t index = 0; index < turns.size(); ++index) {
        ninfer::ChatMessage message;
        const std::size_t turn = leading_system ? index - 1U : index;
        message.role           = leading_system && index == 0 ? ninfer::ChatRole::System
                                 : turn % 2 == 0              ? ninfer::ChatRole::User
                                                              : ninfer::ChatRole::Assistant;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = turns[index], .media = {}});
        prompt.messages.push_back(std::move(message));
    }
    prompt.options.enable_thinking                 = false;
    prompt.context_cache.automatic_private_anchors = automatic_anchors;
    return prompt;
}

// Chat Completions and Anthropic requests carry no PrivateLongAnchor marker, so the serve layer
// stamps automatic anchors on every prompt. An edit below the rewrite checkpoint must then restore
// at the anchor under the edit; with automatic anchors off the same edit has no such candidate.
// A second phase interleaves two growing conversations behind one common system prompt on one lane
// with a one-slot Device state pool and shared prefixes enabled. The automatic anchor after the
// system message then lands on the structural shared-prefix frontier and aliases its StateImage,
// anchors are demoted to Host, and a consumed endpoint carries an anchor another owner also
// references - the shape that latched the engine through a miscounted active entitlement.
int exercise_automatic_private_anchors(const char* artifact) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    const std::vector<std::string> original{
        "Describe the layout of a small workshop with a bench, a lathe and a drill press.",
        "The bench runs along the north wall, the lathe sits under the window and the drill "
        "press stands by the door.",
        "Where should the dust extractor go so it reaches all three machines?",
        "Put it in the north-east corner with a manifold running to each machine.",
        "How long should the ducting be?"};
    std::vector<std::string> edited = original;
    edited[3] = "Mount it on the ceiling above the lathe and drop hoses to each machine.";

    const auto edit_reuse = [&](std::uint32_t automatic_anchors, ninfer::GenerationResult& out) {
        ninfer::Engine engine(automatic_anchor_engine_options(artifact, 4, 4, 0));
        const ninfer::GenerationResult first = engine.generate(
            engine.prepare(automatic_anchor_conversation(original, automatic_anchors)), request);
        if (first.generated_token_ids.size() != 1) { return false; }
        out = engine.generate(
            engine.prepare(automatic_anchor_conversation(edited, automatic_anchors)), request);
        return out.generated_token_ids.size() == 1;
    };
    ninfer::GenerationResult anchored;
    ninfer::GenerationResult control;
    if (!edit_reuse(2, anchored) || !edit_reuse(0, control)) {
        std::cerr << "automatic-anchor conversation did not generate\n";
        return 1;
    }
    if (anchored.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        anchored.reused_prompt_tokens == 0 ||
        anchored.reused_prompt_tokens >= anchored.prompt.prompt_tokens ||
        control.reused_prompt_tokens >= anchored.reused_prompt_tokens) {
        std::cerr << "automatic long anchor did not restore below the edit: anchored_path="
                  << static_cast<int>(anchored.prefix_reuse_path)
                  << " anchored_reused=" << anchored.reused_prompt_tokens
                  << " control_path=" << static_cast<int>(control.prefix_reuse_path)
                  << " control_reused=" << control.reused_prompt_tokens
                  << " prompt=" << anchored.prompt.prompt_tokens << '\n';
        return 1;
    }

    ninfer::Engine engine(automatic_anchor_engine_options(artifact, 1, 4, 4));
    std::string system =
        "You are a careful engineering assistant. Follow the house style: short sentences, units "
        "on every number, and name the assumption behind every estimate.";
    for (int line = 0; line < 12; ++line) {
        system += " Rule " + std::to_string(line) +
                  ": prefer measured figures to recalled ones, and say which you used.";
    }
    std::array<std::vector<std::string>, 2> conversations{
        std::vector<std::string>{system,
                                 "Plan a three-day walking route through the lake district."},
        std::vector<std::string>{system,
                                 "Summarise the trade-offs between paged and contiguous KV."}};
    for (std::uint32_t round = 0; round < 5; ++round) {
        for (std::size_t lineage = 0; lineage < conversations.size(); ++lineage) {
            std::vector<std::string>& turns = conversations[lineage];
            const ninfer::GenerationResult result = engine.generate(
                engine.prepare(automatic_anchor_conversation(turns, 2, true)), request);
            if (result.generated_token_ids.size() != 1 || !engine.is_available()) {
                std::cerr << "interleaved automatic-anchor conversation failed: round=" << round
                          << " lineage=" << lineage << '\n';
                return 1;
            }
            turns.push_back("Answer " + std::to_string(round) + " for lineage " +
                            std::to_string(lineage) + ", with enough detail to fill a page.");
            turns.push_back("Continue with part " + std::to_string(round + 1) + ".");
        }
    }
    return 0;
}

// Conversations recorded from a ninfer-serve run against qwen3_8_27b: one large conversation and
// three small ones with different system prompts, each reply replayed verbatim as the next turn's
// assistant message, visited round-robin. A client that replays a reply exactly is what makes every
// follow-up reuse the previous endpoint; replies produced by another artifact would not match, so
// the scenario checks that it reached endpoint reuse instead of passing without exercising the bug.
struct RecordedTurn {
    ninfer::ChatRole role;
    std::string_view text;
};

const std::vector<RecordedTurn>& recorded_lineage_0() {
    static const std::vector<RecordedTurn> turns{
        {ninfer::ChatRole::System, R"FX(You are the big agent. Answer tersely.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 0 visit 0. Review this log and summarise the faults in one line.
impeller nut engine quill housing idler fulcrum hinge trunnion bracket nozzle gasket idler bearing bracket eccentric.
pawl anchor fulcrum rotor orifice outrigger gasket union bearing bearing bearing trunnion linkage anchor bracket nozzle.
eccentric bearing knuckle orifice fulcrum idler mandrel orifice washer orifice orifice governor spindle bearing detent mandrel.
trunnion gasket lever sprocket spindle housing valve jib eccentric jib manifold tappet spindle outrigger idler jib.
coupling outrigger cable hinge piston coupling detent lever yoke mandrel yoke flange fulcrum jib gasket keeper.
knuckle coupling yoke idler bearing hinge cable tappet ratchet outrigger outrigger coupling trunnion keeper keeper jib.
orifice anchor manifold linkage mandrel orifice coupling jib washer nut washer governor rotor mandrel pawl anchor.
bracket jib impeller knuckle mandrel nozzle eccentric damper hinge yoke nut mandrel manifold jib detent idler.
washer detent washer anchor linkage linkage ratchet ratchet valve governor pawl bearing orifice sprocket lever mandrel.
outrigger lever flange mandrel quill cable engine flange bearing fulcrum anchor rotor piston rotor housing ratchet.
lever washer spindle engine keeper keeper quill knuckle keeper rotor trunnion spindle governor union idler hinge.
housing bearing tappet bracket valve detent manifold quill gasket quill jib nozzle pawl eccentric bearing orifice.
bearing coupling journal cable keeper fulcrum jib eccentric linkage orifice sprocket knuckle fulcrum orifice knuckle trunnion.
bearing coupling nut union sprocket eccentric damper tappet impeller nozzle damper tappet engine engine tappet tappet.
keeper detent nut quill impeller anchor mandrel cable outrigger nozzle nut governor keeper ratchet jib cable.
bracket manifold washer gasket nozzle nut eccentric outrigger manifold idler gasket bracket spindle jib idler bearing.
union ratchet coupling spindle bearing keeper manifold union nut impeller valve eccentric nozzle rotor gasket bracket.
mandrel washer linkage idler linkage piston engine cable flange impeller keeper keeper linkage nozzle rotor valve.
pawl jib quill yoke valve valve housing spindle piston pawl idler impeller.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log contains no faults, as it consists solely of a list of mechanical component)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 0 visit 1. Review this log and summarise the faults in one line.
outrigger mandrel gasket union cable detent engine bracket journal impeller valve housing ratchet outrigger bracket engine.
nut mandrel orifice nut flange rotor yoke spindle nut linkage housing governor rotor gasket cable spindle.
anchor ratchet anchor flange detent housing cable manifold.)FX"},
    };
    return turns;
}

const std::vector<RecordedTurn>& recorded_lineage_1() {
    static const std::vector<RecordedTurn> turns{
        {ninfer::ChatRole::System, R"FX(You are small agent 1. Tersely.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 1 visit 0. Review this log and summarise the faults in one line.
journal governor governor lever coupling washer eccentric jib housing linkage housing flange governor quill damper trunnion.
nozzle valve orifice tappet nozzle lever journal manifold washer yoke sprocket detent nozzle coupling governor mandrel.
rotor bracket keeper trunnion sprocket housing lever anchor pawl coupling journal nut keeper manifold keeper bearing.
piston fulcrum.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log lists only component names without any recorded errors, failures, or operational data)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 1 visit 1. Review this log and summarise the faults in one line.
sprocket bracket impeller ratchet mandrel damper pawl piston ratchet jib spindle outrigger valve knuckle valve linkage.
yoke hinge coupling damper.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log contains only a list of component names with no recorded faults or errors.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 1 visit 2. Review this log and summarise the faults in one line.
gasket tappet trunnion eccentric quill piston eccentric jib idler outrigger quill journal damper outrigger union keeper.
damper rotor housing damper.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log lists only component names with no recorded faults or errors.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 1 visit 3. Review this log and summarise the faults in one line.
pawl orifice tappet knuckle ratchet nozzle trunnion nut washer valve governor anchor housing hinge manifold knuckle.
housing pawl nozzle cable.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log lists only component names with no recorded faults or errors.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 1 visit 4. Review this log and summarise the faults in one line.
bracket gasket coupling journal ratchet trunnion hinge linkage cable fulcrum keeper damper hinge damper manifold piston.
keeper gasket cable impeller.)FX"},
    };
    return turns;
}

const std::vector<RecordedTurn>& recorded_lineage_2() {
    static const std::vector<RecordedTurn> turns{
        {ninfer::ChatRole::System, R"FX(You are small agent 2. Tersely.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 2 visit 0. Review this log and summarise the faults in one line.
outrigger manifold linkage washer governor damper jib nozzle pawl orifice spindle idler nozzle valve fulcrum engine.
quill manifold keeper flange fulcrum linkage yoke manifold hinge union piston sprocket coupling eccentric nozzle engine.
eccentric governor tappet nozzle housing trunnion ratchet valve rotor coupling bracket orifice eccentric engine coupling detent.
anchor governor.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log contains no faults, only a list of mechanical component names.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 2 visit 1. Review this log and summarise the faults in one line.
nozzle linkage gasket washer bearing yoke yoke anchor quill nut journal cable eccentric lever rotor engine.
journal coupling yoke trunnion.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log contains no faults, only a list of mechanical component names.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 2 visit 2. Review this log and summarise the faults in one line.
nut hinge keeper idler jib tappet lever knuckle anchor flange engine cable nut washer linkage fulcrum.
housing ratchet pawl governor.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log contains no faults, only a list of mechanical component names.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 2 visit 3. Review this log and summarise the faults in one line.
sprocket impeller lever manifold mandrel fulcrum impeller flange journal tappet valve bracket engine damper mandrel mandrel.
washer governor damper manifold.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(The log contains no faults, only a list of mechanical component names.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 2 visit 4. Review this log and summarise the faults in one line.
rotor union flange rotor journal quill governor nozzle eccentric housing knuckle union piston gasket rotor trunnion.
lever anchor anchor sprocket.)FX"},
    };
    return turns;
}

const std::vector<RecordedTurn>& recorded_lineage_3() {
    static const std::vector<RecordedTurn> turns{
        {ninfer::ChatRole::System, R"FX(You are small agent 3. Tersely.)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 3 visit 0. Review this log and summarise the faults in one line.
journal ratchet washer keeper mandrel pawl lever bracket spindle ratchet eccentric nut outrigger cable knuckle linkage.
ratchet knuckle lever bracket fulcrum orifice jib impeller flange damper lever pawl nut fulcrum jib orifice.
knuckle knuckle journal idler ratchet detent sprocket hinge engine piston ratchet flange valve bracket manifold jib.
eccentric pawl.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(Recurring faults involve ratchets, levers, brackets, and knuckles)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 3 visit 1. Review this log and summarise the faults in one line.
eccentric coupling impeller keeper yoke nozzle engine outrigger piston sprocket spindle eccentric impeller mandrel ratchet fulcrum.
rotor union sprocket hinge.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(Faults center on eccentrics, impellers, sprockets, and)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 3 visit 2. Review this log and summarise the faults in one line.
sprocket linkage fulcrum sprocket jib nozzle coupling piston nozzle pawl outrigger linkage ratchet keeper linkage yoke.
knuckle damper coupling housing.)FX"},
        {ninfer::ChatRole::Assistant, R"FX(Faults involve sprockets, linkages, nozzles)FX"},
        {ninfer::ChatRole::User, R"FX(Agent 3 visit 3. Review this log and summarise the faults in one line.
hinge lever tappet engine knuckle bracket quill detent lever nozzle union fulcrum manifold keeper orifice knuckle.
detent impeller journal pawl.)FX"},
    };
    return turns;
}

// (lineage, number of leading turns sent), in request order.
constexpr std::array<std::pair<std::uint8_t, std::uint8_t>, 16> kRecordedSchedule{{
    {0, 2}, {1, 2}, {2, 2}, {3, 2}, {1, 4}, {2, 4}, {3, 4}, {1, 6}, {2, 6}, {3, 6}, {0, 4}, {1, 8}, {2, 8}, {3, 8}, {1, 10}, {2, 10}
}};

// A consumed endpoint carries its lineage's long anchors with it. An anchor that aliases another
// owner's StateImage (the automatic anchor after the leading system message lands on the structural
// shared-prefix frontier) is not exclusive when the request is planned; if the pressure target then
// evicts that owner, the anchor becomes exclusive to the consumed lineage and must be part of its
// active entitlement. The projection missed it, so the committed sequence owned more StateImages
// than it had reserved and the Engine reported a worker failure ("materialized sequence does not
// match its active entitlement" / "sequence StateImage entitlement is inconsistent"), dropping the
// whole context cache with it. The recorded visits run under a tiny state and Host KV budget, so
// every follow-up consumes an endpoint while its neighbours are demoted or evicted.
int exercise_endpoint_adopts_unaliased_anchor(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 2048;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.kv_cache                             = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.max_concurrency                      = 2;
    options.max_pending_requests                 = 2;
    options.context_cache.device_state_slots     = 2;
    options.context_cache.host_state_slots       = 6;
    options.context_cache.host_kv_capacity_bytes = std::size_t{8} << 20U;
    options.context_cache.max_private_continuations         = 6;
    options.context_cache.max_shared_prefixes               = 4;
    options.context_cache.max_long_anchors_per_continuation = 2;
    ninfer::Engine engine(std::move(options));

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 16;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    const std::array<const std::vector<RecordedTurn>*, 4> lineages{
        &recorded_lineage_0(), &recorded_lineage_1(), &recorded_lineage_2(), &recorded_lineage_3()};
    std::uint32_t endpoint_reuses = 0;
    for (std::size_t number = 0; number < kRecordedSchedule.size(); ++number) {
        const auto [lineage, turn_count] = kRecordedSchedule[number];
        ninfer::PromptInput prompt;
        for (std::size_t turn = 0; turn < turn_count; ++turn) {
            ninfer::ChatMessage message;
            message.role = (*lineages[lineage])[turn].role;
            message.parts.push_back(ninfer::MessagePart{
                .kind  = ninfer::MessagePartKind::Text,
                .text  = std::string((*lineages[lineage])[turn].text),
                .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking                 = false;
        prompt.context_cache.automatic_private_anchors = 2;
        try {
            const ninfer::GenerationResult result =
                engine.generate(engine.prepare(std::move(prompt)), request);
            if (result.prefix_reuse_path == ninfer::PrefixReusePath::PrivateEndpoint) {
                ++endpoint_reuses;
            }
        } catch (const std::exception& error) {
            std::cerr << "anchor-adoption request " << number + 1U << " of "
                      << kRecordedSchedule.size() << " failed: " << error.what() << '\n';
            return 1;
        }
        if (!engine.is_available()) {
            std::cerr << "Engine became unavailable at anchor-adoption request " << number + 1U
                      << '\n';
            return 1;
        }
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    if (endpoint_reuses < 8 ||
        stats.pressure_private_owners_degraded + stats.pressure_private_owners_evicted == 0) {
        std::cerr << "anchor-adoption scenario did not exercise endpoint reuse under pressure: "
                     "endpoint_reuses="
                  << endpoint_reuses << " private_degraded=" << stats.pressure_private_owners_degraded
                  << " private_evicted=" << stats.pressure_private_owners_evicted
                  << " (the recorded replies belong to qwen3_8_27b)\n";
        return 1;
    }
    return 0;
}

// Cancels the request once its prefill has reached `cancel_at` prompt tokens.
class CancelAtProgressSink final : public ninfer::OutputSink {
public:
    explicit CancelAtProgressSink(std::uint32_t cancel_at) : cancel_at_(cancel_at) {}

    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress progress) override {
        if (progress.processed_prompt_tokens + progress.reused_prompt_tokens >= cancel_at_) {
            reached_.store(true, std::memory_order_release);
        }
    }
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta) override {}

    [[nodiscard]] bool reached() const noexcept { return reached_.load(std::memory_order_acquire); }

private:
    std::uint32_t cancel_at_;
    std::atomic<bool> reached_{false};
};

ninfer::PromptInput progress_anchor_prompt(std::uint32_t stride) {
    // One long user message: no interior message boundary exists, so only the stride gives the
    // prefill anywhere to keep its progress.
    std::string text = "Read the following log and answer the question at the end.";
    for (int line = 0; line < 160; ++line) {
        text += " Entry " + std::to_string(line) + ": the pump on line " +
                std::to_string(line % 17) + " reported " + std::to_string(line * 37 % 1013) +
                " kPa and the operator noted nothing unusual.";
    }
    text += " Question: which line reported the highest pressure?";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
    ninfer::PromptInput prompt;
    prompt.messages.push_back(std::move(message));
    prompt.options.enable_thinking             = false;
    prompt.context_cache.progress_anchor_stride = stride;
    return prompt;
}

// A client that times out part way through a very long prompt and retries must not start over.
// The cancelled prefill publishes the anchors it had captured, the identical retry restores the
// deepest one, and greedy output matches a request that was never cancelled. With progress anchors
// off the same cancel leaves nothing behind, which pins that the saving comes from this feature.
int exercise_cancelled_prefill_progress(const char* artifact) {
    constexpr std::uint32_t kStride = 512;
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 8;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    const auto options = [&] {
        ninfer::EngineOptions engine_options = automatic_anchor_engine_options(artifact, 4, 4, 0);
        engine_options.max_context           = 8192;
        engine_options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        return engine_options;
    };

    ninfer::GenerationResult control;
    {
        ninfer::Engine engine(options());
        control = engine.generate(engine.prepare(progress_anchor_prompt(0)), request);
    }
    const std::uint32_t prompt_tokens = control.prompt.prompt_tokens;
    if (control.generated_token_ids.size() != 8 || prompt_tokens < 6U * kStride) {
        std::cerr << "progress-anchor prompt is too short or did not generate: tokens="
                  << prompt_tokens << '\n';
        return 1;
    }
    const std::uint32_t cancel_at = prompt_tokens * 2U / 5U;

    const auto cancel_then_retry = [&](std::uint32_t stride, ninfer::GenerationResult& retried,
                                       ninfer::FinishReason& cancelled_reason,
                                       ninfer::RuntimeStats& stats) {
        ninfer::Engine engine(options());
        CancelAtProgressSink sink(cancel_at);
        ninfer::GenerationHandle handle = engine.submit(
            engine.prepare(progress_anchor_prompt(stride)), request,
            ninfer::OutputConsumerMode::Streaming,
            ninfer::GenerationObservationOptions{.prompt_progress = true});
        const ninfer::GenerationResult cancelled =
            handle.wait(&sink, ninfer::CancellationView([&sink] { return sink.reached(); }));
        cancelled_reason = cancelled.finish_reason;
        retried = engine.generate(engine.prepare(progress_anchor_prompt(stride)), request);
        stats   = engine.runtime_stats();
        return engine.is_available();
    };

    ninfer::GenerationResult kept;
    ninfer::GenerationResult dropped;
    ninfer::FinishReason kept_reason    = ninfer::FinishReason::None;
    ninfer::FinishReason dropped_reason = ninfer::FinishReason::None;
    ninfer::RuntimeStats kept_stats;
    ninfer::RuntimeStats dropped_stats;
    if (!cancel_then_retry(kStride, kept, kept_reason, kept_stats) ||
        !cancel_then_retry(0, dropped, dropped_reason, dropped_stats)) {
        std::cerr << "engine latched unavailable after a cancelled prefill\n";
        return 1;
    }
    if (kept_reason != ninfer::FinishReason::Cancelled ||
        dropped_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "prefill finished before the cancel landed; the prompt is too short for this "
                     "GPU: kept="
                  << static_cast<int>(kept_reason) << " dropped=" << static_cast<int>(dropped_reason)
                  << '\n';
        return 1;
    }
    if (kept.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        kept.reused_prompt_tokens < 2U * kStride || kept.reused_prompt_tokens % kStride != 0 ||
        kept.reused_prompt_tokens >= prompt_tokens || kept.reused_prompt_tokens > cancel_at + 2048U) {
        std::cerr << "retry did not resume from a progress anchor: path="
                  << static_cast<int>(kept.prefix_reuse_path)
                  << " reused=" << kept.reused_prompt_tokens << " prompt=" << prompt_tokens
                  << " cancel_at=" << cancel_at << '\n';
        return 1;
    }
    // The counters tell the same story as the retry: both prefills were cancelled part way, only
    // the one with progress anchors kept a checkpoint, and it is at least two strides deep.
    if (kept_stats.cancelled_prefills != 1 || kept_stats.cancelled_prefills_retained != 1 ||
        kept_stats.cancelled_prefill_retained_tokens < 2U * kStride ||
        kept_stats.cancelled_prefill_computed_tokens < cancel_at ||
        dropped_stats.cancelled_prefills != 1 || dropped_stats.cancelled_prefills_retained != 0 ||
        dropped_stats.cancelled_prefill_retained_tokens != 0) {
        std::cerr << "cancelled-prefill counters disagree with the retry: kept{cancelled="
                  << kept_stats.cancelled_prefills << " retained=" << kept_stats.cancelled_prefills_retained
                  << " tokens=" << kept_stats.cancelled_prefill_retained_tokens
                  << " computed=" << kept_stats.cancelled_prefill_computed_tokens << "} dropped{cancelled="
                  << dropped_stats.cancelled_prefills << " retained=" << dropped_stats.cancelled_prefills_retained
                  << "}\n";
        return 1;
    }
    if (dropped.reused_prompt_tokens != 0) {
        std::cerr << "a cancelled prefill without progress anchors left reusable state: reused="
                  << dropped.reused_prompt_tokens << '\n';
        return 1;
    }
    if (kept.generated_token_ids != control.generated_token_ids ||
        dropped.generated_token_ids != control.generated_token_ids) {
        std::cerr << "output after a cancelled prefill differs from an uncancelled request\n";
        return 1;
    }
    std::cout << "cancelled prefill kept " << kept.reused_prompt_tokens << " of " << prompt_tokens
              << " prompt tokens (cancel at " << cancel_at << ")\n";
    return 0;
}

ninfer::EngineOptions store_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 1024;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 2;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = std::size_t{64} << 20U;
    // One private cell, so every unrelated conversation evicts the retained one.
    options.context_cache.max_private_continuations         = 1;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 2;
    return options;
}

ninfer::PromptInput store_conversation(const std::vector<std::string>& turns) {
    ninfer::PromptInput prompt;
    for (std::size_t index = 0; index < turns.size(); ++index) {
        ninfer::ChatMessage message;
        message.role = index % 2 == 0 ? ninfer::ChatRole::User : ninfer::ChatRole::Assistant;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = turns[index], .media = {}});
        prompt.messages.push_back(std::move(message));
    }
    prompt.options.enable_thinking = false;
    return prompt;
}

// A session evicted from the cache while the Engine keeps running is read back from the context
// store when a request continues it, instead of being prefilled again. The prompt is long enough
// (well past the minimum gain) for the read to pay; the continuation must reuse the stored prefix,
// count one hydration, and produce exactly what a session that never left the device produces.
int exercise_store_hydration(const char* artifact) {
    namespace fs = std::filesystem;
    const fs::path directory = fs::temp_directory_path() / "ninfer-store-hydration-test";
    fs::remove_all(directory);
    fs::create_directories(directory);
    // Declared before the Engine, so it removes the store only after the Engine has shut down.
    struct DirectoryCleanup {
        fs::path path;
        ~DirectoryCleanup() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    } const cleanup{directory};

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 12;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    std::string log = "Read the following log and answer the question at the end.";
    for (int line = 0; line < 300; ++line) {
        log += " Entry " + std::to_string(line) + ": the pump on line " +
               std::to_string(line % 17) + " reported " + std::to_string(line * 37 % 1013) +
               " kPa and the operator noted nothing unusual.";
    }
    const std::vector<std::string> first{log};
    const std::vector<std::string> unrelated{
        "Describe how a cooling fan bearing wears out over several years of service."};
    const auto second_turn = [&](const ninfer::GenerationResult& reply) {
        std::vector<std::string> turns = first;
        turns.push_back(reply.content);
        turns.push_back("Which line reported the highest pressure?");
        return turns;
    };

    // A context deep enough that the stored prefix is far past the minimum worth reading back, and a
    // host tier large enough to hold the KV of one such session.
    const auto deep_options = [&] {
        ninfer::EngineOptions options = store_engine_options(artifact);
        options.max_context           = 16384;
        options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(16384);
        options.context_cache.host_kv_capacity_bytes = std::size_t{512} << 20U;
        return options;
    };

    std::vector<ninfer::TokenId> control_tokens;
    std::uint32_t control_reused = 0;
    {
        ninfer::Engine control(deep_options());
        const ninfer::GenerationResult reply =
            control.generate(control.prepare(store_conversation(first)), request);
        const ninfer::GenerationResult next =
            control.generate(control.prepare(store_conversation(second_turn(reply))), request);
        control_tokens = next.generated_token_ids;
        control_reused = next.reused_prompt_tokens;
        if (control_reused < 4096) {
            std::cerr << "the control prompt is too short to exercise a hydration: reused "
                      << control_reused << '\n';
            return 1;
        }
    }

    ninfer::EngineOptions options = deep_options();
    options.context_store.directory    = directory;
    options.context_store.idle_persist = std::chrono::seconds(0);
    ninfer::Engine engine(std::move(options));
    const ninfer::GenerationResult reply =
        engine.generate(engine.prepare(store_conversation(first)), request);
    // One private cell: an unrelated conversation evicts the first, which writes it to the store.
    (void)engine.generate(engine.prepare(store_conversation(unrelated)), request);
    bool written = false;
    for (int attempt = 0; attempt < 600 && !written; ++attempt) {
        written = engine.runtime_stats().context_store_writes >= 1;
        if (!written) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
    }
    if (!written) {
        std::cerr << "the evicted session was not written to the store\n";
        return 1;
    }
    const ninfer::GenerationResult next =
        engine.generate(engine.prepare(store_conversation(second_turn(reply))), request);
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    if (stats.context_store_hydrations != 1 || stats.context_store_hydrated_tokens < 4096 ||
        stats.context_store_hydration_failures != 0 || stats.context_store_hydration_seconds <= 0.0) {
        std::cerr << "the continuation did not hydrate the stored session: hydrations="
                  << stats.context_store_hydrations
                  << " tokens=" << stats.context_store_hydrated_tokens
                  << " failures=" << stats.context_store_hydration_failures << '\n';
        return 1;
    }
    if (next.reused_prompt_tokens + 64U < control_reused) {
        std::cerr << "the hydrated continuation reused " << next.reused_prompt_tokens
                  << " tokens against " << control_reused << " for the warm control\n";
        return 1;
    }
    if (next.generated_token_ids != control_tokens) {
        std::cerr << "output after a hydration differs from the warm control\n";
        return 1;
    }
    std::cout << "hydrated " << stats.context_store_hydrated_tokens << " tokens in "
              << stats.context_store_hydration_seconds << " s\n";
    return 0;
}

// The durable context store: a retained session survives an Engine restart without any explicit
// save or restore call, and a damaged store costs cache hits, never correctness. Four properties on
// the real artifact, each against its own store directory:
//  - shutdown writes every retained session; a fresh Engine restores it at start-up, the
//    continuation reuses the restored prefix, and greedy output matches a control that never left
//    the device;
//  - an unrelated conversation evicting the only private cell writes the evicted session;
//  - a session idle for the configured interval is written in the background while the Engine
//    runs, so a crash would lose at most that interval;
//  - a store whose chunk has been damaged is not trusted: the session is dropped, counted, and the
//    next request prefills from scratch.
int exercise_context_store(const char* artifact) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "ninfer-context-store-test";
    fs::remove_all(root);
    fs::create_directories(root);

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 12;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    const std::vector<std::string> first{
        "List three uses for a lathe in a small workshop, one line each."};
    const std::vector<std::string> unrelated{
        "Describe how a cooling fan bearing wears out over several years of service."};
    const auto second_turn = [&](const ninfer::GenerationResult& reply) {
        std::vector<std::string> turns = first;
        turns.push_back(reply.content);
        turns.push_back("Which of those needs the most care with tool speed?");
        return turns;
    };
    const auto store_options = [&](const fs::path& directory, std::chrono::seconds idle) {
        ninfer::EngineOptions options = store_engine_options(artifact);
        options.context_store.directory    = directory;
        options.context_store.idle_persist = idle;
        return options;
    };
    const auto wait_for = [](auto&& condition) {
        for (int attempt = 0; attempt < 400; ++attempt) {
            if (condition()) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return condition();
    };

    // Control: the warm session continues without ever leaving the device.
    std::vector<ninfer::TokenId> control_tokens;
    {
        ninfer::Engine control(store_engine_options(artifact));
        const ninfer::GenerationResult reply =
            control.generate(control.prepare(store_conversation(first)), request);
        const ninfer::GenerationResult next =
            control.generate(control.prepare(store_conversation(second_turn(reply))), request);
        control_tokens = next.generated_token_ids;
        if (next.reused_prompt_tokens == 0) {
            std::cerr << "control continuation did not reuse its warm session\n";
            return 1;
        }
    }

    // Shutdown flush, then restore at start-up.
    const fs::path restart_dir = root / "restart";
    ninfer::GenerationResult reply;
    {
        ninfer::Engine engine(store_options(restart_dir, std::chrono::seconds(0)));
        reply = engine.generate(engine.prepare(store_conversation(first)), request);
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.context_store_writes != 0 || stats.context_store_restored != 0) {
            std::cerr << "the store was written before any session left the cache\n";
            return 1;
        }
    }
    {
        ninfer::Engine engine(store_options(restart_dir, std::chrono::seconds(0)));
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        const auto states                = engine.slot_states();
        if (stats.context_store_restored != 1 || states.size() != 1 || !states[0].retained ||
            states[0].session_digest != reply.session_digest) {
            std::cerr << "shutdown did not persist the session or start-up did not restore it: "
                         "restored="
                      << stats.context_store_restored << " images=" << stats.context_store_images
                      << '\n';
            return 1;
        }
        const ninfer::GenerationResult next =
            engine.generate(engine.prepare(store_conversation(second_turn(reply))), request);
        if (next.reused_prompt_tokens == 0 ||
            next.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
            std::cerr << "the continuation did not reuse the restored session\n";
            return 1;
        }
        if (next.generated_token_ids != control_tokens) {
            std::cerr << "output after a restart differs from the warm control\n";
            return 1;
        }
    }

    // Eviction writes: one private cell, so an unrelated conversation evicts the first.
    {
        ninfer::Engine engine(store_options(root / "evict", std::chrono::seconds(0)));
        (void)engine.generate(engine.prepare(store_conversation(first)), request);
        (void)engine.generate(engine.prepare(store_conversation(unrelated)), request);
        if (!wait_for([&] { return engine.runtime_stats().context_store_writes >= 1; })) {
            std::cerr << "an evicted session was not written to the store\n";
            return 1;
        }
    }

    // Background write of an idle session while the Engine keeps running.
    {
        ninfer::Engine engine(store_options(root / "idle", std::chrono::seconds(1)));
        (void)engine.generate(engine.prepare(store_conversation(first)), request);
        if (!wait_for([&] { return engine.runtime_stats().context_store_writes >= 1; })) {
            std::cerr << "an idle session was not written in the background\n";
            return 1;
        }
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.context_store_images != 1 || stats.context_store_used_bytes == 0) {
            std::cerr << "the background write is not visible in the store statistics\n";
            return 1;
        }
    }

    // A damaged chunk: the session is dropped and counted, never restored.
    {
        fs::path chunk;
        for (const auto& item : fs::recursive_directory_iterator(restart_dir / "chunks")) {
            if (item.is_regular_file()) {
                chunk = item.path();
                break;
            }
        }
        if (chunk.empty()) {
            std::cerr << "the store holds no chunks to damage\n";
            return 1;
        }
        {
            std::fstream file(chunk, std::ios::in | std::ios::out | std::ios::binary);
            char byte = 0;
            file.read(&byte, 1);
            file.seekp(0);
            byte ^= 0x5a;
            file.write(&byte, 1);
        }
        ninfer::Engine engine(store_options(restart_dir, std::chrono::seconds(0)));
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.context_store_restored != 0 || stats.context_store_corrupt != 1) {
            std::cerr << "a damaged store was trusted: restored=" << stats.context_store_restored
                      << " corrupt=" << stats.context_store_corrupt << '\n';
            return 1;
        }
        const ninfer::GenerationResult next =
            engine.generate(engine.prepare(store_conversation(second_turn(reply))), request);
        if (next.reused_prompt_tokens != 0 || next.generated_token_ids.empty()) {
            std::cerr << "a request after a damaged store did not prefill from scratch\n";
            return 1;
        }
    }
    fs::remove_all(root);
    return 0;
}

// A host-side failure in the worker must fail only the request it touched, keep the FIFO queue,
// and leave the Engine serving; the third consecutive failure latches it. Failures are armed
// through the worker-fault seam and thrown after a prefill unit has executed, so each recovery
// releases a lane holding live KV pages and state. The conversations share a system prompt and
// place long anchors on it under Host pressure, so the cache the recovery clears is non-trivial.
// With NINFER_TEST_GRAFT set, the startup-pinned graft the recovery also releases must be
// reinstalled: a grafted request after the failures starts from the graft, as before them.
int exercise_worker_failure_recovery(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 1024;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 4;
    options.context_cache.host_kv_capacity_bytes = std::size_t{64} << 20U;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 4;
    options.context_cache.max_long_anchors_per_continuation = 2;
    ninfer::test::add_test_graft(options);
    ninfer::Engine engine(std::move(options));

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    const auto grafted = [&]() -> std::optional<ninfer::GenerationResult> {
        if (!ninfer::test::graft_configured()) { return std::nullopt; }
        ninfer::PromptInput input;
        input.messages.push_back(ninfer::ChatMessage{
            .role = ninfer::ChatRole::User, .parts = {ninfer::MessagePart{.text = "Who are you?"}}});
        input.options.graft = "g";
        ninfer::RequestOptions graft_request = request;
        graft_request.execution.requested_output_tokens = 8;
        return engine.generate(engine.prepare(std::move(input)), graft_request);
    };
    const std::optional<ninfer::GenerationResult> graft_before = grafted();
    if (graft_before && graft_before->reused_prompt_tokens == 0) {
        std::cerr << "grafted request did not start from the graft\n";
        return 1;
    }

    std::string system =
        "You are a careful engineering assistant. Follow the house style: short sentences, units "
        "on every number, and name the assumption behind every estimate.";
    for (int line = 0; line < 12; ++line) {
        system += " Rule " + std::to_string(line) +
                  ": prefer measured figures to recalled ones, and say which you used.";
    }
    const auto conversation = [&](const std::vector<std::string>& turns) {
        ninfer::PromptInput prompt;
        ninfer::ChatMessage head;
        head.role = ninfer::ChatRole::System;
        head.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = system, .media = {}});
        prompt.messages.push_back(std::move(head));
        for (std::size_t index = 0; index < turns.size(); ++index) {
            ninfer::ChatMessage message;
            message.role = index % 2 == 0 ? ninfer::ChatRole::User : ninfer::ChatRole::Assistant;
            message.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = turns[index], .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking = false;
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = 1,
            .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
            .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
        });
        return prompt;
    };

    enum class Outcome { Served, Failed, Unavailable, Malformed };
    const auto wait_for = [](ninfer::GenerationHandle& handle) {
        try {
            const ninfer::GenerationResult result = handle.wait();
            return result.generated_token_ids.size() == 1 ? Outcome::Served : Outcome::Malformed;
        } catch (const ninfer::RequestError& error) {
            return error.kind() == ninfer::RequestErrorKind::Unavailable ? Outcome::Unavailable
                                                                         : Outcome::Failed;
        } catch (const std::exception&) { return Outcome::Failed; }
    };

    std::array<std::vector<std::string>, 2> turns{
        std::vector<std::string>{"Plan a three-day walking route through the lake district."},
        std::vector<std::string>{"Summarise the trade-offs between paged and contiguous KV."}};
    const auto grow = [&](std::uint32_t round) {
        for (std::size_t lineage = 0; lineage < turns.size(); ++lineage) {
            turns[lineage].push_back("Answer " + std::to_string(round) + " for lineage " +
                                     std::to_string(lineage) + ", with enough detail to fill a page.");
            turns[lineage].push_back("Continue with part " + std::to_string(round + 1) + ".");
        }
    };

    // Recovery: one failure per round, with the second conversation queued behind it. Three
    // rounds pass the latch threshold only because each served request resets the streak.
    constexpr std::uint32_t kRecoveryRounds = 3;
    for (std::uint32_t round = 0; round < kRecoveryRounds; ++round) {
        ninfer::runtime::arm_worker_failures(1);
        ninfer::GenerationHandle first =
            engine.submit(engine.prepare(conversation(turns[0])), request);
        ninfer::GenerationHandle second =
            engine.submit(engine.prepare(conversation(turns[1])), request);
        const Outcome first_outcome  = wait_for(first);
        const Outcome second_outcome = wait_for(second);
        if (first_outcome != Outcome::Failed || second_outcome != Outcome::Served ||
            !engine.is_available()) {
            ninfer::runtime::arm_worker_failures(0);
            std::cerr << "worker-recovery round " << round
                      << ": first=" << static_cast<int>(first_outcome)
                      << " second=" << static_cast<int>(second_outcome)
                      << " available=" << engine.is_available() << "\n";
            return 1;
        }
        grow(round);
    }
    const std::uint64_t recoveries = engine.runtime_stats().engine_recoveries;
    if (recoveries != kRecoveryRounds) {
        std::cerr << "worker recovery accounting is inconsistent: recoveries=" << recoveries
                  << "\n";
        return 1;
    }
    if (graft_before) {
        const std::optional<ninfer::GenerationResult> graft_after = grafted();
        if (graft_after->reused_prompt_tokens != graft_before->reused_prompt_tokens ||
            graft_after->generated_token_ids != graft_before->generated_token_ids) {
            std::cerr << "the pinned graft was not reinstalled by recovery: reused before="
                      << graft_before->reused_prompt_tokens
                      << " after=" << graft_after->reused_prompt_tokens << "\n";
            return 1;
        }
    }

    // Latch: three consecutive failures with no served request between them. The first two
    // recover; the third latches, and the Engine then refuses work.
    ninfer::runtime::arm_worker_failures(3);
    std::array<Outcome, 3> streak{};
    for (std::size_t index = 0; index < streak.size(); ++index) {
        ninfer::GenerationHandle handle =
            engine.submit(engine.prepare(conversation(turns[index % 2])), request);
        streak[index] = wait_for(handle);
    }
    ninfer::runtime::arm_worker_failures(0);
    const std::uint64_t streak_recoveries = engine.runtime_stats().engine_recoveries - recoveries;
    if (streak[0] != Outcome::Failed || streak[1] != Outcome::Failed ||
        streak[2] == Outcome::Served || streak_recoveries != 2 || engine.is_available()) {
        std::cerr << "consecutive worker failures did not latch on the third: outcomes="
                  << static_cast<int>(streak[0]) << ',' << static_cast<int>(streak[1]) << ','
                  << static_cast<int>(streak[2]) << " recoveries=" << streak_recoveries
                  << " available=" << engine.is_available() << "\n";
        return 1;
    }
    std::cout << "worker-failure-recovery: recoveries=" << recoveries
              << " queued_served=" << kRecoveryRounds << " latched_after=3"
              << " graft_reinstalled=" << (graft_before ? "yes" : "not configured") << "\n";
    return 0;
}

// A request that cannot be planned fails alone. A running request is not touched, the Engine does
// not recover (recovery would fail every running lane and clear the context cache), and the
// failure does not count toward the latch.
int exercise_admission_planning_failure_is_contained(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 1024;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 2;
    options.max_pending_requests                 = 2;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 4;
    options.context_cache.host_kv_capacity_bytes = std::size_t{64} << 20U;
    ninfer::Engine engine(std::move(options));

    const auto conversation = [](std::string text) {
        ninfer::PromptInput prompt;
        prompt.messages.push_back(ninfer::ChatMessage{
            .role  = ninfer::ChatRole::User,
            .parts = {ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                          .text  = std::move(text),
                                          .media = {}}}});
        prompt.options.enable_thinking = false;
        return prompt;
    };
    ninfer::RequestOptions running_request;
    running_request.execution.requested_output_tokens = 200;
    running_request.execution.sampling.temperature    = 0.0F;
    running_request.stop.include_model_defaults       = false;
    ninfer::RequestOptions short_request = running_request;
    short_request.execution.requested_output_tokens = 1;

    ninfer::GenerationHandle running = engine.submit(
        engine.prepare(conversation("Write a long story about a lighthouse keeper.")),
        running_request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (engine.runtime_stats().running_requests == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "the running request never started\n";
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const std::uint64_t recoveries_before = engine.runtime_stats().engine_recoveries;

    ninfer::runtime::arm_planning_failures(1);
    ninfer::GenerationHandle poisoned = engine.submit(
        engine.prepare(conversation("Summarise the trade-offs between paged and contiguous KV.")),
        short_request);
    bool poisoned_failed = false;
    try {
        (void)poisoned.wait();
    } catch (const std::exception&) { poisoned_failed = true; }
    ninfer::runtime::arm_planning_failures(0);

    std::size_t running_tokens = 0;
    bool running_failed        = false;
    try {
        running_tokens = running.wait().generated_token_ids.size();
    } catch (const std::exception&) { running_failed = true; }

    bool served_after = false;
    try {
        served_after = engine
                           .generate(engine.prepare(conversation("Say hello.")), short_request)
                           .generated_token_ids.size() == 1;
    } catch (const std::exception&) {}

    const std::uint64_t recoveries = engine.runtime_stats().engine_recoveries - recoveries_before;
    if (!poisoned_failed || running_failed || running_tokens < 2 || !served_after ||
        recoveries != 0 || !engine.is_available()) {
        std::cerr << "admission planning failure was not contained: poisoned_failed="
                  << poisoned_failed << " running_failed=" << running_failed
                  << " running_tokens=" << running_tokens << " served_after=" << served_after
                  << " recoveries=" << recoveries << " available=" << engine.is_available()
                  << "\n";
        return 1;
    }
    std::cout << "admission-planning-failure: contained, running_tokens=" << running_tokens
              << " recoveries=0\n";
    return 0;
}

int exercise_private_long_anchor_capture_and_replacement(const char* artifact) {
    ninfer::Engine engine(private_long_anchor_engine_options(artifact));

    const auto input = [](std::vector<std::string> turns,
                          std::optional<std::uint32_t> marker_after) {
        ninfer::PromptInput prompt;
        for (std::string& text : turns) {
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            message.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking = false;
        if (marker_after) {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = *marker_after,
                .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        }
        return prompt;
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    constexpr std::string_view stable =
        "This is the stable conversation prefix retained for a later branch.";
    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(input({std::string(stable), "Follow the original branch."}, 1)), request);
    if (source.generated_token_ids.size() != 1 ||
        source.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "first private long-anchor capture did not complete from Root\n";
        return 1;
    }

    ninfer::PromptInput replacement_input =
        input({std::string(stable), "Follow the replacement branch.",
               "This suffix belongs only to the replacement source."},
              2);
    // Name the replacement lineage so the final request selects this continuation rather than
    // legitimately preferring the older anonymous branch when the private catalog is full.
    replacement_input.context_cache.session_key = "private-long-anchor-replacement";
    replacement_input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    const ninfer::GenerationResult replacement =
        engine.generate(engine.prepare(std::move(replacement_input)), request);
    if (replacement.generated_token_ids.size() != 1 ||
        replacement.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        replacement.reused_prompt_tokens == 0 ||
        replacement.reused_prompt_tokens >= replacement.prompt.prompt_tokens) {
        std::cerr << "private long anchor was not selected before full-capacity replacement: path="
                  << static_cast<int>(replacement.prefix_reuse_path)
                  << " reused=" << replacement.reused_prompt_tokens
                  << " prompt=" << replacement.prompt.prompt_tokens << '\n';
        return 1;
    }

    ninfer::PromptInput replaced_input =
        input({std::string(stable), "Follow the replacement branch.",
               "Continue through a different branch suffix."},
              std::nullopt);
    replaced_input.context_cache.session_key = "private-long-anchor-replacement";
    replaced_input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    const ninfer::GenerationResult replaced =
        engine.generate(engine.prepare(std::move(replaced_input)), request);
    if (replaced.generated_token_ids.size() != 1 ||
        replaced.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        replaced.reused_prompt_tokens <= replacement.reused_prompt_tokens ||
        replaced.reused_prompt_tokens >= replaced.prompt.prompt_tokens) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "replacement private long anchor was not reusable: path="
                  << static_cast<int>(replaced.prefix_reuse_path)
                  << " first_reused=" << replacement.reused_prompt_tokens
                  << " replaced_reused=" << replaced.reused_prompt_tokens
                  << " prompt=" << replaced.prompt.prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    return 0;
}

int exercise_last_private_alias_eviction(const char* artifact) {
    ninfer::Engine engine(last_alias_engine_options(artifact));
    std::string prompt_text;
    prompt_text.reserve(6U * 300U + 8U);
    for (std::uint32_t index = 0; index < 300; ++index) { prompt_text += "alpha "; }

    const auto input = [&](std::string session, ninfer::CacheRetentionHint retention) {
        ninfer::PromptInput prompt;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = prompt_text, .media = {}});
        prompt.messages.push_back(std::move(user));
        prompt.options.enable_thinking   = false;
        prompt.context_cache.session_key = std::move(session);
        prompt.context_cache.retention   = retention;
        return prompt;
    };
    if (engine.count_tokens(input("probe", ninfer::CacheRetentionHint::Disposable)) % 64U == 0) {
        prompt_text += "beta";
    }

    ninfer::RequestOptions one_token;
    one_token.execution.requested_output_tokens = 1;
    one_token.execution.sampling.temperature    = 0.0F;
    one_token.execution.allow_prefix_reuse      = true;
    one_token.stop.include_model_defaults       = false;

    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(input("last-alias-source", ninfer::CacheRetentionHint::LiveSession)),
        one_token);
    const ninfer::GenerationResult branch = engine.generate(
        engine.prepare(input("last-alias-branch", ninfer::CacheRetentionHint::Disposable)),
        one_token);
    if (source.generated_token_ids.size() != 1 || branch.generated_token_ids.size() != 1 ||
        branch.reused_prompt_tokens == 0 ||
        (branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint)) {
        std::cerr << "last-alias fixture did not establish two private prefix aliases: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens << '\n';
        return 1;
    }

    ninfer::RequestOptions full_capacity            = one_token;
    full_capacity.execution.requested_output_tokens = std::numeric_limits<std::uint32_t>::max();
    full_capacity.stop.token_ids                    = {source.generated_token_ids.front()};
    full_capacity.stop.publish_stop_token           = true;
    const ninfer::RuntimeStats before               = engine.runtime_stats();
    const ninfer::GenerationResult consumed         = engine.generate(
        engine.prepare(input("last-alias-source", ninfer::CacheRetentionHint::LiveSession)),
        full_capacity);
    const ninfer::RuntimeStats after = engine.runtime_stats();
    if (consumed.generated_token_ids.size() != 1 || consumed.reused_prompt_tokens == 0 ||
        (consumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         consumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint) ||
        after.pressure_private_owners_evicted <= before.pressure_private_owners_evicted ||
        after.device_main_kv_occupied_pages == 0 || after.device_main_kv_occupied_pages > 8 ||
        after.device_backend_kv_occupied_pages == 0 || after.device_backend_kv_occupied_pages > 8) {
        std::cerr << "last private prefix alias did not transfer into the active entitlement: path="
                  << static_cast<int>(consumed.prefix_reuse_path)
                  << " reused=" << consumed.reused_prompt_tokens
                  << " evictions=" << before.pressure_private_owners_evicted << '/'
                  << after.pressure_private_owners_evicted
                  << " main=" << after.device_main_kv_occupied_pages
                  << " backend=" << after.device_backend_kv_occupied_pages << '\n';
        return 1;
    }
    return 0;
}

enum class RewriteCheckpointCacheTopology : std::uint8_t {
    PrivateOnly,
    SharedAlias,
};

int exercise_rewrite_checkpoints(ninfer::Engine& engine, RewriteCheckpointCacheTopology topology) {
    const bool shared_alias = topology == RewriteCheckpointCacheTopology::SharedAlias;
    const ninfer::RuntimeStats initial_stats = engine.runtime_stats();

    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto assistant_call = [&](std::string reasoning, std::string id, std::string key) {
        ninfer::ChatMessage message = text_message(ninfer::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(ninfer::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    auto input_with_history = [&](int completed_responses, bool preserve_thinking) {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (completed_responses >= 1) {
            input.messages.push_back(
                assistant_call("The first lookup should be alpha.", "call_alpha", "alpha"));
            ninfer::ChatMessage tool =
                text_message(ninfer::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
            tool.tool_call_id = "call_alpha";
            input.messages.push_back(std::move(tool));
        }
        if (completed_responses >= 2) {
            input.messages.push_back(
                assistant_call("The alpha result requests beta.", "call_beta", "beta"));
            ninfer::ChatMessage tool = text_message(ninfer::ChatRole::Tool, "{\"value\":25}");
            tool.tool_call_id        = "call_beta";
            input.messages.push_back(std::move(tool));
        }
        input.options.preserve_thinking = preserve_thinking;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return input;
    };
    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 4;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input_with_history(0, true)), options(true));
    if (first.generated_token_ids.size() != 4 ||
        first.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "response-checkpoint source request did not complete from a cold lane\n";
        return 1;
    }

    const ninfer::GenerationResult exact_replay =
        engine.generate(engine.prepare(input_with_history(0, false)), options(true));
    if (exact_replay.generated_token_ids.size() != 4 ||
        exact_replay.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        exact_replay.reused_prompt_tokens == 0) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "pre-generation response checkpoint was not restored on an exact replay: "
                  << "path=" << static_cast<int>(exact_replay.prefix_reuse_path)
                  << " reused=" << exact_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    const ninfer::GenerationResult exact_baseline =
        engine.generate(engine.prepare(input_with_history(0, false)), options(false));
    // Replay can rebuild the MTP bridge at T=1; capture can also split the source prefill.
    // Each route must finish within its own budget and publish a valid reuse frontier.
    if (exact_baseline.generated_token_ids.size() != 4 ||
        exact_baseline.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        exact_baseline.reused_prompt_tokens != 0) {
        std::cerr << "uncached response-checkpoint baseline did not complete from Root: path="
                  << static_cast<int>(exact_baseline.prefix_reuse_path)
                  << " reused=" << exact_baseline.reused_prompt_tokens
                  << " outputs=" << exact_baseline.generated_token_ids.size() << '\n';
        return 1;
    }

    const ninfer::RuntimeStats before_first_replay = engine.runtime_stats();
    const ninfer::GenerationResult first_replay =
        engine.generate(engine.prepare(input_with_history(1, true)), options(true));
    const ninfer::RuntimeStats after_first_replay = engine.runtime_stats();
    const ninfer::PrefixReusePath expected_first_replay =
        shared_alias ? ninfer::PrefixReusePath::SharedStablePrefix
                     : ninfer::PrefixReusePath::PrivateResponseReplay;
    if (first_replay.generated_token_ids.size() != 4 ||
        first_replay.prefix_reuse_path != expected_first_replay ||
        first_replay.reused_prompt_tokens == 0 ||
        (shared_alias && first_replay.reused_prompt_tokens <= exact_replay.reused_prompt_tokens)) {
        std::cerr << "normalized first response selected the wrong cache frontier: path="
                  << static_cast<int>(first_replay.prefix_reuse_path)
                  << " expected=" << static_cast<int>(expected_first_replay)
                  << " reused=" << first_replay.reused_prompt_tokens << '\n';
        return 1;
    }
    if (shared_alias &&
        after_first_replay.historical_fork_hits <= before_first_replay.historical_fork_hits &&
        after_first_replay.state_restores <= before_first_replay.state_restores) {
        std::cerr << "shared rewrite source had no StateImage materialization transition: forks="
                  << before_first_replay.historical_fork_hits << '/'
                  << after_first_replay.historical_fork_hits
                  << " restores=" << before_first_replay.state_restores << '/'
                  << after_first_replay.state_restores << '\n';
        return 1;
    }

    const ninfer::GenerationResult second_replay =
        engine.generate(engine.prepare(input_with_history(2, true)), options(true));
    if (second_replay.generated_token_ids.size() != 4 ||
        second_replay.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        second_replay.reused_prompt_tokens <= first_replay.reused_prompt_tokens) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "rolling response checkpoint did not advance across the tool loop: first="
                  << first_replay.reused_prompt_tokens
                  << " second=" << second_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }

    const ninfer::GenerationResult mode_change =
        engine.generate(engine.prepare(input_with_history(2, false)), options(true));
    if (mode_change.generated_token_ids.size() != 4 ||
        mode_change.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        mode_change.reused_prompt_tokens == 0) {
        std::cerr << "preserve-thinking policy change discarded a compatible response checkpoint: "
                  << "path=" << static_cast<int>(mode_change.prefix_reuse_path)
                  << " reused=" << mode_change.reused_prompt_tokens << '\n';
        return 1;
    }

    if (shared_alias) {
        const ninfer::RuntimeStats final_stats   = engine.runtime_stats();
        const std::uint64_t initial_degradations = initial_stats.pressure_private_owners_degraded +
                                                   initial_stats.pressure_shared_owners_degraded;
        const std::uint64_t final_degradations = final_stats.pressure_private_owners_degraded +
                                                 final_stats.pressure_shared_owners_degraded;
        if (final_degradations <= initial_degradations ||
            final_stats.pressure_private_owners_evicted !=
                initial_stats.pressure_private_owners_evicted ||
            final_stats.pressure_shared_owners_evicted !=
                initial_stats.pressure_shared_owners_evicted ||
            final_stats.pressure_checkpoints_dropped !=
                initial_stats.pressure_checkpoints_dropped ||
            final_stats.active_captures_aborted != initial_stats.active_captures_aborted) {
            std::cerr << "shared/rewrite rotation did not preserve both cache owners: degraded="
                      << initial_degradations << '/' << final_degradations
                      << " private_evicted=" << initial_stats.pressure_private_owners_evicted << '/'
                      << final_stats.pressure_private_owners_evicted
                      << " shared_evicted=" << initial_stats.pressure_shared_owners_evicted << '/'
                      << final_stats.pressure_shared_owners_evicted
                      << " checkpoint_drops=" << initial_stats.pressure_checkpoints_dropped << '/'
                      << final_stats.pressure_checkpoints_dropped
                      << " capture_aborts=" << initial_stats.active_captures_aborted << '/'
                      << final_stats.active_captures_aborted << '\n';
            return 1;
        }
    }

    return 0;
}

int exercise_rewrite_branch(const char* artifact) {
    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    const auto input = [&](bool branch) {
        ninfer::PromptInput value;
        value.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (branch) {
            value.messages.push_back(text_message(ninfer::ChatRole::User,
                                                  "Summarize the conversation before answering."));
        }
        value.options.preserve_thinking = true;
        value.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return value;
    };
    const auto options = [](bool reuse) {
        ninfer::RequestOptions value;
        value.execution.requested_output_tokens = 4;
        value.execution.sampling.temperature    = 0.0F;
        value.execution.allow_prefix_reuse      = reuse;
        value.stop.include_model_defaults       = false;
        return value;
    };

    ninfer::EngineOptions configured             = engine_options(artifact);
    configured.context_cache.device_state_slots  = 2;
    configured.context_cache.max_shared_prefixes = 0;
    ninfer::Engine engine(std::move(configured));
    const ninfer::GenerationResult source =
        engine.generate(engine.prepare(input(false)), options(true));
    if (source.generated_token_ids.size() != 4 ||
        source.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "rewrite branch source did not establish a response checkpoint\n";
        return 1;
    }
    const ninfer::GenerationResult branch =
        engine.generate(engine.prepare(input(true)), options(true));
    const ninfer::GenerationResult branch_baseline =
        engine.generate(engine.prepare(input(true)), options(false));
    if (branch.generated_token_ids.size() != 4 || branch.reused_prompt_tokens == 0 ||
        branch.reused_prompt_tokens >= branch.prompt.prompt_tokens ||
        (branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure) ||
        branch_baseline.generated_token_ids.size() != 4 ||
        branch_baseline.reused_prompt_tokens != 0) {
        std::cerr << "replacement user suffix did not reuse the stable conversation prefix: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens
                  << " prompt=" << branch.prompt.prompt_tokens
                  << " target_count=" << branch.materialization.targets_evaluated
                  << " degradation=" << branch.materialization.selected_degradation_units
                  << " maximal=" << branch.materialization.selected_maximal_fallback << " stop="
                  << ninfer::materialization_stop_reason_name(branch.materialization.stop_reason)
                  << " now_ns=" << branch.materialization.predicted_now_ns
                  << " future_ns=" << branch.materialization.predicted_future_loss_ns << '\n';
        return 1;
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    const auto image_bytes = gradient_ppm();
    auto image_part        = [](const std::vector<std::uint8_t>& bytes, std::string name) {
        ninfer::MessagePart image;
        image.kind              = ninfer::MessagePartKind::Media;
        image.media.kind        = ninfer::MediaKind::Image;
        image.media.bytes       = bytes;
        image.media.media_type  = "image/x-portable-pixmap";
        image.media.source_name = std::move(name);
        return image;
    };
    auto assistant_message = [](const ninfer::GenerationResult& result) {
        ninfer::ChatMessage message;
        message.role              = ninfer::ChatRole::Assistant;
        message.reasoning_content = result.reasoning;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        return message;
    };
    auto first_input = [&](const std::vector<std::uint8_t>& bytes) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(image_part(bytes, "inline.ppm"));
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "vision-prefix-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };
    auto followup_input = [&](const std::vector<std::uint8_t>& bytes,
                              const ninfer::GenerationResult& first) {
        ninfer::PromptInput input = first_input(bytes);
        input.messages.push_back(assistant_message(first));
        ninfer::ChatMessage followup;
        followup.role = ninfer::ChatRole::User;
        followup.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "Give one more detail.", .media = {}});
        input.messages.push_back(std::move(followup));
        return input;
    };
    auto appended_media_input =
        [&](const std::vector<std::uint8_t>& old_bytes, const ninfer::GenerationResult& first,
            const ninfer::GenerationResult& second, const std::vector<std::uint8_t>& new_bytes) {
            ninfer::PromptInput input = followup_input(old_bytes, first);
            input.messages.push_back(assistant_message(second));
            ninfer::ChatMessage followup;
            followup.role = ninfer::ChatRole::User;
            followup.parts.push_back(image_part(new_bytes, "second.ppm"));
            followup.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = "Compare the images.", .media = {}});
            input.messages.push_back(std::move(followup));
            return input;
        };

    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 2;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    // The 1024 merged Vision columns begin after the chat prefix, so the same item necessarily
    // crosses a 1024-token prefill boundary. Its host payload may be released after the first
    // encode, while later chunks must continue to reuse the resident Vision transient.
    ninfer::RequestOptions cross_chunk_options            = options(false);
    cross_chunk_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult cross_chunk =
        engine.generate(engine.prepare(first_input(gradient_ppm(1024, 1024))), cross_chunk_options);
    if (!cross_chunk.prompt.has_media || cross_chunk.generated_token_ids.size() != 1) {
        std::cerr << "cross-chunk Vision item did not complete after releasing its host payload\n";
        return 1;
    }

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_input(image_bytes)), options(true));
    if (!first.prompt.has_media || first.generated_token_ids.size() != 2 ||
        first.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "real Vision request did not complete through the public Engine\n";
        return 1;
    }

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(followup_input(image_bytes, first)), options(true));
    if (reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0.0 ||
        reused.generated_token_ids.size() != 2) {
        std::cerr << "same-media continuation did not reuse the resident Vision prefix: reused="
                  << reused.reused_prompt_tokens << " vision=" << reused.timings.vision_seconds
                  << '\n';
        return 1;
    }

    std::vector<std::uint8_t> second_image = image_bytes;
    second_image.back() ^= 0x5aU;
    const ninfer::GenerationResult appended = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(true));
    if (appended.reused_prompt_tokens == 0 || !(appended.timings.vision_seconds > 0.0) ||
        appended.generated_token_ids.size() != 2) {
        std::cerr << "new-media suffix did not preserve the old multimodal prefix: reused="
                  << appended.reused_prompt_tokens << " vision=" << appended.timings.vision_seconds
                  << '\n';
        return 1;
    }

    const ninfer::GenerationResult baseline = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(false));
    if (baseline.generated_token_ids.size() != 2 || baseline.reused_prompt_tokens != 0 ||
        !(baseline.timings.vision_seconds > 0.0)) {
        std::cerr << "uncached multimodal prefill did not recompute its media\n";
        return 1;
    }

    std::vector<std::uint8_t> changed_prefix = image_bytes;
    changed_prefix[changed_prefix.size() - 2] ^= 0x33U;
    const ninfer::GenerationResult miss = engine.generate(
        engine.prepare(appended_media_input(changed_prefix, first, reused, second_image)),
        options(true));
    if (miss.reused_prompt_tokens != 0) {
        std::cerr << "changed media content incorrectly reused placeholder-token KV\n";
        return 1;
    }

    ninfer::RequestOptions mtp_options            = options(false);
    mtp_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult mtp_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), mtp_options);
    if (mtp_baseline.generated_token_ids.size() != 5 ||
        mtp_baseline.generated_token_ids[0] == mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal stop fixture did not produce distinct leading tokens\n";
        return 1;
    }
    ninfer::RequestOptions stop_options       = mtp_options;
    stop_options.execution.allow_prefix_reuse = true;
    stop_options.stop.token_ids.push_back(mtp_baseline.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare(first_input(image_bytes)), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.empty() || stopped.speculative.rounds == 0 ||
        stopped.generated_token_ids.back() != stop_options.stop.token_ids.front()) {
        std::cerr << "multimodal custom stop did not terminate at the selected token\n";
        return 1;
    }
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare(followup_input(image_bytes, stopped)), options(true));
    if (stopped_reuse.reused_prompt_tokens == 0 || stopped_reuse.timings.vision_seconds != 0.0) {
        std::cerr << "multimodal stop discarded its reusable boundary: reused="
                  << stopped_reuse.reused_prompt_tokens
                  << " vision=" << stopped_reuse.timings.vision_seconds << '\n';
        return 1;
    }

    // Exact registered rendering prefix before the first image-pad column:
    // <|im_start|>user\n<|vision_start|>. Reusing it places the MTP bridge directly on the first
    // Vision merger column rather than on an ordinary token embedding.
    const std::vector<ninfer::TokenId> visual_prefix{248045, 846, 198, 248053};
    ninfer::RequestOptions source_options            = options(true);
    source_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult bridge_source =
        engine.generate(engine.prepare_tokens(visual_prefix), source_options);
    ninfer::RequestOptions bridge_options            = options(true);
    bridge_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult visual_bridge =
        engine.generate(engine.prepare(first_input(image_bytes)), bridge_options);
    if (bridge_source.generated_token_ids.size() != 1 ||
        visual_bridge.reused_prompt_tokens != visual_prefix.size() ||
        !(visual_bridge.timings.vision_seconds > 0.0) || visual_bridge.speculative.rounds == 0) {
        std::cerr << "visual MTP bridge did not append the prefix and enter speculative decode: "
                  << "source_outputs=" << bridge_source.generated_token_ids.size()
                  << " reused=" << visual_bridge.reused_prompt_tokens
                  << " vision=" << visual_bridge.timings.vision_seconds
                  << " rounds=" << visual_bridge.speculative.rounds
                  << " fallbacks=" << visual_bridge.speculative.fallback_steps << '\n';
        return 1;
    }
    if (visual_bridge.generated_token_ids.size() != 5 ||
        visual_bridge.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "visual MTP bridge did not commit the requested output budget\n";
        return 1;
    }
    const auto bridge_followup =
        engine.generate(engine.prepare(followup_input(image_bytes, visual_bridge)), options(true));
    if (bridge_followup.reused_prompt_tokens == 0 ||
        bridge_followup.timings.vision_seconds != 0.0 ||
        bridge_followup.generated_token_ids.size() != 2) {
        std::cerr << "visual MTP bridge lost its retained continuation\n";
        return 1;
    }
    return 0;
}

ninfer::PromptInput session_turn(std::string session, std::string question) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = std::move(session);
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    return input;
}

ninfer::PromptInput pressure_turn(std::string text, std::string session,
                                  ninfer::CacheRetentionHint retention) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    if (!session.empty()) { input.context_cache.session_key = std::move(session); }
    input.context_cache.retention = retention;
    return input;
}

std::optional<std::string> exact_repeated_prompt_text(const ninfer::Engine& engine,
                                                      std::uint32_t target_tokens,
                                                      std::string_view word) {
    const auto text = [word](std::uint32_t repetitions) {
        std::string value;
        value.reserve(static_cast<std::size_t>(repetitions) * (word.size() + 1U));
        for (std::uint32_t index = 0; index < repetitions; ++index) {
            value.push_back(' ');
            value.append(word);
        }
        return value;
    };
    const auto count = [&](std::uint32_t repetitions) {
        return engine.count_tokens(
            pressure_turn(text(repetitions), "", ninfer::CacheRetentionHint::Disposable));
    };

    std::uint32_t low  = 0;
    std::uint32_t high = target_tokens;
    while (low <= high) {
        const std::uint32_t middle = low + (high - low) / 2U;
        const std::uint32_t tokens = count(middle);
        if (tokens == target_tokens) { return text(middle); }
        if (tokens < target_tokens) {
            low = middle + 1U;
        } else {
            if (middle == 0) { break; }
            high = middle - 1U;
        }
    }
    return std::nullopt;
}

ninfer::RequestOptions fixed_output(std::uint32_t tokens, bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int exercise_pressure_partial_spill_and_resume(const char* artifact) {
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 31;
    constexpr std::uint32_t kShortPromptTokens = 350;
    ninfer::Engine engine(pressure_resume_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_a_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    const std::optional<std::string> short_b_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "charlie");
    if (!long_text || !short_a_text || !short_b_text) {
        std::cerr << "pressure-resume fixture could not construct exact prompt geometry\n";
        return 1;
    }

    const ninfer::GenerationResult long_result = engine.generate(
        engine.prepare(pressure_turn(*long_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(kLongOutputTokens));
    if (long_result.prompt.prompt_tokens != kLongPromptTokens ||
        long_result.generated_token_ids.size() != kLongOutputTokens) {
        std::cerr << "pressure-resume long source did not establish its 121-page endpoint: prompt="
                  << long_result.prompt.prompt_tokens
                  << " output=" << long_result.generated_token_ids.size() << '\n';
        return 1;
    }

    const ninfer::GenerationResult short_a =
        engine.generate(engine.prepare(pressure_turn(*short_a_text, "pressure-short-a",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    if (short_a.prompt.prompt_tokens != kShortPromptTokens ||
        short_a.generated_token_ids.size() != 1) {
        std::cerr << "pressure-resume short source did not establish its six-page reservation\n";
        return 1;
    }

    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult short_b =
        engine.generate(engine.prepare(pressure_turn(*short_b_text, "pressure-short-b",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    const std::uint64_t pressure_main_pages =
        after_pressure.main_kv_d2h_pages - before_pressure.main_kv_d2h_pages;
    const std::uint64_t pressure_spill_pages =
        after_pressure.pressure_spill_pages - before_pressure.pressure_spill_pages;
    const std::uint64_t pressure_drops =
        after_pressure.pressure_checkpoints_dropped - before_pressure.pressure_checkpoints_dropped;
    const std::uint64_t pressure_degraded = after_pressure.pressure_private_owners_degraded -
                                            before_pressure.pressure_private_owners_degraded;
    const std::uint64_t pressure_evicted = after_pressure.pressure_private_owners_evicted -
                                           before_pressure.pressure_private_owners_evicted;
    if (short_b.generated_token_ids.size() != 1 || pressure_main_pages != 4 ||
        pressure_spill_pages != 4 || pressure_drops != 1 || pressure_degraded != 1 ||
        pressure_evicted != 0 ||
        after_pressure.state_d2h_count != before_pressure.state_d2h_count ||
        short_b.materialization.selected_maximal_fallback) {
        std::cerr << "pressure-resume did not select endpoint-drop plus four-page spill: main="
                  << pressure_main_pages << " spill=" << pressure_spill_pages
                  << " drops=" << pressure_drops << " degraded=" << pressure_degraded
                  << " evicted=" << pressure_evicted
                  << " state=" << (after_pressure.state_d2h_count - before_pressure.state_d2h_count)
                  << " device_pages=" << before_pressure.device_main_kv_occupied_pages << '/'
                  << after_pressure.device_main_kv_occupied_pages
                  << " maximal=" << short_b.materialization.selected_maximal_fallback
                  << " budget=" << short_b.materialization.budget_exhausted << '\n';
        return 1;
    }
    const ninfer::RuntimeStats before_resume = engine.runtime_stats();
    const ninfer::GenerationResult resumed   = engine.generate(
        engine.prepare(pressure_turn(*long_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(1));
    const ninfer::RuntimeStats after_resume = engine.runtime_stats();
    const std::uint64_t restored_pages =
        after_resume.main_kv_h2d_pages - before_resume.main_kv_h2d_pages;
    const std::uint32_t reused_pages = (resumed.reused_prompt_tokens + 63U) / 64U;
    if (resumed.generated_token_ids.size() != 1 ||
        resumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
        reused_pages != 120 || restored_pages != 4) {
        std::cerr << "pressure-resume did not restore the retained turn closure: path="
                  << static_cast<int>(resumed.prefix_reuse_path)
                  << " reused=" << resumed.reused_prompt_tokens << " reused_pages=" << reused_pages
                  << " restored=" << restored_pages << '\n';
        return 1;
    }
    return 0;
}

int exercise_materialization_source_pressure_protection(const char* artifact) {
    // The source occupies 121 pages and the second owner occupies six, leaving one free page. The
    // branch reuses the source's 120-page turn closure but needs two suffix pages. Under the old
    // guided closure, the source's unprotected endpoint tail was selected as the one-page Host KV
    // victim even though the same continuation was the materialization source.
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 31;
    constexpr std::uint32_t kShortPromptTokens = 350;
    ninfer::Engine engine(pressure_resume_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    if (!long_text || !short_text) {
        std::cerr << "source-pressure fixture could not construct exact prompt geometry\n";
        return 1;
    }

    const ninfer::GenerationResult source =
        engine.generate(engine.prepare(pressure_turn(*long_text, "source-pressure-origin",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(kLongOutputTokens));
    const ninfer::GenerationResult resident =
        engine.generate(engine.prepare(pressure_turn(*short_text, "source-pressure-resident",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    const ninfer::RuntimeStats before_branch = engine.runtime_stats();
    if (source.prompt.prompt_tokens != kLongPromptTokens ||
        source.generated_token_ids.size() != kLongOutputTokens ||
        resident.prompt.prompt_tokens != kShortPromptTokens ||
        resident.generated_token_ids.size() != 1 ||
        before_branch.device_main_kv_occupied_pages != 127) {
        std::cerr << "source-pressure fixture did not establish 127 resident pages: source="
                  << source.prompt.prompt_tokens << '+' << source.generated_token_ids.size()
                  << " resident=" << resident.prompt.prompt_tokens << '+'
                  << resident.generated_token_ids.size()
                  << " pages=" << before_branch.device_main_kv_occupied_pages << '\n';
        return 1;
    }

    ninfer::PromptInput branch = pressure_turn(*long_text, "source-pressure-branch",
                                               ninfer::CacheRetentionHint::LiveSession);
    std::string suffix;
    for (std::uint32_t index = 0; index < 96; ++index) { suffix += " delta"; }
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(suffix), .media = {}});
    branch.messages.push_back(std::move(followup));

    const ninfer::GenerationResult branched =
        engine.generate(engine.prepare(std::move(branch)), fixed_output(1));
    const ninfer::RuntimeStats after_branch = engine.runtime_stats();
    const std::uint64_t demoted_pages =
        after_branch.main_kv_d2h_pages - before_branch.main_kv_d2h_pages;
    const std::uint64_t degraded = after_branch.pressure_private_owners_degraded -
                                   before_branch.pressure_private_owners_degraded;
    const bool private_partial_source =
        branched.prefix_reuse_path == ninfer::PrefixReusePath::PrivateResponseReplay ||
        branched.prefix_reuse_path == ninfer::PrefixReusePath::PrivateTurnClosure;
    if (branched.generated_token_ids.size() != 1 || !private_partial_source ||
        branched.reused_prompt_tokens == 0 ||
        branched.reused_prompt_tokens >= branched.prompt.prompt_tokens || demoted_pages == 0 ||
        degraded == 0 || branched.materialization.selected_maximal_fallback) {
        std::cerr << "source-pressure branch did not preserve its source under guided Host KV "
                     "pressure: path="
                  << static_cast<int>(branched.prefix_reuse_path)
                  << " reused=" << branched.reused_prompt_tokens
                  << " prompt=" << branched.prompt.prompt_tokens << " demoted=" << demoted_pages
                  << " degraded=" << degraded
                  << " maximal=" << branched.materialization.selected_maximal_fallback << " stop="
                  << ninfer::materialization_stop_reason_name(branched.materialization.stop_reason)
                  << '\n';
        return 1;
    }
    return 0;
}

int exercise_private_checkpoint_pressure_retention(const char* artifact) {
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 16;
    constexpr std::uint32_t kShortPromptTokens = 350;
    constexpr std::uint32_t kShortOutputTokens = 256;
    ninfer::Engine engine(private_checkpoint_pressure_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_b_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    const std::optional<std::string> short_c_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "charlie");
    if (!long_text || !short_b_text || !short_c_text) {
        std::cerr << "private-checkpoint pressure fixture could not construct prompt geometry\n";
        return 1;
    }

    const std::string session             = "private-checkpoint-pressure-source";
    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(pressure_turn(*long_text, session, ninfer::CacheRetentionHint::LiveSession)),
        fixed_output(kLongOutputTokens));
    if (source.prompt.prompt_tokens != kLongPromptTokens ||
        source.generated_token_ids.size() != kLongOutputTokens) {
        std::cerr << "private-checkpoint pressure source did not establish its long session\n";
        return 1;
    }

    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    auto short_b                               = engine.submit(
        engine.prepare(pressure_turn(*short_b_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(kShortOutputTokens));
    auto short_c = engine.submit(
        engine.prepare(pressure_turn(*short_c_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(kShortOutputTokens));
    const ninfer::GenerationResult short_b_result = short_b.wait();
    const ninfer::GenerationResult short_c_result = short_c.wait();
    const ninfer::RuntimeStats after_pressure     = engine.runtime_stats();
    if (short_b_result.generated_token_ids.size() != kShortOutputTokens ||
        short_c_result.generated_token_ids.size() != kShortOutputTokens ||
        after_pressure.pressure_checkpoints_dropped <=
            before_pressure.pressure_checkpoints_dropped ||
        after_pressure.pressure_private_owners_degraded <=
            before_pressure.pressure_private_owners_degraded ||
        after_pressure.pressure_private_owners_evicted !=
            before_pressure.pressure_private_owners_evicted) {
        std::cerr << "private-checkpoint pressure did not produce a retained checkpoint "
                     "degradation: drops="
                  << before_pressure.pressure_checkpoints_dropped << '/'
                  << after_pressure.pressure_checkpoints_dropped
                  << " degraded=" << before_pressure.pressure_private_owners_degraded << '/'
                  << after_pressure.pressure_private_owners_degraded
                  << " evicted=" << before_pressure.pressure_private_owners_evicted << '/'
                  << after_pressure.pressure_private_owners_evicted << '\n';
        return 1;
    }

    ninfer::PromptInput resume =
        pressure_turn(*long_text, session, ninfer::CacheRetentionHint::LiveSession);
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = source.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = source.content, .media = {}});
    resume.messages.push_back(std::move(assistant));
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Return the retained answer in one line.",
        .media = {},
    });
    resume.messages.push_back(std::move(followup));
    const ninfer::GenerationResult resumed =
        engine.generate(engine.prepare(std::move(resume)), fixed_output(1));
    if (resumed.generated_token_ids.size() != 1 ||
        resumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
        resumed.reused_prompt_tokens == 0) {
        std::cerr << "private checkpoint pressure discarded the reusable turn closure: path="
                  << static_cast<int>(resumed.prefix_reuse_path)
                  << " reused=" << resumed.reused_prompt_tokens
                  << " future_ns=" << resumed.materialization.predicted_future_loss_ns << '\n';
        return 1;
    }
    return 0;
}

int exercise_concurrent_resource_settlement(const char* artifact) {
    ninfer::Engine engine(concurrent_engine_options(artifact));

    constexpr std::string_view kSession = "publication-order-real";
    constexpr std::string_view kOlderQuestion =
        "Describe deterministic scheduling using exactly one concise paragraph.";
    constexpr std::string_view kNewerQuestion =
        "Describe prefix caching using exactly one concise paragraph.";
    auto older = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kOlderQuestion))),
        fixed_output(24));
    auto newer = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    const ninfer::GenerationResult newer_result = newer.wait();
    const ninfer::GenerationResult older_result = older.wait();
    if (newer_result.generated_token_ids.size() != 2 ||
        older_result.generated_token_ids.size() != 24) {
        std::cerr << "concurrent session requests did not reach staggered terminal boundaries\n";
        return 1;
    }

    for (std::uint32_t index = 0; index < 6; ++index) {
        const std::string suffix              = std::to_string(index);
        const ninfer::GenerationResult filler = engine.generate(
            engine.prepare(session_turn("publication-filler-" + suffix,
                                        "Give one deterministic token for filler " + suffix + '.')),
            fixed_output(1));
        if (filler.generated_token_ids.size() != 1) {
            std::cerr << "session-order catalog filler did not complete\n";
            return 1;
        }
    }
    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult pressure    = engine.generate(
        engine.prepare(session_turn("publication-pressure",
                                       "Give one deterministic token for the pressure request.")),
        fixed_output(1));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    if (pressure.generated_token_ids.size() != 1 ||
        after_pressure.pressure_private_owners_evicted <=
            before_pressure.pressure_private_owners_evicted) {
        std::cerr << "full session catalog did not execute its canonical eviction\n";
        return 1;
    }

    const ninfer::GenerationResult replay = engine.generate(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    if (replay.generated_token_ids.size() != 2 || replay.reused_prompt_tokens == 0 ||
        replay.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << "late older finish exposed the newer session binding to pressure: path="
                  << static_cast<int>(replay.prefix_reuse_path)
                  << " reused=" << replay.reused_prompt_tokens << '\n';
        return 1;
    }

    {
        std::vector<ninfer::TokenId> long_prompt(400, 198);
        auto cancelled =
            engine.submit(engine.prepare_tokens(std::move(long_prompt)), fixed_output(32, false));
        if (!cancelled) {
            std::cerr << "materialization cancellation fixture did not create a handle\n";
            return 1;
        }
    }
    const ninfer::GenerationResult after_cancel = engine.generate(
        engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198}), fixed_output(1, false));
    if (after_cancel.generated_token_ids.size() != 1) {
        std::cerr << "request after materialization cancellation did not complete\n";
        return 1;
    }

    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(8);
    for (std::uint32_t row = 0; row < 8; ++row) {
        std::vector<ninfer::TokenId> prompt{
            248045, 846, 198, static_cast<ninfer::TokenId>(1000 + row), 248046, 198};
        handles.push_back(
            engine.submit(engine.prepare_tokens(std::move(prompt)), fixed_output(row + 1, false)));
    }
    for (std::uint32_t row = 0; row < handles.size(); ++row) {
        const ninfer::GenerationResult result = handles[row].wait();
        if (result.generated_token_ids.size() != row + 1 ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            std::cerr << "C=8 staggered row " << row << " did not terminate independently\n";
            return 1;
        }
    }
    const ninfer::RuntimeStats settled = engine.runtime_stats();
    if (settled.running_requests != 0 || settled.materializing_requests != 0 ||
        settled.prefilling_requests != 0 || settled.decode_ready_requests != 0 ||
        settled.capture_pending_requests != 0 || settled.terminal_pending_requests != 0) {
        std::cerr << "C=8 terminal settlement left live logical membership: running="
                  << settled.running_requests << " materializing=" << settled.materializing_requests
                  << " prefill=" << settled.prefilling_requests
                  << " decode=" << settled.decode_ready_requests
                  << " capture=" << settled.capture_pending_requests
                  << " terminal=" << settled.terminal_pending_requests << '\n';
        return 1;
    }
    return 0;
}

// The "speculative" flavour of the interleaving scenarios: MTP unless a scenario selects DFlash2.
ninfer::SpeculativeBackend g_interleave_speculative = ninfer::SpeculativeBackend::Mtp;

const char* interleave_label(bool speculative) {
    if (!speculative) { return "plain"; }
    return g_interleave_speculative == ninfer::SpeculativeBackend::DFlash2 ? "DFlash2" : "MTP";
}
ninfer::EngineOptions interleaved_prefill_engine_options(const char* artifact,
                                                         std::uint32_t prefill_lanes, bool mtp,
                                                         std::uint32_t concurrency = 3) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 4096;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk = 256;
    if (mtp) {
        options.speculative.backend       = g_interleave_speculative;
        options.speculative.draft_tokens  = g_interleave_speculative == ninfer::SpeculativeBackend::DFlash2 ? 7 : 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    } else {
        options.speculative.backend = ninfer::SpeculativeBackend::None;
    }
    options.max_concurrency                  = concurrency;
    options.max_pending_requests             = concurrency;
    options.max_prefill_lanes                = prefill_lanes;
    options.context_cache.device_state_slots = 2 * concurrency;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = concurrency;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

constexpr std::uint32_t kInterleaveOutputTokens = 8;

std::vector<ninfer::TokenId> interleave_long_prompt(std::uint32_t tokens, std::uint32_t salt) {
    std::vector<ninfer::TokenId> prompt(tokens);
    for (std::uint32_t index = 0; index < tokens; ++index) {
        prompt[index] = static_cast<ninfer::TokenId>(1000 + (index * 37U + salt * 101U) % 500U);
    }
    return prompt;
}

std::vector<ninfer::TokenId> interleave_short_prompt() {
    return {248045, 846, 198, 5834, 248046, 198};
}

// Greedy output of each prompt run alone on an engine that serves one request at a time.
std::vector<std::vector<ninfer::TokenId>>
interleave_references(const char* artifact, bool mtp,
                      const std::vector<std::vector<ninfer::TokenId>>& prompts) {
    ninfer::Engine engine(interleaved_prefill_engine_options(artifact, 1, mtp));
    std::vector<std::vector<ninfer::TokenId>> references;
    for (const auto& prompt : prompts) {
        references.push_back(
            engine.generate(engine.prepare_tokens(prompt), fixed_output(kInterleaveOutputTokens, false))
                .generated_token_ids);
        if (references.back().size() != kInterleaveOutputTokens) {
            std::cerr << "interleaved-prefill reference run did not produce its output\n";
            return {};
        }
    }
    return references;
}

void print_interleave_ids(const char* label, const std::vector<ninfer::TokenId>& ids) {
    std::cerr << "  " << label << ':';
    for (const ninfer::TokenId id : ids) { std::cerr << ' ' << id; }
    std::cerr << '\n';
}

// The engine is idle once the worker has settled every request, including cancelled ones.
bool interleave_engine_settles(const ninfer::Engine& engine) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.running_requests == 0 && stats.prefilling_requests == 0 &&
            stats.materializing_requests == 0 && stats.capture_pending_requests == 0 &&
            stats.waiting_requests == 0 && stats.terminal_pending_requests == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

// A long prompt submitted first must not hold back a short one submitted behind it once more than
// one request may prefill, and interleaving must not change either request's greedy output.
int exercise_interleaved_prefill(const char* artifact, bool mtp) {
    const char* label = interleave_label(mtp);
    const auto long_prompt  = interleave_long_prompt(3000, 0);
    const auto short_prompt = interleave_short_prompt();
    const auto references   = interleave_references(artifact, mtp, {long_prompt, short_prompt});
    if (references.empty()) { return 1; }

    {
        // One prefill lane, both requests in flight: separates what concurrent decode batching
        // does to greedy output from what interleaving prefill does.
        ninfer::Engine control(interleaved_prefill_engine_options(artifact, 1, mtp));
        auto long_handle  = control.submit(control.prepare_tokens(long_prompt),
                                           fixed_output(kInterleaveOutputTokens, false));
        auto short_handle = control.submit(control.prepare_tokens(short_prompt),
                                           fixed_output(kInterleaveOutputTokens, false));
        const ninfer::GenerationResult control_short = short_handle.wait();
        const ninfer::GenerationResult control_long  = long_handle.wait();
        std::cout << label << ", 1 prefill lane : short total "
                  << control_short.timings.total_seconds << " s, long first token "
                  << control_long.timings.first_token_seconds << " s, long total "
                  << control_long.timings.total_seconds << " s\n";
        if (control_long.generated_token_ids != references[0] ||
            control_short.generated_token_ids != references[1]) {
            std::cerr << label
                      << ": concurrent requests with one prefill lane already differ from serial "
                         "runs\n";
            print_interleave_ids("serial long ", references[0]);
            print_interleave_ids("1-lane long ", control_long.generated_token_ids);
            print_interleave_ids("serial short", references[1]);
            print_interleave_ids("1-lane short", control_short.generated_token_ids);
        }
    }

    ninfer::Engine engine(interleaved_prefill_engine_options(artifact, 2, mtp));
    auto long_handle = engine.submit(engine.prepare_tokens(long_prompt),
                                     fixed_output(kInterleaveOutputTokens, false));
    auto short_handle = engine.submit(engine.prepare_tokens(short_prompt),
                                      fixed_output(kInterleaveOutputTokens, false));
    const ninfer::GenerationResult short_result = short_handle.wait();
    const ninfer::GenerationResult long_result  = long_handle.wait();
    std::cout << label << ", 2 prefill lanes: short total " << short_result.timings.total_seconds
              << " s, long first token " << long_result.timings.first_token_seconds
              << " s, long total " << long_result.timings.total_seconds << " s\n";
    if (short_result.generated_token_ids != references[1] ||
        long_result.generated_token_ids != references[0]) {
        std::cerr << label << ": interleaved prefill changed a request's greedy output\n";
        print_interleave_ids("serial long ", references[0]);
        print_interleave_ids("2-lane long ", long_result.generated_token_ids);
        print_interleave_ids("serial short", references[1]);
        print_interleave_ids("2-lane short", short_result.generated_token_ids);
        return 1;
    }
    // Serial prefill would finish the short request only after the long prompt's whole prefill
    // (3000 tokens, twelve 256-token units), so the long request's first token comes first.
    if (!(short_result.timings.total_seconds < long_result.timings.first_token_seconds)) {
        std::cerr << label << ": short request did not finish before the long prompt's first token\n";
        return 1;
    }
    if (!interleave_engine_settles(engine)) {
        std::cerr << label << ": interleaved prefill left live logical membership\n";
        return 1;
    }
    return 0;
}

struct CancelledWait {
    std::optional<ninfer::GenerationResult> result;
    // Set when wait() threw. A cancelled request completes with FinishReason::Cancelled and never
    // throws, so an exception is a failure of the request, not a cancellation.
    bool threw = false;
};

// Waits on `handle` with a cancellation that fires once `flag` is set.
CancelledWait wait_with_cancellation(ninfer::GenerationHandle& handle, std::atomic<bool>& flag) {
    CancelledWait outcome;
    try {
        outcome.result = handle.wait(nullptr, ninfer::CancellationView([&flag] {
                                         return flag.load(std::memory_order_acquire);
                                     }));
    } catch (const std::exception& error) {
        outcome.threw = true;
        std::cerr << "wait() threw instead of completing: " << error.what() << '\n';
    }
    return outcome;
}

bool was_cancelled(const CancelledWait& outcome) {
    return outcome.result && outcome.result->finish_reason == ninfer::FinishReason::Cancelled;
}

// Cancelling or abandoning one of several interleaved requests must free its lane, KV and state
// without disturbing the others, at each point of the request's life: queued, prefilling, decoding.
int exercise_interleaved_prefill_cancel(const char* artifact, bool mtp) {
    const char* label       = interleave_label(mtp);
    const auto long_prompt  = interleave_long_prompt(3000, 0);
    const auto short_prompt = interleave_short_prompt();
    const auto references   = interleave_references(artifact, mtp, {long_prompt, short_prompt});
    if (references.empty()) { return 1; }
    ninfer::Engine engine(interleaved_prefill_engine_options(artifact, 2, mtp));
    const auto submit_long = [&] {
        return engine.submit(engine.prepare_tokens(long_prompt),
                             fixed_output(kInterleaveOutputTokens, false));
    };
    const auto submit_short = [&] {
        return engine.submit(engine.prepare_tokens(short_prompt),
                             fixed_output(kInterleaveOutputTokens, false));
    };
    // The follow-up request runs on the same engine: it proves the cancelled request released
    // everything and that no state or KV it touched leaked into the next one.
    const auto follow_up = [&](const char* what) {
        const ninfer::GenerationResult again = engine.generate(
            engine.prepare_tokens(short_prompt), fixed_output(kInterleaveOutputTokens, false));
        if (again.generated_token_ids != references[1] || !interleave_engine_settles(engine)) {
            std::cerr << label << ": engine was not clean after " << what << '\n';
            return 1;
        }
        return 0;
    };

    // 1. The long prompt is cancelled while it prefills and the short request is in flight.
    {
        auto long_handle  = submit_long();
        auto short_handle = submit_short();
        std::atomic<bool> cancel{false};
        CancelledWait long_outcome;
        std::thread long_waiter(
            [&] { long_outcome = wait_with_cancellation(long_handle, cancel); });
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        cancel.store(true, std::memory_order_release);
        const ninfer::GenerationResult short_result = short_handle.wait();
        long_waiter.join();
        if (!was_cancelled(long_outcome) || short_result.generated_token_ids != references[1]) {
            std::cerr << label << ": cancelling the long prompt mid-prefill "
                      << (was_cancelled(long_outcome) ? "changed the short request's output"
                                                      : "did not cancel it")
                      << '\n';
            return 1;
        }
        if (const int result = follow_up("cancelling the long prompt"); result != 0) {
            return result;
        }
    }

    // 2. The short request is cancelled while the long prompt prefills.
    {
        auto long_handle  = submit_long();
        auto short_handle = submit_short();
        std::atomic<bool> cancel{false};
        CancelledWait short_outcome;
        std::thread short_waiter(
            [&] { short_outcome = wait_with_cancellation(short_handle, cancel); });
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        cancel.store(true, std::memory_order_release);
        const ninfer::GenerationResult long_result = long_handle.wait();
        short_waiter.join();
        if (!was_cancelled(short_outcome) || long_result.generated_token_ids != references[0]) {
            std::cerr << label << ": cancelling the short request "
                      << (was_cancelled(short_outcome) ? "changed the long prompt's output"
                                                       : "did not cancel it")
                      << '\n';
            return 1;
        }
        if (const int result = follow_up("cancelling the short request"); result != 0) {
            return result;
        }
    }

    // 3. The short request is cancelled before it is admitted.
    {
        auto long_handle  = submit_long();
        auto short_handle = submit_short();
        std::atomic<bool> cancel{true};
        const CancelledWait short_outcome = wait_with_cancellation(short_handle, cancel);
        const ninfer::GenerationResult long_result = long_handle.wait();
        if (long_result.generated_token_ids != references[0]) {
            std::cerr << label << ": an early cancellation changed the long prompt's output\n";
            return 1;
        }
        // The flag is already set, so the request is normally cancelled; it may instead complete
        // before the engine observes the flag, and then its output must be the serial one.
        if (!short_outcome.result ||
            (!was_cancelled(short_outcome) &&
             short_outcome.result->generated_token_ids != references[1])) {
            std::cerr << label << ": a request cancelled at submission "
                      << (short_outcome.threw ? "failed" : "returned wrong output") << '\n';
            return 1;
        }
        if (const int result = follow_up("an early cancellation"); result != 0) { return result; }
    }

    // 4. The long prompt's handle is abandoned mid-prefill.
    {
        auto short_handle = [&] {
            auto long_handle = submit_long();
            auto short_h     = submit_short();
            std::this_thread::sleep_for(std::chrono::milliseconds(700));
            return short_h;
            // long_handle is destroyed here, abandoning the request mid-prefill.
        }();
        const ninfer::GenerationResult short_result = short_handle.wait();
        if (short_result.generated_token_ids != references[1]) {
            std::cerr << label << ": abandoning the long prompt changed the short output\n";
            return 1;
        }
        if (const int result = follow_up("abandoning the long prompt"); result != 0) {
            return result;
        }
    }
    return 0;
}

// Two long prompts and a short one. With three prefill lanes all three prefill at once; with two,
// the short request waits for a lane. Both must reproduce the serial outputs.
int exercise_interleaved_two_long(const char* artifact, bool mtp) {
    const char* label = interleave_label(mtp);
    const auto long_a = interleave_long_prompt(2500, 1);
    const auto long_b = interleave_long_prompt(2400, 2);
    const auto shorty = interleave_short_prompt();
    const auto references = interleave_references(artifact, mtp, {long_a, long_b, shorty});
    if (references.empty()) { return 1; }
    for (const std::uint32_t lanes : {2U, 3U}) {
        ninfer::Engine engine(interleaved_prefill_engine_options(artifact, lanes, mtp, 4));
        auto a_handle = engine.submit(engine.prepare_tokens(long_a),
                                      fixed_output(kInterleaveOutputTokens, false));
        auto b_handle = engine.submit(engine.prepare_tokens(long_b),
                                      fixed_output(kInterleaveOutputTokens, false));
        auto s_handle = engine.submit(engine.prepare_tokens(shorty),
                                      fixed_output(kInterleaveOutputTokens, false));
        const ninfer::GenerationResult s_result = s_handle.wait();
        const ninfer::GenerationResult a_result = a_handle.wait();
        const ninfer::GenerationResult b_result = b_handle.wait();
        std::cout << label << ", " << lanes << " prefill lanes, two long + short: short total "
                  << s_result.timings.total_seconds << " s, long A first token "
                  << a_result.timings.first_token_seconds << " s total "
                  << a_result.timings.total_seconds << " s, long B first token "
                  << b_result.timings.first_token_seconds << " s total "
                  << b_result.timings.total_seconds << " s\n";
        if (a_result.generated_token_ids != references[0] ||
            b_result.generated_token_ids != references[1] ||
            s_result.generated_token_ids != references[2]) {
            std::cerr << label << ", " << lanes << " lanes: two long prompts changed an output\n";
            print_interleave_ids("serial A", references[0]);
            print_interleave_ids("got    A", a_result.generated_token_ids);
            print_interleave_ids("serial B", references[1]);
            print_interleave_ids("got    B", b_result.generated_token_ids);
            print_interleave_ids("serial S", references[2]);
            print_interleave_ids("got    S", s_result.generated_token_ids);
            return 1;
        }
        // With a lane for it, the short request must not wait out either long prefill.
        if (lanes == 3 && !(s_result.timings.total_seconds <
                            std::min(a_result.timings.first_token_seconds,
                                     b_result.timings.first_token_seconds))) {
            std::cerr << label << ": the short request waited behind two long prompts despite a "
                                  "free prefill lane\n";
            return 1;
        }
        if (!interleave_engine_settles(engine)) {
            std::cerr << label << ", " << lanes << " lanes: left live logical membership\n";
            return 1;
        }
    }
    return 0;
}

int exercise_interleaved_worker_failure(const char* artifact, bool mtp);

int exercise_interleaved_prefill_all(const char* artifact, bool mtp) {
    if (const int result = exercise_interleaved_prefill(artifact, mtp); result != 0) {
        return result;
    }
    if (const int result = exercise_interleaved_prefill_cancel(artifact, mtp); result != 0) {
        return result;
    }
    if (const int result = exercise_interleaved_two_long(artifact, mtp); result != 0) {
        return result;
    }
    return exercise_interleaved_worker_failure(artifact, mtp);
}

constexpr std::uint32_t kLargeContext = 204800;
constexpr std::uint32_t kLargePrompt  = 200000;

ninfer::EngineOptions large_context_engine_options(const char* artifact,
                                                   std::uint32_t prefill_lanes,
                                                   std::uint32_t kv_tokens) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = kLargeContext;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.kv_cache                         = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.prefill_chunk                    = 512;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 3;
    options.max_pending_requests             = 3;
    // A request that cannot fit waits out a 200k prefill (several minutes).
    options.pending_timeout_ms               = 3600000;
    options.max_prefill_lanes                = prefill_lanes;
    options.context_cache.device_state_slots = 6;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 3;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

// The retry-loop case at its real size: a ~200k-token single-message prompt, cancelled at 150k
// tokens (a client timeout), then retried. Serving defaults throughout: context cache on with its
// default capacities, progress anchors at the default stride, serve's 1024-token prefill chunk.
// Minutes of prefill on an RTX 3090; opt-in through its scenario name.
int exercise_cancelled_prefill_200k(const char* artifact) {
    constexpr std::uint32_t kStride     = 16384;
    constexpr std::uint32_t kCancelAt   = 150000;
    const auto options = [&] {
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = artifact;
        engine_options.max_context   = kLargeContext;
        engine_options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(kLargeContext);
        engine_options.kv_cache      = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
        engine_options.max_concurrency      = 1;
        engine_options.max_pending_requests = 1;
        engine_options.pending_timeout_ms   = 3600000;
        return engine_options;
    };
    int entries = 6400;
    const auto prompt = [&](std::uint32_t stride) {
        std::string text = "Read the following log and answer the question at the end.";
        for (int line = 0; line < entries; ++line) {
            text += " Entry " + std::to_string(line) + ": the pump on line " +
                    std::to_string(line % 17) + " reported " + std::to_string(line * 37 % 1013) +
                    " kPa and the operator noted nothing unusual.";
        }
        text += " Question: which line reported the highest pressure?";
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(
            ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking              = false;
        input.context_cache.progress_anchor_stride = stride;
        return input;
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 8;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    ninfer::GenerationResult control;
    {
        ninfer::Engine engine(options());
        // Entry numbers grow from one digit to four, so size the log from a measured count to
        // land the prompt near 195k tokens.
        for (int attempt = 0; attempt < 3; ++attempt) {
            const std::uint32_t counted = engine.count_tokens(prompt(0));
            entries = static_cast<int>(static_cast<std::uint64_t>(entries) * 195000U / counted);
        }
        control = engine.generate(engine.prepare(prompt(0)), request);
        std::cout << "uncancelled: prompt " << control.prompt.prompt_tokens << " tokens, prefill "
                  << control.timings.prefill_seconds << " s\n";
    }
    if (control.generated_token_ids.size() != 8 || control.prompt.prompt_tokens < 180000 ||
        control.prompt.prompt_tokens > kLargeContext - 64U) {
        std::cerr << "200k prompt is outside the intended size: " << control.prompt.prompt_tokens
                  << '\n';
        return 1;
    }

    ninfer::Engine engine(options());
    CancelAtProgressSink sink(kCancelAt);
    const auto cancel_started = std::chrono::steady_clock::now();
    ninfer::GenerationHandle handle =
        engine.submit(engine.prepare(prompt(kStride)), request, ninfer::OutputConsumerMode::Streaming,
                      ninfer::GenerationObservationOptions{.prompt_progress = true});
    const ninfer::GenerationResult cancelled =
        handle.wait(&sink, ninfer::CancellationView([&sink] { return sink.reached(); }));
    const double cancel_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cancel_started).count();
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled || !engine.is_available()) {
        std::cerr << "request was not cancelled mid-prefill\n";
        return 1;
    }
    const ninfer::RuntimeStats after_cancel = settled_runtime_stats(engine);
    const ninfer::GenerationResult retried  = engine.generate(engine.prepare(prompt(kStride)), request);
    std::cout << "cancelled after " << cancel_seconds << " s at >=" << kCancelAt
              << " tokens; retry reused " << retried.reused_prompt_tokens << " of "
              << retried.prompt.prompt_tokens << " (path "
              << static_cast<int>(retried.prefix_reuse_path) << "), retry prefill "
              << retried.timings.prefill_seconds << " s vs uncancelled "
              << control.timings.prefill_seconds << " s; running_requests "
              << after_cancel.running_requests << '\n';
    if (retried.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        retried.reused_prompt_tokens < kCancelAt - kStride ||
        retried.reused_prompt_tokens > kCancelAt + 2U * kStride ||
        retried.reused_prompt_tokens % kStride != 0) {
        std::cerr << "retry did not resume near the cancel point\n";
        return 1;
    }
    if (retried.generated_token_ids != control.generated_token_ids) {
        std::cerr << "output after the cancelled 200k prefill differs from an uncancelled request\n";
        return 1;
    }
    return 0;
}

// One 200k-token prompt on its own: whether it fits and how long it takes to ingest.
int exercise_large_probe(const char* artifact) {
    ninfer::Engine engine(large_context_engine_options(artifact, 1, kLargeContext));
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(interleave_long_prompt(kLargePrompt, 0)),
                        fixed_output(kInterleaveOutputTokens, false));
    std::cout << "200k probe: prompt " << result.prompt.prompt_tokens << " tokens, first token "
              << result.timings.first_token_seconds << " s, prefill "
              << result.timings.prefill_seconds << " s, total " << result.timings.total_seconds
              << " s\n";
    return result.generated_token_ids.size() == kInterleaveOutputTokens ? 0 : 1;
}

// A 200k prompt with a short request behind it, then two 200k prompts whose KV cannot be held at
// once. The second long prompt must wait for capacity without wedging the engine, and the short
// request must still finish with its serial output.
int exercise_large_context(const char* artifact) {
    constexpr std::uint32_t kKvTokens = 260000; // one 200k request fits; two do not
    const auto long_a     = interleave_long_prompt(kLargePrompt, 0);
    const auto long_b     = interleave_long_prompt(kLargePrompt, 1);
    const auto shorty     = interleave_short_prompt();
    std::vector<ninfer::TokenId> reference_a;
    std::vector<ninfer::TokenId> reference_short;
    {
        ninfer::Engine engine(large_context_engine_options(artifact, 1, kKvTokens));
        const ninfer::GenerationResult a = engine.generate(
            engine.prepare_tokens(long_a), fixed_output(kInterleaveOutputTokens, false));
        const ninfer::GenerationResult s = engine.generate(
            engine.prepare_tokens(shorty), fixed_output(kInterleaveOutputTokens, false));
        reference_a     = a.generated_token_ids;
        reference_short = s.generated_token_ids;
        std::cout << "200k serial: long prefill " << a.timings.prefill_seconds << " s, total "
                  << a.timings.total_seconds << " s\n";
        if (reference_a.size() != kInterleaveOutputTokens ||
            reference_short.size() != kInterleaveOutputTokens) {
            std::cerr << "200k reference runs did not produce their output\n";
            return 1;
        }
    }

    ninfer::Engine engine(large_context_engine_options(artifact, 3, kKvTokens));
    const auto start = std::chrono::steady_clock::now();
    auto a_handle    = engine.submit(engine.prepare_tokens(long_a),
                                     fixed_output(kInterleaveOutputTokens, false));
    auto b_handle    = engine.submit(engine.prepare_tokens(long_b),
                                     fixed_output(kInterleaveOutputTokens, false));
    auto s_handle    = engine.submit(engine.prepare_tokens(shorty),
                                     fixed_output(kInterleaveOutputTokens, false));
    const ninfer::GenerationResult s_result = s_handle.wait();
    const double short_done = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const ninfer::GenerationResult a_result = a_handle.wait();
    const double a_done = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const ninfer::GenerationResult b_result = b_handle.wait();
    const double b_done = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "200k x2 + short, KV for one: short done " << short_done << " s, long A done "
              << a_done << " s (first token " << a_result.timings.first_token_seconds
              << " s), long B done " << b_done << " s (first token "
              << b_result.timings.first_token_seconds << " s, queue wait "
              << b_result.engine_timing.queue_wait_seconds << " s)\n";
    if (a_result.generated_token_ids != reference_a ||
        s_result.generated_token_ids != reference_short ||
        b_result.generated_token_ids.size() != kInterleaveOutputTokens) {
        std::cerr << "200k requests that cannot all fit changed an output or failed\n";
        print_interleave_ids("serial A", reference_a);
        print_interleave_ids("got    A", a_result.generated_token_ids);
        print_interleave_ids("serial S", reference_short);
        print_interleave_ids("got    S", s_result.generated_token_ids);
        return 1;
    }
    if (!interleave_engine_settles(engine)) {
        std::cerr << "200k requests left live logical membership\n";
        return 1;
    }
    return 0;
}

// A prompt over max_context must be refused up front, without prefill and without disturbing the
// engine. A prompt of exactly max_context is valid, so it is not exercised here: it would prefill
// 200k tokens.
int exercise_over_context(const char* artifact) {
    ninfer::Engine engine(large_context_engine_options(artifact, 2, kLargeContext));
    const auto shorty = interleave_short_prompt();
    const auto expect_refused = [&](std::uint32_t prompt_tokens, std::uint32_t output_tokens) {
        try {
            auto handle = engine.submit(
                engine.prepare_tokens(interleave_long_prompt(prompt_tokens, 3)),
                fixed_output(output_tokens, false));
            (void)handle.wait();
        } catch (const ninfer::RequestError& error) {
            // Only the documented refusal counts; any other failure is a different problem.
            if (error.kind() != ninfer::RequestErrorKind::ContextLengthExceeded) {
                std::cerr << "oversize prompt failed with the wrong error: " << error.what() << '\n';
                return false;
            }
            std::cout << "refused " << prompt_tokens << " + " << output_tokens
                      << " tokens: " << error.what() << '\n';
            return true;
        } catch (const std::exception& error) {
            std::cerr << "oversize prompt failed with an unexpected error: " << error.what() << '\n';
            return false;
        }
        return false;
    };
    if (!expect_refused(kLargeContext + 1000, 8)) {
        std::cerr << "a prompt over max_context was accepted\n";
        return 1;
    }
    const ninfer::GenerationResult after =
        engine.generate(engine.prepare_tokens(shorty), fixed_output(kInterleaveOutputTokens, false));
    if (after.generated_token_ids.size() != kInterleaveOutputTokens ||
        !interleave_engine_settles(engine)) {
        std::cerr << "engine was not clean after refusing oversize prompts\n";
        return 1;
    }
    return 0;
}

// A worker fault while two requests own a staged prefill fails both lanes, and recovery must release
// both prefill owners: the Engine then serves two interleaved prompts again with their serial outputs.
int exercise_interleaved_worker_failure(const char* artifact, bool mtp) {
    const char* label = interleave_label(mtp);
    const auto long_a = interleave_long_prompt(3000, 1);
    const auto long_b = interleave_long_prompt(2900, 2);
    const auto references = interleave_references(artifact, mtp, {long_a, long_b});
    if (references.empty()) { return 1; }
    ninfer::Engine engine(interleaved_prefill_engine_options(artifact, 2, mtp));
    const auto submit = [&](const std::vector<ninfer::TokenId>& prompt) {
        return engine.submit(engine.prepare_tokens(prompt),
                             fixed_output(kInterleaveOutputTokens, false));
    };
    struct Result {
        bool served      = false;
        bool unavailable = false; // the Engine refused the request instead of failing it
        std::vector<ninfer::TokenId> ids;
    };
    const auto wait_for = [](ninfer::GenerationHandle& handle) {
        Result outcome;
        try {
            outcome.ids    = handle.wait().generated_token_ids;
            outcome.served = true;
        } catch (const ninfer::RequestError& error) {
            outcome.unavailable = error.kind() == ninfer::RequestErrorKind::Unavailable;
        } catch (const std::exception&) {}
        return outcome;
    };
    constexpr std::uint32_t kRounds = 3;
    for (std::uint32_t round = 0; round < kRounds; ++round) {
        auto a = submit(long_a);
        auto b = submit(long_b);
        // Both requests must be prefilling before the fault, so it lands with two owners.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (engine.runtime_stats().prefilling_requests < 2) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::cerr << label << ": two requests never prefilled at once\n";
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ninfer::runtime::arm_worker_failures(1);
        const Result a_result = wait_for(a);
        const Result b_result = wait_for(b);
        ninfer::runtime::arm_worker_failures(0);
        // Both lanes are failed by the fault. A request that was refused as Unavailable would mean
        // the Engine latched or never recovered, which is a different outcome.
        if (a_result.served || b_result.served || a_result.unavailable || b_result.unavailable ||
            !engine.is_available() || engine.runtime_stats().engine_recoveries != round + 1) {
            std::cerr << label << ": worker fault round " << round
                      << ": a served=" << a_result.served << " b served=" << b_result.served
                      << " a unavailable=" << a_result.unavailable
                      << " b unavailable=" << b_result.unavailable
                      << " available=" << engine.is_available()
                      << " recoveries=" << engine.runtime_stats().engine_recoveries << '\n';
            return 1;
        }
        // Recovery released both owners and every lane: interleaving works again, exactly.
        auto a_again = submit(long_a);
        auto b_again = submit(long_b);
        const Result a_after = wait_for(a_again);
        const Result b_after = wait_for(b_again);
        if (!a_after.served || !b_after.served || a_after.ids != references[0] ||
            b_after.ids != references[1] || !interleave_engine_settles(engine)) {
            std::cerr << label << ": engine did not interleave correctly after recovery round "
                      << round << '\n';
            return 1;
        }
    }
    return 0;
}

struct PrefillRoute {
    const char* label;
    std::uint32_t chunk;
    bool cublas;
};

constexpr PrefillRoute kPlainRoute{"chunk 512", 512, false};
constexpr PrefillRoute kCublasRoute{"cuBLAS, chunk 4096", 4096, true};

ninfer::EngineOptions pair_engine_options(const char* artifact, std::uint32_t prefill_lanes,
                                          const PrefillRoute& route) {
    ninfer::EngineOptions options =
        large_context_engine_options(artifact, prefill_lanes, 200000);
    options.max_context    = 90000;
    options.prefill_chunk  = route.chunk;
    options.prefill_cublas = route.cublas;
    return options;
}

// Two 80k prompts and a short one all fit at once. They must reproduce their serial outputs when
// interleaved, and cancelling any of them, at 80k scale, must leave the others and the engine intact.
int exercise_two_80k(const char* artifact, const PrefillRoute& route) {
    constexpr std::uint32_t kPromptTokens = 80000;
    const auto long_a = interleave_long_prompt(kPromptTokens, 1);
    const auto long_b = interleave_long_prompt(kPromptTokens, 2);
    const auto shorty = interleave_short_prompt();
    const std::array<const std::vector<ninfer::TokenId>*, 3> prompts{&long_a, &long_b, &shorty};
    std::array<std::vector<ninfer::TokenId>, 3> reference;
    {
        ninfer::Engine engine(pair_engine_options(artifact, 1, route));
        for (std::size_t index = 0; index < prompts.size(); ++index) {
            const ninfer::GenerationResult result =
                engine.generate(engine.prepare_tokens(*prompts[index]),
                                fixed_output(kInterleaveOutputTokens, false));
            reference[index] = result.generated_token_ids;
            if (reference[index].size() != kInterleaveOutputTokens) {
                std::cerr << route.label << ": 80k reference run did not produce its output\n";
                return 1;
            }
            if (index == 0) {
                std::cout << route.label << ", 80k serial: prefill " << result.timings.prefill_seconds
                          << " s\n";
            }
        }
    }

    ninfer::Engine engine(pair_engine_options(artifact, 3, route));
    const auto submit = [&](std::size_t index) {
        return engine.submit(engine.prepare_tokens(*prompts[index]),
                             fixed_output(kInterleaveOutputTokens, false));
    };
    const auto follow_up = [&](const char* what) {
        const ninfer::GenerationResult again = engine.generate(
            engine.prepare_tokens(shorty), fixed_output(kInterleaveOutputTokens, false));
        if (again.generated_token_ids != reference[2] || !interleave_engine_settles(engine)) {
            std::cerr << route.label << ": engine was not clean after " << what << '\n';
            return 1;
        }
        return 0;
    };

    {
        auto a = submit(0);
        auto b = submit(1);
        auto s = submit(2);
        const ninfer::GenerationResult s_result = s.wait();
        const ninfer::GenerationResult a_result = a.wait();
        const ninfer::GenerationResult b_result = b.wait();
        std::cout << route.label << ", two 80k + short, 3 prefill lanes: short "
                  << s_result.timings.total_seconds << " s, A first token "
                  << a_result.timings.first_token_seconds << " s total "
                  << a_result.timings.total_seconds << " s, B first token "
                  << b_result.timings.first_token_seconds << " s total "
                  << b_result.timings.total_seconds << " s\n";
        if (a_result.generated_token_ids != reference[0] ||
            b_result.generated_token_ids != reference[1] ||
            s_result.generated_token_ids != reference[2]) {
            std::cerr << route.label << ": two 80k prompts changed an output\n";
            print_interleave_ids("serial A", reference[0]);
            print_interleave_ids("got    A", a_result.generated_token_ids);
            print_interleave_ids("serial B", reference[1]);
            print_interleave_ids("got    B", b_result.generated_token_ids);
            print_interleave_ids("serial S", reference[2]);
            print_interleave_ids("got    S", s_result.generated_token_ids);
            return 1;
        }
        if (!(s_result.timings.total_seconds < std::min(a_result.timings.first_token_seconds,
                                                         b_result.timings.first_token_seconds))) {
            std::cerr << route.label << ": the short request waited behind the 80k prompts\n";
            return 1;
        }
        if (const int result = follow_up("two interleaved 80k prompts"); result != 0) {
            return result;
        }
    }

    // Cancel a subset after `delay_ms`; the rest must reproduce their serial output.
    const auto cancel_case = [&](const char* what, std::array<bool, 3> cancel_request,
                                 int delay_ms) {
        std::array<ninfer::GenerationHandle, 3> handles{submit(0), submit(1), submit(2)};
        std::array<std::atomic<bool>, 3> flags{};
        std::array<CancelledWait, 3> outcomes;
        std::array<std::thread, 3> waiters;
        for (std::size_t index = 0; index < 3; ++index) {
            waiters[index] = std::thread([&, index] {
                outcomes[index] = wait_with_cancellation(handles[index], flags[index]);
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        for (std::size_t index = 0; index < 3; ++index) {
            if (cancel_request[index]) { flags[index].store(true, std::memory_order_release); }
        }
        for (std::thread& waiter : waiters) { waiter.join(); }
        for (std::size_t index = 0; index < 3; ++index) {
            const bool ok = cancel_request[index]
                                ? was_cancelled(outcomes[index])
                                : (outcomes[index].result &&
                                   outcomes[index].result->generated_token_ids == reference[index]);
            if (!ok) {
                std::cerr << route.label << ": " << what << ": request " << index
                          << (cancel_request[index] ? " was not cancelled" : " changed its output")
                          << '\n';
                return 1;
            }
        }
        return follow_up(what);
    };
    if (const int result = cancel_case("cancelling long A mid-prefill", {true, false, false}, 12000);
        result != 0) {
        return result;
    }
    if (const int result = cancel_case("cancelling long B mid-prefill", {false, true, false}, 12000);
        result != 0) {
        return result;
    }
    if (const int result = cancel_case("cancelling the short request", {false, false, true}, 1000);
        result != 0) {
        return result;
    }
    {
        // Abandon (destroy) long A's handle while both long prompts prefill.
        auto b = submit(1);
        auto s = submit(2);
        {
            auto a = submit(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(12000));
        }
        const ninfer::GenerationResult s_result = s.wait();
        const ninfer::GenerationResult b_result = b.wait();
        if (b_result.generated_token_ids != reference[1] ||
            s_result.generated_token_ids != reference[2]) {
            std::cerr << route.label << ": abandoning long A changed another request's output\n";
            return 1;
        }
        if (const int result = follow_up("abandoning long A"); result != 0) { return result; }
    }
    return 0;
}

ninfer::EngineOptions vision_interleave_engine_options(const char* artifact,
                                                       std::uint32_t prefill_lanes,
                                                       ninfer::VisionResidency residency,
                                                       bool mtp) {
    ninfer::EngineOptions options =
        interleaved_prefill_engine_options(artifact, prefill_lanes, mtp, 3);
    options.max_context      = 8192;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(16384);
    options.enable_vision    = true;
    options.vision_residency = residency;
    return options;
}

ninfer::PromptInput vision_interleave_prompt(int width, int height, std::string_view question) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm(width, height);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "interleave.ppm";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::string(question), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

// Image prompts prefill alongside a long text prompt and each other. Two images at once is the case
// the overlay residency cannot serve concurrently: one lane's encode window blocks the other's.
int exercise_interleaved_vision(const char* artifact, ninfer::VisionResidency residency,
                                bool mtp) {
    const char* label = residency == ninfer::VisionResidency::Overlay ? "overlay" : "resident";
    const auto long_prompt = interleave_long_prompt(3000, 0);
    const auto shorty      = interleave_short_prompt();
    const auto make_image_a = [] { return vision_interleave_prompt(1024, 1024, "What is visible?"); };
    const auto make_image_b = [] { return vision_interleave_prompt(512, 768, "Describe the colors."); };

    std::vector<ninfer::TokenId> reference_long;
    std::vector<ninfer::TokenId> reference_short;
    std::vector<ninfer::TokenId> reference_a;
    std::vector<ninfer::TokenId> reference_b;
    {
        ninfer::Engine engine(vision_interleave_engine_options(artifact, 1, residency, mtp));
        const auto run = [&](ninfer::PreparedPrompt prompt) {
            return engine.generate(std::move(prompt), fixed_output(kInterleaveOutputTokens, false));
        };
        reference_long  = run(engine.prepare_tokens(long_prompt)).generated_token_ids;
        reference_short = run(engine.prepare_tokens(shorty)).generated_token_ids;
        const ninfer::GenerationResult a = run(engine.prepare(make_image_a()));
        const ninfer::GenerationResult b = run(engine.prepare(make_image_b()));
        if (!a.prompt.has_media || !b.prompt.has_media) {
            std::cerr << label << ": vision reference prompts carried no media\n";
            return 1;
        }
        reference_a = a.generated_token_ids;
        reference_b = b.generated_token_ids;
    }

    ninfer::Engine engine(vision_interleave_engine_options(artifact, 3, residency, mtp));
    const auto submit = [&](ninfer::PreparedPrompt prompt) {
        return engine.submit(std::move(prompt), fixed_output(kInterleaveOutputTokens, false));
    };
    auto long_handle  = submit(engine.prepare_tokens(long_prompt));
    auto a_handle     = submit(engine.prepare(make_image_a()));
    auto b_handle     = submit(engine.prepare(make_image_b()));
    auto short_handle = submit(engine.prepare_tokens(shorty));
    const ninfer::GenerationResult short_result = short_handle.wait();
    const ninfer::GenerationResult long_result  = long_handle.wait();
    const ninfer::GenerationResult a_result     = a_handle.wait();
    const ninfer::GenerationResult b_result     = b_handle.wait();
    std::cout << label << (mtp ? " + MTP" : "") << ", 3 prefill lanes, long text + 2 images + short: short "
              << short_result.timings.total_seconds << " s, long " << long_result.timings.total_seconds
              << " s, image A " << a_result.timings.total_seconds << " s (vision "
              << a_result.timings.vision_seconds << " s), image B " << b_result.timings.total_seconds
              << " s (vision " << b_result.timings.vision_seconds << " s)\n";
    if (long_result.generated_token_ids != reference_long ||
        short_result.generated_token_ids != reference_short ||
        a_result.generated_token_ids != reference_a ||
        b_result.generated_token_ids != reference_b) {
        std::cerr << label << ": interleaving vision prefill changed an output\n";
        print_interleave_ids("serial image A", reference_a);
        print_interleave_ids("got    image A", a_result.generated_token_ids);
        print_interleave_ids("serial image B", reference_b);
        print_interleave_ids("got    image B", b_result.generated_token_ids);
        print_interleave_ids("serial long   ", reference_long);
        print_interleave_ids("got    long   ", long_result.generated_token_ids);
        print_interleave_ids("serial short  ", reference_short);
        print_interleave_ids("got    short  ", short_result.generated_token_ids);
        return 1;
    }
    if (!interleave_engine_settles(engine)) {
        std::cerr << label << ": interleaved vision left live logical membership\n";
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.architecture != "Qwen3_5ForCausalLM" || load.model_name.empty() ||
        load.weight_formats.empty() || load.host_to_device_bytes == 0 ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.architecture
                  << " weights=" << load.prefill_signature << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    const auto* vision = memory.vision_workspace ? &*memory.vision_workspace : nullptr;
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || vision == nullptr ||
        vision->aggregate_prompt_tokens != 4096 || vision->max_item_tokens != 4096 ||
        vision->general_capacity_bytes == 0 || vision->encode_peak_bytes == 0 ||
        vision->handoff_offset_bytes > memory.workspace.capacity_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.workspace.capacity_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

// Output reservation (EngineOptions::output_reservation_tokens): a request reserves KV for only part
// of its output when it is admitted and the rest as it decodes.
//  - growth is invisible: a long generation that crosses several reservation chunks produces exactly
//    the tokens, and the finish reason, of one that reserved everything up front;
//  - it is not a fixed cut: two concurrent requests whose full budgets cannot both fit are both
//    admitted and share the free pages; one that finds none free stops, with the length finish
//    reason, where its reservation ran out; reserving everything up front instead runs them one
//    after the other and completes both.
int exercise_lazy_output_reservation(const char* artifact) {
    const auto base_options = [&](std::uint32_t context, std::uint32_t reservation,
                                  std::uint32_t lanes) {
        ninfer::EngineOptions options = store_engine_options(artifact);
        options.max_context                = context;
        options.kv_capacity                = ninfer::KvCapacityPolicy::explicit_capacity(context);
        options.max_concurrency            = lanes;
        options.max_pending_requests       = 4;
        options.output_reservation_tokens  = reservation;
        // The cache has to be able to cover every active request.
        options.context_cache.max_private_continuations = lanes;
        options.context_cache.device_state_slots        = lanes + 1;
        options.context_cache.host_state_slots          = lanes + 1;
        return options;
    };
    const auto request_for = [](std::uint32_t tokens) {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = tokens;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;
        return request;
    };
    const std::vector<std::string> prompt{"Count upward from one, separated by commas."};

    // Growth across several chunks equals reserving everything.
    ninfer::GenerationResult full;
    {
        ninfer::Engine engine(base_options(4096, 0, 1));
        full = engine.generate(engine.prepare(store_conversation(prompt)), request_for(1200));
    }
    ninfer::GenerationResult lazy;
    ninfer::RuntimeStats lazy_stats;
    {
        ninfer::Engine engine(base_options(4096, 100, 1));
        lazy       = engine.generate(engine.prepare(store_conversation(prompt)), request_for(1200));
        lazy_stats = engine.runtime_stats();
    }
    if (full.generated_token_ids.size() != 1200 || lazy.generated_token_ids != full.generated_token_ids ||
        lazy.finish_reason != full.finish_reason ||
        lazy.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "a lazily reserved generation differs from a fully reserved one: full="
                  << full.generated_token_ids.size() << " lazy=" << lazy.generated_token_ids.size()
                  << " finish=" << static_cast<int>(lazy.finish_reason) << '\n';
        return 1;
    }
    if (lazy_stats.output_reservation_growths < 2 || lazy_stats.output_reservation_exhaustions != 0) {
        std::cerr << "the reservation did not grow as the request decoded: growths="
                  << lazy_stats.output_reservation_growths
                  << " exhaustions=" << lazy_stats.output_reservation_exhaustions << '\n';
        return 1;
    }

    // Two requests whose full budgets cannot both fit a 1024-token pool.
    const auto run_pair = [&](std::uint32_t reservation, ninfer::RuntimeStats& stats) {
        ninfer::Engine engine(base_options(1024, reservation, 2));
        auto first  = engine.submit(engine.prepare(store_conversation(prompt)), request_for(900));
        auto second = engine.submit(engine.prepare(store_conversation(
                                        std::vector<std::string>{"Count down from nine hundred."})),
                                    request_for(900));
        std::pair<ninfer::GenerationResult, ninfer::GenerationResult> out{first.wait(), second.wait()};
        stats = engine.runtime_stats();
        return out;
    };
    ninfer::RuntimeStats serial_stats;
    ninfer::RuntimeStats lazy_pair_stats;
    const auto serial = run_pair(0, serial_stats);
    const auto lazy_pair = run_pair(256, lazy_pair_stats);
    if (serial.first.generated_token_ids.size() != 900 ||
        serial.second.generated_token_ids.size() != 900) {
        std::cerr << "fully reserved requests did not both complete\n";
        return 1;
    }
    const std::size_t a = lazy_pair.first.generated_token_ids.size();
    const std::size_t b = lazy_pair.second.generated_token_ids.size();
    if (lazy_pair_stats.output_reservation_exhaustions == 0 || std::min(a, b) >= 900 ||
        std::min(a, b) == 0) {
        std::cerr << "expected a request to stop at its reservation, not at zero: a=" << a
                  << " b=" << b << " exhaustions=" << lazy_pair_stats.output_reservation_exhaustions
                  << '\n';
        return 1;
    }
    if (lazy_pair.first.finish_reason != ninfer::FinishReason::OutputLimit ||
        lazy_pair.second.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "a request did not finish with the length reason\n";
        return 1;
    }
    std::cout << "lazy: growths=" << lazy_stats.output_reservation_growths << "; pair a=" << a
              << " b=" << b << '\n';
    return 0;
}

// The context store's remote copy: a session written by one Engine is restored by another whose
// directory is empty, through the bucket alone, and continues with the warm session's output.
class MemoryBucket final : public ninfer::ObjectStore {
public:
    void put(const std::string& key, std::span<const std::uint8_t> bytes) override {
        std::scoped_lock lock(mutex);
        objects[key] = {std::vector<std::uint8_t>(bytes.begin(), bytes.end()), ++clock};
    }
    std::optional<std::vector<std::uint8_t>> get(const std::string& key) override {
        std::scoped_lock lock(mutex);
        const auto found = objects.find(key);
        if (found == objects.end()) { return std::nullopt; }
        return found->second.first;
    }
    bool exists(const std::string& key) override {
        std::scoped_lock lock(mutex);
        return objects.find(key) != objects.end();
    }
    std::vector<ninfer::ObjectInfo> list(const std::string& prefix) override {
        std::scoped_lock lock(mutex);
        std::vector<ninfer::ObjectInfo> out;
        for (const auto& [key, value] : objects) {
            if (key.rfind(prefix, 0) == 0) {
                out.push_back({key, value.first.size(), static_cast<std::int64_t>(clock)});
            }
        }
        return out;
    }
    bool touch(const std::string& key) override {
        std::scoped_lock lock(mutex);
        return objects.find(key) != objects.end();
    }
    void remove(const std::string& key) override {
        std::scoped_lock lock(mutex);
        objects.erase(key);
    }
    std::mutex mutex;
    std::map<std::string, std::pair<std::vector<std::uint8_t>, std::uint64_t>> objects;
    std::uint64_t clock = 1'800'000'000'000ULL;
};

int exercise_context_store_remote(const char* artifact) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "ninfer-context-store-remote-test";
    fs::remove_all(root);
    fs::create_directories(root);

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 12;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;
    const std::vector<std::string> first{
        "List three uses for a lathe in a small workshop, one line each."};
    const auto second_turn = [&](const ninfer::GenerationResult& reply) {
        std::vector<std::string> turns = first;
        turns.push_back(reply.content);
        turns.push_back("Which of those needs the most care with tool speed?");
        return turns;
    };
    auto bucket        = std::make_shared<MemoryBucket>();
    const auto options = [&](const fs::path& directory) {
        ninfer::EngineOptions engine_options = store_engine_options(artifact);
        engine_options.context_store.directory     = directory;
        engine_options.context_store.idle_persist  = std::chrono::seconds(0);
        engine_options.context_store.remote        = bucket;
        engine_options.context_store.remote_prefix = "test/";
        return engine_options;
    };

    std::vector<ninfer::TokenId> control_tokens;
    {
        ninfer::Engine control(store_engine_options(artifact));
        const ninfer::GenerationResult reply =
            control.generate(control.prepare(store_conversation(first)), request);
        control_tokens =
            control.generate(control.prepare(store_conversation(second_turn(reply))), request)
                .generated_token_ids;
    }

    ninfer::GenerationResult reply;
    {
        ninfer::Engine engine(options(root / "a"));
        reply = engine.generate(engine.prepare(store_conversation(first)), request);
    } // shutdown writes the session and waits for its upload
    {
        std::scoped_lock lock(bucket->mutex);
        bool manifest = false;
        bool chunk    = false;
        for (const auto& [key, value] : bucket->objects) {
            manifest = manifest || key.find("test/manifests/") == 0;
            chunk    = chunk || key.find("test/chunks/") == 0;
        }
        if (!manifest || !chunk) {
            std::cerr << "shutdown did not upload the session to the bucket\n";
            return 1;
        }
    }
    {
        ninfer::Engine engine(options(root / "b")); // an empty directory
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.context_store_restored != 1 || stats.context_store_remote_downloads == 0) {
            std::cerr << "a fresh engine did not restore the session through the bucket: restored="
                      << stats.context_store_restored
                      << " downloads=" << stats.context_store_remote_downloads << '\n';
            return 1;
        }
        const ninfer::GenerationResult next =
            engine.generate(engine.prepare(store_conversation(second_turn(reply))), request);
        if (next.reused_prompt_tokens == 0 ||
            next.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
            std::cerr << "the continuation did not reuse the session restored from the bucket\n";
            return 1;
        }
        if (next.generated_token_ids != control_tokens) {
            std::cerr << "output after a restore from the bucket differs from the warm control\n";
            return 1;
        }
    }
    std::cout << "restored through the bucket: " << bucket->objects.size() << " objects\n";
    return 0;
}

} // namespace

int exercise_artifact(const char* artifact) {
    {
        ninfer::EngineOptions options             = engine_options(artifact);
        options.context_cache.device_state_slots  = 2;
        options.context_cache.max_shared_prefixes = 0;
        ninfer::Engine engine(std::move(options));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        if (const int result = exercise_registered_frontend(engine); result != 0) { return result; }
        if (const int result = exercise_stream_observations(engine); result != 0) { return result; }
        if (const int result = exercise_full_prefill_chunk(engine); result != 0) { return result; }
        if (const int result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::PrivateOnly);
            result != 0) {
            return result;
        }
        if (const int result = exercise_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_abandoned_handle_capacity(engine); result != 0) {
            return result;
        }
    }
    if (const int result = exercise_rewrite_branch(artifact); result != 0) { return result; }
    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = exercise_vision(engine); result != 0) { return result; }
    }
    if (const int result = exercise_host_restore(artifact); result != 0) { return result; }
    {
        // Production C=1/H=1 topology: repeated exact use promotes the shared prefix under one
        // cache Device slot; its Fork/Restore and the later ResponseReplay must then rotate
        // without a session identity or dropping either owner.
        ninfer::Engine engine(shared_replacement_engine_options(artifact));
        if (const int result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::SharedAlias);
            result != 0) {
            return result;
        }
    }
    if (const int result = exercise_shared_replacement_and_full_capacity_reuse(artifact);
        result != 0) {
        return result;
    }
    if (const int result = exercise_private_long_anchor_capture_and_replacement(artifact);
        result != 0) {
        return result;
    }
    if (const int result = exercise_last_private_alias_eviction(artifact); result != 0) {
        return result;
    }
    if (const int result = exercise_concurrent_resource_settlement(artifact); result != 0) {
        return result;
    }
    for (const bool mtp : {false, true}) {
        if (const int result = exercise_interleaved_prefill_all(artifact, mtp); result != 0) {
            return result;
        }
    }
    return 0;
}

namespace {

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const char* selected            = std::getenv("NINFER_PREFIX_REAL_SCENARIO");
    const std::string_view scenario = selected ? selected : "all";
    int result                      = 0;
    if (scenario == "vision") {
        ninfer::Engine engine(engine_options(artifact));
        result = exercise_vision(engine);
    } else if (scenario == "all") {
        result = exercise_artifact(artifact);
    } else if (scenario == "concurrent") {
        result = exercise_concurrent_resource_settlement(artifact);
    } else if (scenario == "interleaved-prefill") {
        result = exercise_interleaved_prefill_all(artifact, false);
    } else if (scenario == "interleaved-prefill-mtp") {
        result = exercise_interleaved_prefill_all(artifact, true);
    } else if (scenario == "interleaved-prefill-dflash2") {
        g_interleave_speculative = ninfer::SpeculativeBackend::DFlash2;
        result                   = exercise_interleaved_prefill_all(artifact, true);
    } else if (scenario == "interleaved-prefill-cancel") {
        result = exercise_interleaved_prefill_cancel(artifact, false);
    } else if (scenario == "interleaved-prefill-cancel-mtp") {
        result = exercise_interleaved_prefill_cancel(artifact, true);
    } else if (scenario == "large-probe") {
        result = exercise_large_probe(artifact);
    } else if (scenario == "large-context") {
        result = exercise_large_context(artifact);
    } else if (scenario == "interleaved-worker-failure") {
        result = exercise_interleaved_worker_failure(artifact, false);
    } else if (scenario == "interleaved-worker-failure-mtp") {
        result = exercise_interleaved_worker_failure(artifact, true);
    } else if (scenario == "two-80k") {
        result = exercise_two_80k(artifact, kPlainRoute);
    } else if (scenario == "two-80k-cublas") {
        result = exercise_two_80k(artifact, kCublasRoute);
    } else if (scenario == "over-context") {
        result = exercise_over_context(artifact);
    } else if (scenario == "interleaved-vision") {
        result = exercise_interleaved_vision(artifact, ninfer::VisionResidency::Resident, false);
    } else if (scenario == "interleaved-vision-overlay") {
        result = exercise_interleaved_vision(artifact, ninfer::VisionResidency::Overlay, false);
    } else if (scenario == "interleaved-vision-mtp") {
        result = exercise_interleaved_vision(artifact, ninfer::VisionResidency::Resident, true);
    } else if (scenario == "interleaved-two-long") {
        result = exercise_interleaved_two_long(artifact, false);
    } else if (scenario == "interleaved-two-long-mtp") {
        result = exercise_interleaved_two_long(artifact, true);
    } else if (scenario == "anthropic-prefix-regression") {
        result = exercise_anthropic_prefix_regression(artifact);
    } else if (scenario == "shared-rewrite-materialization") {
        result = exercise_shared_rewrite_materialization(artifact);
    } else if (scenario == "pressure-resume") {
        result = exercise_pressure_partial_spill_and_resume(artifact);
    } else if (scenario == "private-checkpoint-pressure") {
        result = exercise_private_checkpoint_pressure_retention(artifact);
    } else if (scenario == "source-pressure-protection") {
        result = exercise_materialization_source_pressure_protection(artifact);
    } else if (scenario == "shared-replacement") {
        result = exercise_shared_replacement_and_full_capacity_reuse(artifact);
    } else if (scenario == "shared-anchor-entitlement") {
        result = exercise_shared_anchor_entitlement(artifact);
    } else if (scenario == "automatic-private-anchors") {
        result = exercise_automatic_private_anchors(artifact);
    } else if (scenario == "endpoint-anchor-adoption") {
        result = exercise_endpoint_adopts_unaliased_anchor(artifact);
    } else if (scenario == "cancelled-prefill-200k") {
        result = exercise_cancelled_prefill_200k(artifact);
    } else if (scenario == "cancelled-prefill-progress") {
        result = exercise_cancelled_prefill_progress(artifact);
    } else if (scenario == "context-store") {
        result = exercise_context_store(artifact);
    } else if (scenario == "lazy-output-reservation") {
        result = exercise_lazy_output_reservation(artifact);
    } else if (scenario == "context-store-remote") {
        result = exercise_context_store_remote(artifact);
    } else if (scenario == "store-hydration") {
        result = exercise_store_hydration(artifact);
    } else if (scenario == "worker-failure-recovery") {
        result = exercise_worker_failure_recovery(artifact);
    } else if (scenario == "admission-planning-failure") {
        result = exercise_admission_planning_failure_is_contained(artifact);
    } else if (scenario == "private-long-anchor") {
        result = exercise_private_long_anchor_capture_and_replacement(artifact);
    } else if (scenario == "rewrite-checkpoint-shared") {
        ninfer::Engine engine(shared_replacement_engine_options(artifact));
        result = exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::SharedAlias);
    } else if (scenario == "rewrite-checkpoint") {
        auto options                              = engine_options(artifact);
        options.context_cache.device_state_slots  = 2;
        options.context_cache.max_shared_prefixes = 0;
        ninfer::Engine engine(std::move(options));
        result = exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::PrivateOnly);
    } else if (scenario == "stream-observations") {
        auto options          = engine_options(artifact);
        options.enable_vision = false;
        options.context_cache = ninfer::ContextCacheOptions{.enabled = false};
        ninfer::Engine engine(std::move(options));
        result = exercise_stream_observations(engine);
    } else {
        throw std::invalid_argument("unknown prefix integration scenario");
    }
    if (result == 0) { std::cout << "ok\n"; }
    return result;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
