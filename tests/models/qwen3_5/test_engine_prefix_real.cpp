#include "guarded_main.h"
#include "ninfer/engine.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
    const ninfer::RuntimeStats after_replacement = engine.runtime_stats();
    if (replacement.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not produce its deterministic stop token\n";
        return 1;
    }
    // Remove the exact private endpoint without publishing another shared marker. The following
    // identical Bravo prompt must therefore materialize from the retained shared prefix.
    const ninfer::GenerationResult filled =
        engine.generate(engine.prepare(plain_prompt("Another private endpoint.")), capture_request);
    const ninfer::RuntimeStats after_filler = engine.runtime_stats();
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
    const ninfer::RuntimeStats after_reuse = engine.runtime_stats();

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

ninfer::EngineOptions slot_engine_options(const char* artifact, bool auto_save,
                                           std::vector<ninfer::SlotAutoSaveEvent>* events,
                                           std::mutex* events_mutex) {
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
    options.slot_auto_save.enabled                          = auto_save;
    if (events != nullptr) {
        options.slot_auto_save.listener = [events, events_mutex](const ninfer::SlotAutoSaveEvent& e) {
            std::scoped_lock lock(*events_mutex);
            events->push_back(e);
        };
    }
    return options;
}

ninfer::PromptInput slot_conversation(const std::vector<std::string>& turns) {
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

// A saved session restored into a fresh Engine must behave exactly like the warm session it was
// saved from: the continuation reuses the restored prefix, and greedy output matches a control
// Engine that never evicted it token for token. Digest preconditions, corrupt files, and the
// auto-save spill on involuntary eviction are checked on the same artifact.
int exercise_slot_persistence(const char* artifact) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "ninfer-slot-persistence-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::string file = (directory / "session.bin").string();

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

    // Control: the warm session continues without ever leaving the device.
    std::vector<ninfer::TokenId> control_tokens;
    {
        ninfer::Engine control(slot_engine_options(artifact, false, nullptr, nullptr));
        const ninfer::GenerationResult reply =
            control.generate(control.prepare(slot_conversation(first)), request);
        const ninfer::GenerationResult next =
            control.generate(control.prepare(slot_conversation(second_turn(reply))), request);
        control_tokens = next.generated_token_ids;
        if (next.reused_prompt_tokens == 0) {
            std::cerr << "control continuation did not reuse its warm session\n";
            return 1;
        }
    }

    std::mutex events_mutex;
    std::vector<ninfer::SlotAutoSaveEvent> events;
    std::string saved_digest;
    std::uint32_t saved_tokens = 0;
    std::vector<std::string> continued;
    {
        ninfer::Engine engine(slot_engine_options(artifact, true, &events, &events_mutex));
        const ninfer::GenerationResult reply =
            engine.generate(engine.prepare(slot_conversation(first)), request);
        if (reply.slot < 0 || reply.session_digest.size() != 16) {
            std::cerr << "retained session reported no slot identity: slot=" << reply.slot << '\n';
            return 1;
        }
        const auto states = engine.slot_states();
        if (states.size() != 1 || !states[0].retained ||
            states[0].session_digest != reply.session_digest || states[0].checkpoints.empty() ||
            states[0].checkpoints.back().frontier + 1U != states[0].prompt_tokens) {
            std::cerr << "slot listing does not describe the retained session\n";
            return 1;
        }
        bool mismatch_refused = false;
        try {
            (void)engine.save_slot(static_cast<std::uint32_t>(reply.slot), file,
                                   "0000000000000000");
        } catch (const ninfer::SlotSessionMismatch&) { mismatch_refused = true; }
        if (!mismatch_refused) {
            std::cerr << "save accepted a mismatched if_digest\n";
            return 1;
        }
        const ninfer::SlotSaveResult saved = engine.save_slot(
            static_cast<std::uint32_t>(reply.slot), file, reply.session_digest);
        if (saved.session_digest != reply.session_digest || saved.tokens != states[0].prompt_tokens ||
            saved.bytes == 0 || std::filesystem::file_size(file) != saved.bytes) {
            std::cerr << "save reported an inconsistent snapshot\n";
            return 1;
        }
        saved_digest = saved.session_digest;
        saved_tokens = saved.tokens;
        continued    = second_turn(reply);

        // Grow the bound session one turn, then let an unrelated conversation evict it: the
        // auto-save must write the deeper session back to the bound file.
        const ninfer::GenerationResult grown =
            engine.generate(engine.prepare(slot_conversation(continued)), request);
        (void)engine.generate(
            engine.prepare(slot_conversation({"Name a hardwood that turns well on a lathe."})),
            request);
        const ninfer::SlotRestoreResult reread = engine.restore_slot(0, file);
        std::scoped_lock lock(events_mutex);
        if (events.size() != 1 || !events[0].error.empty() || events[0].skipped_behind_tokens ||
            events[0].tokens <= saved_tokens || reread.session_digest != grown.session_digest ||
            reread.tokens != events[0].tokens) {
            std::cerr << "eviction did not auto-save the grown session: events=" << events.size()
                      << " spilled=" << (events.empty() ? 0U : events[0].tokens)
                      << " saved=" << saved_tokens << '\n';
            return 1;
        }
    }

    // Restore the explicit first-turn snapshot into a fresh Engine and continue it.
    const std::string first_file = (directory / "first.bin").string();
    {
        ninfer::Engine writer(slot_engine_options(artifact, false, nullptr, nullptr));
        const ninfer::GenerationResult reply =
            writer.generate(writer.prepare(slot_conversation(first)), request);
        const ninfer::SlotSaveResult saved =
            writer.save_slot(static_cast<std::uint32_t>(reply.slot), first_file);
        if (saved.session_digest != saved_digest) {
            std::cerr << "the same first turn produced a different session digest\n";
            return 1;
        }
    }
    ninfer::Engine engine(slot_engine_options(artifact, false, nullptr, nullptr));
    {
        std::ofstream corrupt((directory / "corrupt.bin").string(), std::ios::binary);
        corrupt << "not a snapshot";
    }
    bool corrupt_refused = false;
    try {
        (void)engine.restore_slot(0, (directory / "corrupt.bin").string());
    } catch (const std::invalid_argument&) { corrupt_refused = true; }
    if (!corrupt_refused) {
        std::cerr << "restore accepted a corrupt snapshot\n";
        return 1;
    }
    const ninfer::SlotRestoreResult restored = engine.restore_slot(0, first_file);
    if (restored.session_digest != saved_digest || restored.tokens != saved_tokens) {
        std::cerr << "restore did not reproduce the saved session\n";
        return 1;
    }
    const ninfer::GenerationResult next =
        engine.generate(engine.prepare(slot_conversation(continued)), request);
    if (next.reused_prompt_tokens == 0 || next.generated_token_ids != control_tokens) {
        std::cerr << "restored continuation diverged from the warm control: reused="
                  << next.reused_prompt_tokens << " tokens=" << next.generated_token_ids.size()
                  << "/" << control_tokens.size() << '\n';
        return 1;
    }
    bool erase_refused = false;
    try {
        (void)engine.erase_slot(0, "0000000000000000");
    } catch (const ninfer::SlotSessionMismatch&) { erase_refused = true; }
    const std::uint32_t erased = engine.erase_slot(0, next.session_digest);
    if (!erase_refused || erased == 0 || engine.slot_states()[0].retained) {
        std::cerr << "erase did not honor its digest precondition or leave the slot vacant\n";
        return 1;
    }
    std::filesystem::remove_all(directory);
    return 0;
}

// A host-side failure in the worker must fail only the requests it touched, not latch the
// Engine. Two conversations share a system prompt and each places an explicit long anchor on the
// system boundary, where it aliases the structural shared-prefix StateImage, on one lane with one
// spare Device state slot: the shape of the entitlement miscount behind the sibling fork's issue
// #9. Whether or not that fault fires on this build, every request after a failure must be served,
// and the Engine must stay available.
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
    // Both conversations are submitted before either is waited on, so while the first runs (and
    // may fail during materialization) the second sits in the FIFO queue. A recovery must keep
    // that queued request and serve it from the cleared cache.
    std::uint32_t failures               = 0;
    std::uint32_t queued_after_recovery  = 0;
    const auto wait_for = [&](ninfer::GenerationHandle& handle) -> std::optional<bool> {
        try {
            const ninfer::GenerationResult result = handle.wait();
            if (result.generated_token_ids.size() != 1) { return std::nullopt; }
            return true;
        } catch (const ninfer::RequestError& error) {
            if (error.kind() == ninfer::RequestErrorKind::Unavailable) {
                std::cerr << "Engine latched unavailable after a worker failure: " << error.what()
                          << "\n";
                return std::nullopt;
            }
            throw;
        } catch (const std::exception&) {
            ++failures;
            return false;
        }
    };
    for (std::uint32_t round = 0; round < 5; ++round) {
        ninfer::GenerationHandle first =
            engine.submit(engine.prepare(conversation(turns[0])), request);
        ninfer::GenerationHandle second =
            engine.submit(engine.prepare(conversation(turns[1])), request);
        const std::optional<bool> first_ok  = wait_for(first);
        const std::optional<bool> second_ok = wait_for(second);
        if (!first_ok || !second_ok) {
            std::cerr << "worker-recovery round " << round << " did not complete\n";
            return 1;
        }
        if (!*first_ok) {
            if (!*second_ok) {
                std::cerr << "the request queued behind a recovered failure was not served\n";
                return 1;
            }
            ++queued_after_recovery;
        }
        for (std::size_t lineage = 0; lineage < turns.size(); ++lineage) {
            turns[lineage].push_back("Answer " + std::to_string(round) + " for lineage " +
                                     std::to_string(lineage) + ", with enough detail to fill a page.");
            turns[lineage].push_back("Continue with part " + std::to_string(round + 1) + ".");
        }
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    if (!engine.is_available() || stats.engine_recoveries != failures) {
        std::cerr << "worker recovery accounting is inconsistent: failures=" << failures
                  << " recoveries=" << stats.engine_recoveries << "\n";
        return 1;
    }
    std::cout << "worker-failure-recovery: failures=" << failures
              << " recoveries=" << stats.engine_recoveries
              << " queued_after_recovery=" << queued_after_recovery << "\n";
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
    } else if (scenario == "slot-persistence") {
        result = exercise_slot_persistence(artifact);
    } else if (scenario == "worker-failure-recovery") {
        result = exercise_worker_failure_recovery(artifact);
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
