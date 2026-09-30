#pragma once

#include "ninfer/engine.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::test {

// NINFER_TEST_GRAFT=<container.bin> loads a direct_kv graft named "g" into the Engine under test.
// Unset, both helpers do nothing, so the real tests keep running without a graft file.
inline bool graft_configured() {
    const char* path = std::getenv("NINFER_TEST_GRAFT");
    return path != nullptr && *path != '\0';
}

inline void add_test_graft(EngineOptions& options) {
    if (!graft_configured()) { return; }
    options.grafts.push_back(GraftSource{.name = "g", .path = std::getenv("NINFER_TEST_GRAFT")});
}

// A direct graft carries target state only, so under a speculative backend the draft cache over its
// positions is a zero-filled stand-in. The target still verifies every proposal against the injected
// state, so these requests must run, speculate, start from the graft and complete their budget.
// Greedy text is not compared with ordinary decoding: speculative batching can flip a near tie.
inline void speculative_graft(Engine& engine, SpeculativeBackend backend, unsigned lanes) {
    if (!graft_configured()) { return; }
    const auto check = [](bool condition, const char* message) {
        if (!condition) { throw std::runtime_error(message); }
    };
    const auto grafted = [&] {
        PromptInput input;
        input.messages.push_back(ChatMessage{
            .role = ChatRole::User, .parts = {MessagePart{.text = "Who are you?"}}});
        input.options.graft = "g";
        return engine.prepare(std::move(input));
    };
    const auto options = [](std::uint32_t outputs) {
        RequestOptions request;
        request.execution.requested_output_tokens = outputs;
        request.execution.sampling.temperature    = 0.0F;
        request.stop.include_model_defaults       = false;
        return request;
    };
    const auto valid = [&](const GenerationResult& result, std::size_t outputs) {
        check(result.generated_token_ids.size() == outputs &&
                  result.finish_reason == FinishReason::OutputLimit,
              "grafted request did not complete its output budget");
        check(result.speculative.backend == backend &&
                  result.speculative.rounds + result.speculative.fallback_steps != 0,
              "grafted request bypassed speculative decoding");
        check(result.reused_prompt_tokens != 0, "grafted request did not start from the graft");
    };

    // The first use forks the pinned graft prefix's target and draft state; the second forks it again.
    const auto first  = engine.generate(grafted(), options(24));
    const auto second = engine.generate(grafted(), options(24));
    valid(first, 24);
    valid(second, 24);
    check(first.generated_token_ids == second.generated_token_ids,
          "repeating a grafted request changed its greedy output");
    // The graft is only the root: the first request reuses just the graft, and the second must
    // also find what the first left in the context cache beyond it.
    check(second.reused_prompt_tokens > first.reused_prompt_tokens,
          "a repeated grafted request reused nothing beyond the graft");

    // Lanes fork the same pinned prefix at once, each with its own budget.
    std::vector<GenerationHandle> handles;
    for (unsigned lane = 0; lane < lanes; ++lane) {
        handles.push_back(engine.submit(grafted(), options(6 + lane * 5)));
    }
    for (unsigned lane = 0; lane < lanes; ++lane) { valid(handles[lane].wait(), 6 + lane * 5); }

    // The graft must not leave anything behind that changes an ungrafted request.
    const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");
    const auto plain  = engine.generate(engine.prepare_tokens(prompt), options(16));
    check(plain.generated_token_ids.size() == 16 && plain.speculative.backend == backend &&
              plain.speculative.rounds + plain.speculative.fallback_steps != 0,
          "an ungrafted request after grafted ones did not speculate");
}

} // namespace ninfer::test
