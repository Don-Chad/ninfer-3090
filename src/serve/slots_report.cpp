#include "serve/slots_report.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace ninfer::serve {

std::string make_slots_report(const std::vector<ninfer::SlotState>& states,
                              std::uint32_t max_context, bool speculative) {
    using Json  = nlohmann::json;
    Json slots  = Json::array();
    for (std::size_t index = 0; index < states.size(); ++index) {
        const ninfer::SlotState& state = states[index];
        Json checkpoints               = Json::array();
        for (const ninfer::SlotCheckpoint& checkpoint : state.checkpoints) {
            checkpoints.push_back({{"frontier", checkpoint.frontier},
                                   {"session_digest", checkpoint.session_digest}});
        }
        Json entry = {{"id", index},
                      {"is_processing", state.processing},
                      {"retained", state.retained},
                      {"session_digest", state.session_digest},
                      {"checkpoints", std::move(checkpoints)},
                      {"n_ctx", max_context},
                      {"n_prompt_tokens", state.prompt_tokens},
                      {"n_prompt_tokens_cache", state.cached_tokens},
                      {"speculative", speculative}};
        // Usage describes a retained session only.
        entry["last_used_unix_ms"] = state.retained ? Json(state.last_used_unix_ms) : Json(nullptr);
        entry["reuse_count"]       = state.retained ? Json(state.reuse_count) : Json(nullptr);
        entry["reused_tokens"]     = state.retained ? Json(state.reused_tokens) : Json(nullptr);
        slots.push_back(std::move(entry));
    }
    return slots.dump();
}

} // namespace ninfer::serve
