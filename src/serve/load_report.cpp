#include "serve/load_report.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <utility>

namespace ninfer::serve {

LoadCapacity make_load_capacity(std::string model_id, const ninfer::EngineOptions& engine,
                                const ninfer::MemorySummary& memory) {
    LoadCapacity capacity;
    capacity.model_id             = std::move(model_id);
    capacity.max_concurrency      = engine.max_concurrency;
    capacity.max_pending_requests = engine.max_pending_requests;
    capacity.max_context          = memory.max_context;
    capacity.kv_capacity_tokens   = memory.kv_capacity;
    capacity.kv_capacity_pages    = memory.kv_capacity_page_groups;
    // Engine::options() carries resolved values: C lane slots plus H extra checkpoint slots.
    capacity.device_state_slots =
        engine.max_concurrency + engine.context_cache.device_state_slots.value_or(0);
    capacity.host_context_capacity_bytes = memory.host_context_capacity_bytes;
    return capacity;
}

std::string make_load_report(const LoadCapacity& capacity, const LoadSample& sample) {
    using Json        = nlohmann::json;
    const auto& stats = sample.stats;
    // kv_capacity is page_groups * page_tokens exactly (SequenceCapacityCurve::resolved_tokens).
    const std::uint32_t page_tokens =
        capacity.kv_capacity_pages == 0 ? 0
                                        : capacity.kv_capacity_tokens / capacity.kv_capacity_pages;
    const std::uint64_t occupied_tokens =
        static_cast<std::uint64_t>(stats.device_main_kv_occupied_pages) * page_tokens;

    Json report = {
        {"object", "ninfer.load"},
        {"model", capacity.model_id},
        {"uptime_seconds", std::isfinite(sample.uptime_seconds) ? sample.uptime_seconds : 0.0},
        {"capacity",
         Json{{"max_concurrency", capacity.max_concurrency},
              {"max_pending_requests", capacity.max_pending_requests},
              {"max_admitted_requests", static_cast<std::uint64_t>(capacity.max_concurrency) +
                                            capacity.max_pending_requests},
              {"max_context", capacity.max_context},
              {"kv_capacity_tokens", capacity.kv_capacity_tokens},
              {"kv_capacity_pages", capacity.kv_capacity_pages},
              {"kv_page_tokens", page_tokens},
              {"device_state_slots", capacity.device_state_slots},
              {"host_context_bytes", capacity.host_context_capacity_bytes}}},
        {"requests", Json{{"admitted", sample.admitted_requests},
                          {"running", stats.running_requests},
                          {"prefilling", stats.prefilling_requests},
                          {"decode_ready", stats.decode_ready_requests},
                          {"waiting", stats.waiting_requests},
                          {"paused", stats.paused_requests},
                          {"replaying", stats.replaying_requests},
                          {"materializing", stats.materializing_requests}}},
        {"occupancy", Json{{"device_main_kv_pages", stats.device_main_kv_occupied_pages},
                           {"device_main_kv_tokens", occupied_tokens},
                           {"device_state_slots", stats.device_state_occupied_slots},
                           {"host_state_slots", stats.host_state_occupied_slots},
                           {"host_kv_bytes", stats.host_kv_occupied_bytes},
                           {"host_context_bytes", stats.host_context_occupied_bytes}}},
        {"counters", Json{{"computed_prefill_tokens", stats.computed_prefill_tokens},
                          {"committed_decode_tokens", stats.committed_decode_tokens},
                          {"reused_prompt_tokens", stats.reused_prompt_tokens},
                          {"decode_rounds", stats.decode_rounds},
                          {"decode_row_rounds", stats.decode_row_rounds}}},
        // Engine host time and the last request's token pace: a supervisor's "healthy but slow"
        // signal, since /health stays 200 in that state.
        {"host", Json{{"active_seconds", static_cast<double>(stats.host_work.active_ns()) / 1e9},
                      {"device_wait_seconds",
                       static_cast<double>(stats.host_work.device_wait_ns) / 1e9}}},
        {"last_request", nullptr},
    };
    if (const auto& pace = sample.last_generation) {
        report["last_request"] = Json{{"completion_tokens", pace->completion_tokens},
                                      {"generation_wall_seconds", pace->generation_wall_seconds},
                                      {"inter_token_seconds", pace->inter_token_seconds()},
                                      {"decode_host_seconds", pace->decode_host_seconds}};
    }
    return report.dump();
}

} // namespace ninfer::serve
