#include "serve/serve_metrics.h"

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace ninfer::serve {

namespace {

void append_metric(std::string& out, std::string_view name, std::string_view type,
                   std::string_view help, const char* value) {
    out.append("# HELP ").append(name).append(" ").append(help).append("\n");
    out.append("# TYPE ").append(name).append(" ").append(type).append("\n");
    out.append(name).append(" ").append(value).append("\n");
}

void append_metric(std::string& out, std::string_view name, std::string_view type,
                   std::string_view help, std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%llu", static_cast<unsigned long long>(value));
    append_metric(out, name, type, help, text);
}

void append_metric(std::string& out, std::string_view name, std::string_view type,
                   std::string_view help, double value) {
    char text[48];
    std::snprintf(text, sizeof(text), "%.6f", value);
    append_metric(out, name, type, help, text);
}

// A labelled family: one HELP/TYPE header, then one sample per label value.
void append_family_header(std::string& out, std::string_view name, std::string_view type,
                          std::string_view help) {
    out.append("# HELP ").append(name).append(" ").append(help).append("\n");
    out.append("# TYPE ").append(name).append(" ").append(type).append("\n");
}

void append_sample(std::string& out, std::string_view name, std::string_view labels,
                   std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%llu", static_cast<unsigned long long>(value));
    out.append(name).append("{").append(labels).append("} ").append(text).append("\n");
}

// Context-cache series from the Engine's live RuntimeStats. Selections count admissions by the
// source they started from: `root` is a miss (full prefill from token zero), every other source is
// a reuse. Pressure events are what the planner did to inactive owners to make room.
void append_abandoned_requests(std::string& out, const ninfer::RuntimeStats& live) {
    append_metric(out, "ninfer:waiting_cancelled_requests_total", "counter",
                  "Requests the client cancelled while they waited for admission.",
                  live.waiting_cancelled_requests);
    append_metric(out, "ninfer:waiting_expired_requests_total", "counter",
                  "Requests that hit the pending timeout before admission.",
                  live.waiting_expired_requests);
    append_metric(out, "ninfer:waiting_abandoned_seconds_total", "counter",
                  "Time cancelled and expired requests had waited before they left the queue.",
                  live.waiting_abandoned_seconds);
    append_metric(out, "ninfer:cancelled_prefills_total", "counter",
                  "Requests cancelled while their prompt was prefilling.", live.cancelled_prefills);
    append_metric(out, "ninfer:cancelled_prefill_computed_tokens_total", "counter",
                  "Prompt tokens those requests had computed when they were cancelled.",
                  live.cancelled_prefill_computed_tokens);
    append_metric(out, "ninfer:cancelled_prefills_retained_total", "counter",
                  "Cancelled prefills that kept a checkpoint a retry can resume from.",
                  live.cancelled_prefills_retained);
    append_metric(out, "ninfer:cancelled_prefill_retained_tokens_total", "counter",
                  "Tokens of context held by the checkpoints cancelled prefills kept.",
                  live.cancelled_prefill_retained_tokens);
}

void append_output_reservation(std::string& out, const ninfer::RuntimeStats& live) {
    append_metric(out, "ninfer:output_reservation_growths_total", "counter",
                  "Times a request's KV reservation was extended while it decoded.",
                  live.output_reservation_growths);
    append_metric(out, "ninfer:output_reservation_exhaustions_total", "counter",
                  "Requests that stopped at their reserved output because no KV page was free.",
                  live.output_reservation_exhaustions);
}

void append_context_cache(std::string& out, const ninfer::RuntimeStats& live) {
    append_family_header(out, "ninfer:context_selections_total", "counter",
                         "Admissions by the context-cache source they started from; root is a "
                         "miss.");
    append_sample(out, "ninfer:context_selections_total", "source=\"root\"", live.root_selections);
    append_sample(out, "ninfer:context_selections_total", "source=\"private_endpoint\"",
                  live.private_endpoint_selections);
    append_sample(out, "ninfer:context_selections_total", "source=\"private_turn_closure\"",
                  live.private_turn_closure_selections);
    append_sample(out, "ninfer:context_selections_total", "source=\"private_response_replay\"",
                  live.private_response_replay_selections);
    append_sample(out, "ninfer:context_selections_total", "source=\"private_long_anchor\"",
                  live.private_long_anchor_selections);
    append_sample(out, "ninfer:context_selections_total", "source=\"shared_stable_prefix\"",
                  live.shared_stable_prefix_selections);

    append_family_header(out, "ninfer:context_pressure_events_total", "counter",
                         "What pressure planning did to inactive context-cache owners.");
    append_sample(out, "ninfer:context_pressure_events_total", "event=\"private_owner_evicted\"",
                  live.pressure_private_owners_evicted);
    append_sample(out, "ninfer:context_pressure_events_total", "event=\"private_owner_degraded\"",
                  live.pressure_private_owners_degraded);
    append_sample(out, "ninfer:context_pressure_events_total", "event=\"shared_owner_evicted\"",
                  live.pressure_shared_owners_evicted);
    append_sample(out, "ninfer:context_pressure_events_total", "event=\"shared_owner_degraded\"",
                  live.pressure_shared_owners_degraded);
    append_sample(out, "ninfer:context_pressure_events_total", "event=\"checkpoint_dropped\"",
                  live.pressure_checkpoints_dropped);

    append_family_header(out, "ninfer:context_pressure_searches_total", "counter",
                         "Pressure planning searches by how they ended.");
    append_sample(out, "ninfer:context_pressure_searches_total", "result=\"started\"",
                  live.pressure_searches);
    append_sample(out, "ninfer:context_pressure_searches_total", "result=\"budget_exhausted\"",
                  live.pressure_search_budget_exhaustions);
    append_sample(out, "ninfer:context_pressure_searches_total", "result=\"maximal_fallback\"",
                  live.pressure_maximal_fallback_selections);

    append_family_header(out, "ninfer:context_transfer_bytes_total", "counter",
                         "Context-cache bytes moved between Device and Host, by object and "
                         "direction.");
    const auto transfer = [&](const char* object, const char* direction, std::uint64_t bytes) {
        std::string labels = "object=\"";
        labels.append(object).append("\",direction=\"").append(direction).append("\"");
        append_sample(out, "ninfer:context_transfer_bytes_total", labels, bytes);
    };
    transfer("state", "d2h", live.state_d2h_bytes);
    transfer("state", "h2d", live.state_h2d_bytes);
    transfer("main_kv", "d2h", live.main_kv_d2h_bytes);
    transfer("main_kv", "h2d", live.main_kv_h2d_bytes);
    transfer("backend_kv", "d2h", live.backend_kv_d2h_bytes);
    transfer("backend_kv", "h2d", live.backend_kv_h2d_bytes);

    append_metric(out, "ninfer:context_transfer_seconds_total", "counter",
                  "Time spent on context-cache transfers that admissions waited for.",
                  live.actual_context_transfer_seconds);
    append_metric(out, "ninfer:context_historical_fork_hits_total", "counter",
                  "Admissions that forked a historical checkpoint instead of the latest endpoint.",
                  live.historical_fork_hits);

    append_family_header(out, "ninfer:context_occupancy", "gauge",
                         "Context-cache occupancy by pool: state slots and KV pages on the "
                         "Device, state slots and bytes on the Host.");
    append_sample(out, "ninfer:context_occupancy", "pool=\"device_state_slots\"",
                  live.device_state_occupied_slots);
    append_sample(out, "ninfer:context_occupancy", "pool=\"host_state_slots\"",
                  live.host_state_occupied_slots);
    append_sample(out, "ninfer:context_occupancy", "pool=\"device_main_kv_pages\"",
                  live.device_main_kv_occupied_pages);
    append_sample(out, "ninfer:context_occupancy", "pool=\"device_backend_kv_pages\"",
                  live.device_backend_kv_occupied_pages);
    append_sample(out, "ninfer:context_occupancy", "pool=\"host_kv_bytes\"",
                  static_cast<std::uint64_t>(live.host_kv_occupied_bytes));
}

// The durable context store (--context-store); all zero when it is off.
void append_context_store(std::string& out, const ninfer::RuntimeStats& live) {
    append_metric(out, "ninfer:context_store_images", "gauge",
                  "Sessions held by the context store.", live.context_store_images);
    append_metric(out, "ninfer:context_store_used_bytes", "gauge",
                  "Bytes of chunks and manifests held by the context store.",
                  live.context_store_used_bytes);
    append_metric(out, "ninfer:context_store_writes_total", "counter",
                  "Sessions written to the context store.", live.context_store_writes);
    append_metric(out, "ninfer:context_store_write_failures_total", "counter",
                  "Context store writes that failed.", live.context_store_write_failures);
    append_metric(out, "ninfer:context_store_dropped_total", "counter",
                  "Sessions not written because the write queue was full.",
                  live.context_store_dropped);
    append_metric(out, "ninfer:context_store_bytes_written_total", "counter",
                  "New chunk bytes written to the context store.",
                  live.context_store_bytes_written);
    append_metric(out, "ninfer:context_store_bytes_reused_total", "counter",
                  "Chunk bytes a write found already stored and did not rewrite.",
                  live.context_store_bytes_reused);
    append_metric(out, "ninfer:context_store_evicted_total", "counter",
                  "Sessions removed for space, age or because a newer state replaced them.",
                  live.context_store_evicted);
    append_metric(out, "ninfer:context_store_corrupt_total", "counter",
                  "Sessions removed because they could not be read back intact.",
                  live.context_store_corrupt);
    append_metric(out, "ninfer:context_store_restored_sessions", "gauge",
                  "Sessions restored into the cache at start-up.", live.context_store_restored);
    append_metric(out, "ninfer:context_store_restored_bytes", "gauge",
                  "Bytes of sessions restored at start-up.", live.context_store_restored_bytes);
    append_metric(out, "ninfer:context_store_restore_seconds", "gauge",
                  "Time spent restoring sessions at start-up.", live.context_store_restore_seconds);
}

} // namespace

void ServeMetrics::record_done(const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const std::lock_guard lock(mutex_);
    ++requests_total_;
    prefix_cache_hit_tokens_total_ += metrics.prefix_cache_hit_tokens;
    speculative_draft_tokens_total_ += metrics.speculative_draft_tokens;
    speculative_accepted_tokens_total_ += metrics.speculative_accepted_tokens;
}

void ServeMetrics::record_failure() {
    const std::lock_guard lock(mutex_);
    ++requests_failed_total_;
}

void ServeMetrics::record_rejection() {
    const std::lock_guard lock(mutex_);
    ++requests_rejected_total_;
}

std::string ServeMetrics::render(std::uint32_t max_concurrency, const ninfer::RuntimeStats& live,
                                 std::size_t admitted_requests) const {
    const std::uint64_t admitted   = admitted_requests;
    const std::uint64_t processing = std::min<std::uint64_t>(admitted, max_concurrency);
    std::string out;
    out.reserve(2048);
    append_metric(out, "llamacpp:prompt_tokens_total", "counter",
                  "Number of prompt tokens processed (prefix-cache hits excluded).",
                  live.computed_prefill_tokens);
    append_metric(out, "llamacpp:prompt_seconds_total", "counter",
                  "Prompt process time in seconds.", live.prefill_seconds_total);
    append_metric(out, "llamacpp:tokens_predicted_total", "counter",
                  "Number of generation tokens processed.", live.committed_decode_tokens);
    append_metric(out, "llamacpp:tokens_predicted_seconds_total", "counter",
                  "Predict process time in seconds.", live.decode_seconds_total);
    append_metric(out, "llamacpp:requests_processing", "gauge",
                  "Number of requests processing.", processing);
    append_metric(out, "llamacpp:requests_deferred", "gauge", "Number of requests deferred.",
                  admitted - processing);

    append_abandoned_requests(out, live);
    append_output_reservation(out, live);
    append_context_cache(out, live);
    append_context_store(out, live);

    const std::lock_guard lock(mutex_);
    append_metric(out, "ninfer:requests_total", "counter", "Requests completed with an outcome.",
                  requests_total_);
    append_metric(out, "ninfer:requests_failed_total", "counter",
                  "Accepted requests that ended in an error.", requests_failed_total_);
    append_metric(out, "ninfer:requests_rejected_total", "counter",
                  "Generation requests rejected during preparation, one per request_rejected "
                  "log event (overload, invalid or oversized prompt or media). Unparseable and "
                  "oversized HTTP bodies are not counted; failures after acceptance, including "
                  "a queue timeout after submission, count in ninfer:requests_failed_total.",
                  requests_rejected_total_);
    append_metric(out, "ninfer:prefix_cache_hit_tokens_total", "counter",
                  "Prompt tokens served from the context cache instead of prefill.",
                  prefix_cache_hit_tokens_total_);
    append_metric(out, "ninfer:draft_tokens_total", "counter",
                  "Speculative draft tokens proposed.", speculative_draft_tokens_total_);
    append_metric(out, "ninfer:draft_accepted_tokens_total", "counter",
                  "Speculative draft tokens accepted.", speculative_accepted_tokens_total_);
    return out;
}

} // namespace ninfer::serve
