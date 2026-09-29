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
