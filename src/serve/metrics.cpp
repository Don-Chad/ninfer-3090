#include "serve/metrics.h"

#include "product/speculative_options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::array kBuckets{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1,   0.25,  0.5,   1.0,
                              2.5,   5.0,   10.0,  20.0, 40.0,  80.0, 160.0, 320.0, 640.0, 1280.0};

std::string escape_label(std::string_view value) {
    std::string out;
    for (char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

// llama.cpp --metrics compatibility series. Routers read these names and formats; keep them
// byte-for-byte stable.
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

std::string llamacpp_series(std::uint32_t max_concurrency, const RuntimeStats& live,
                            std::size_t admitted_requests) {
    const std::uint64_t admitted   = admitted_requests;
    const std::uint64_t processing = std::min<std::uint64_t>(admitted, max_concurrency);
    std::string out;
    out.reserve(1024);
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
    return out;
}

} // namespace

void Metrics::Histogram::observe(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) { return; }
    ++bins[std::lower_bound(kBuckets.begin(), kBuckets.end(), seconds) - kBuckets.begin()];
    sum += seconds;
}

void Metrics::configure(std::string model, const EngineOptions& options,
                        const MemorySummary& memory, RuntimeStats baseline) {
    model_    = std::move(model);
    options_  = options;
    memory_   = memory;
    baseline_ = baseline;
    start_seconds_ =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void Metrics::first_token(const GenerationFirstTokenObservation& observation) {
    std::lock_guard lock(mutex_);
    requests_.ttft.observe(observation.prepare_seconds + observation.elapsed_since_submit_seconds);
}

void Metrics::done(const GenerationOutcome& outcome) {
    std::lock_guard lock(mutex_);
    if (outcome.finish_reason == FinishReason::Cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.completed;
    }
    if (outcome.constraint) {
        const auto& c = *outcome.constraint;
        ++requests_.constraint_outcomes[c.terminated ? 0 : c.complete ? 1 : 2];
        ++requests_.constraint_cache[static_cast<unsigned>(c.cache)];
        requests_.constraint_prepare_seconds += c.prepare_seconds;
        requests_.constraint_mask_seconds += c.mask_seconds;
        requests_.constraint_matcher_seconds += c.matcher_seconds;
        requests_.constraint_positions += c.mask_positions;
        requests_.constraint_upload_bytes += c.mask_upload_bytes;
    }
    requests_.duration.observe(outcome.metrics.total_seconds);
    requests_.queue.observe(outcome.metrics.engine_timing.queue_wait_seconds);
    const GenerationPace pace{
        .completion_tokens       = static_cast<std::uint32_t>(std::max(outcome.completion_tokens, 0)),
        .generation_wall_seconds = outcome.metrics.generation_wall_seconds,
        .decode_host_seconds     = outcome.metrics.engine_timing.decode_host_exposed_seconds};
    // A one-token answer has no interval and leaves the last pace as it was.
    if (pace.token_intervals() > 0) {
        requests_.token_intervals += pace.token_intervals();
        requests_.token_interval_seconds += pace.generation_wall_seconds;
        requests_.last_pace = pace;
    }
}

std::optional<GenerationPace> Metrics::last_generation_pace() const {
    std::lock_guard lock(mutex_);
    return requests_.last_pace;
}

void Metrics::rejected() {
    std::lock_guard lock(mutex_);
    ++requests_.rejected;
}

void Metrics::failed(bool cancelled) {
    std::lock_guard lock(mutex_);
    if (cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.failed;
    }
}

void Metrics::response_failed() {
    std::lock_guard lock(mutex_);
    ++requests_.response_failures;
}

std::string Metrics::render(const RuntimeStats& stats, bool ready,
                            std::size_t admitted_requests) const {
    Requests requests;
    {
        std::lock_guard lock(mutex_);
        requests = requests_;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const auto header = [&](std::string_view name, std::string_view type, std::string_view help) {
        out << "# HELP ninfer_" << name << ' ' << help << '\n'
            << "# TYPE ninfer_" << name << ' ' << type << '\n';
    };
    const auto gauge = [&](std::string_view name, auto value, std::string_view help) {
        header(name, "gauge", help);
        out << "ninfer_" << name << ' ' << value << '\n';
    };
    const auto counter = [&](std::string_view name, auto current, auto baseline,
                             std::string_view help) {
        header(name, "counter", help);
        out << "ninfer_" << name << ' ' << current - baseline << '\n';
    };
    header("constraint_requests_total", "counter",
           "Settled constrained requests by language completion.");
    const char* outcomes[]    = {"terminated", "complete_interrupted", "incomplete_interrupted"};
    const char* cache_names[] = {"hit", "built", "waited"};
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_requests_total{outcome=\"" << outcomes[i] << "\"} "
            << requests.constraint_outcomes[i] << '\n';
    header("constraint_cache_total", "counter",
           "Compilation cache access for settled constrained requests.");
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_cache_total{result=\"" << cache_names[i] << "\"} "
            << requests.constraint_cache[i] << '\n';
    counter("constraint_prepare_seconds_total", requests.constraint_prepare_seconds, 0.0,
            "Observed constraint preparation work.");
    counter("constraint_mask_seconds_total", requests.constraint_mask_seconds, 0.0,
            "Observed CPU mask work, including lookahead rollback.");
    counter("constraint_matcher_seconds_total", requests.constraint_matcher_seconds, 0.0,
            "Observed matcher acceptance and discard work.");
    counter("constraint_mask_positions_total", requests.constraint_positions, 0,
            "Evaluated constrained prediction positions.");
    counter("constraint_mask_upload_bytes_total", requests.constraint_upload_bytes, 0,
            "Submitted mask payload bytes.");
    counter("constraint_draft_wait_seconds_total", stats.host_work.constraint_draft_wait_ns * 1e-9,
            baseline_.host_work.constraint_draft_wait_ns * 1e-9,
            "Draft-ready wait, counted once per batch.");
    gauge("engine_ready", ready ? 1 : 0, "Whether Engine can accept work.");
    gauge("server_start_time_seconds", start_seconds_,
          "Unix time at service attachment after warmup.");
    header("model_info", "gauge", "Resident model and speculative backend.");
    out << "ninfer_model_info{model_name=\"" << escape_label(model_) << "\",speculative_backend=\""
        << product::speculative_backend_name(options_.speculative.backend) << "\"} 1\n";
    gauge("max_concurrency", options_.max_concurrency, "Maximum resident execution lanes.");
    gauge("max_context_tokens", options_.max_context, "Per-request logical context limit.");
    gauge("spec_decode_draft_window", options_.speculative.draft_tokens,
          "Configured speculative draft window.");

#define GAUGE(field, name, help)   gauge(name, stats.field, help)
#define COUNTER(field, name, help) counter(name, stats.field, baseline_.field, help)
    GAUGE(running_requests, "requests_running", "Resident requests, including prefill and decode.");
    GAUGE(waiting_requests, "requests_waiting", "Requests waiting for initial admission.");
    GAUGE(paused_requests, "requests_paused", "Paused requests waiting for recovery.");
    GAUGE(prefilling_requests, "requests_prefilling", "Resident requests in initial prefill.");
    GAUGE(decode_ready_requests, "requests_decode_ready", "Resident requests ready for decode.");
    GAUGE(replaying_requests, "requests_replaying",
          "Resident requests rebuilding committed history.");
    GAUGE(materializing_requests, "requests_materializing",
          "Requests with a binding transaction in progress.");
    COUNTER(prompt_tokens, "prompt_tokens_total",
            "Full input tokens counted once on initial binding.");
    COUNTER(reused_prompt_tokens, "prompt_tokens_cached_total",
            "Input tokens supplied by exact checkpoints.");
    COUNTER(computed_prefill_tokens, "prefill_tokens_total",
            "Initial input tokens actually computed, excluding Replay.");
    COUNTER(generated_tokens, "generation_tokens_total",
            "Committed output tokens, including first and control tokens.");
    COUNTER(committed_decode_tokens, "decode_tokens_total",
            "Committed decode/control tokens, excluding the first token.");
    COUNTER(replayed_tokens, "replayed_tokens_total", "History tokens recomputed during recovery.");
    COUNTER(decode_rounds, "decode_rounds_total", "Decode batch executions.");
    COUNTER(decode_row_rounds, "decode_row_rounds_total", "Sum of decode batch sizes.");
    COUNTER(preemptions, "preemptions_total", "Resource-pressure pauses.");
    COUNTER(snapshot_restores, "snapshot_restores_total",
            "Restores from complete paused snapshots.");
    COUNTER(replay_restores, "replay_restores_total", "Recovery operations that rebuild history.");
    COUNTER(root_selections, "root_selections_total", "Initial bindings without checkpoint reuse.");
    COUNTER(checkpoint_selections, "checkpoint_selections_total",
            "Initial bindings with checkpoint reuse.");
    COUNTER(speculative_rounds, "spec_decode_rounds_total",
            "Native speculative verification rounds.");
    COUNTER(speculative_draft_tokens, "spec_decode_draft_tokens_total",
            "Draft tokens evaluated by native verification.");
    COUNTER(speculative_accepted_tokens, "spec_decode_accepted_tokens_total",
            "Draft tokens accepted by native verification.");
    COUNTER(speculative_fallback_steps, "spec_decode_fallback_steps_total",
            "Speculative fallback steps without drafts.");
    GAUGE(device_state_occupied_slots, "device_state_used_slots",
          "Occupied device StateImage slots.");
    gauge("device_state_capacity_slots",
          options_.max_concurrency + options_.context_cache.device_state_slots.value(),
          "Total device StateImage slots, including resident lanes.");
    GAUGE(device_main_kv_occupied_pages, "device_kv_used_pages",
          "Occupied physical Main KV pages, including retained history.");
    gauge("device_kv_capacity_pages", memory_.kv_capacity_page_groups,
          "Main KV pool capacity in pages.");
    GAUGE(device_backend_kv_occupied_pages, "device_backend_kv_used_pages",
          "Occupied speculative backend KV pages.");
    GAUGE(host_context_occupied_bytes, "host_context_used_bytes",
          "Occupied Host backing, including reservations.");
    GAUGE(host_context_reserved_bytes, "host_context_reserved_bytes",
          "Reserved Host bytes, a subset of used bytes.");
    gauge("host_context_capacity_bytes", memory_.host_context_capacity_bytes,
          "Shared pinned Host backing capacity.");
    GAUGE(host_context_peak_occupied_bytes, "host_context_peak_bytes",
          "Host backing high-water mark, including reservations.");
    GAUGE(host_state_occupied_slots, "host_state_images",
          "StateImages in the shared Host backing.");
    GAUGE(host_kv_occupied_bytes, "host_kv_used_bytes",
          "KV bytes in the shared Host backing, a subset of used bytes.");
    COUNTER(pressure_spill_pages, "context_pressure_spill_pages_total",
            "KV pages spilled to the Host to make room under capacity pressure.");

    // Series this fork's Engine publishes beyond upstream's RuntimeStats.
    COUNTER(prefill_seconds_total, "prefill_seconds_total",
            "Execution time of prefill units, advancing per unit.");
    COUNTER(decode_seconds_total, "decode_seconds_total",
            "Execution time of decode rounds, advancing per round.");
    COUNTER(waiting_cancelled_requests, "waiting_cancelled_requests_total",
            "Requests the client cancelled while they waited for admission.");
    COUNTER(waiting_expired_requests, "waiting_expired_requests_total",
            "Requests that hit the pending timeout before admission.");
    COUNTER(waiting_abandoned_seconds, "waiting_abandoned_seconds_total",
            "Time cancelled and expired requests had waited before they left the queue.");
    COUNTER(cancelled_prefills, "cancelled_prefills_total",
            "Requests cancelled while their prompt was prefilling.");
    COUNTER(cancelled_prefill_computed_tokens, "cancelled_prefill_computed_tokens_total",
            "Prompt tokens those requests had computed when they were cancelled.");
    COUNTER(engine_recoveries, "engine_recoveries_total",
            "Host-side worker failures survived by failing in-flight requests and clearing the "
            "context cache.");

    // The durable context store (--context-store); all zero when it is off.
    GAUGE(context_store_images, "context_store_images", "Sessions held by the context store.");
    GAUGE(context_store_used_bytes, "context_store_used_bytes",
          "Bytes of chunks and manifests held by the context store.");
    COUNTER(context_store_writes, "context_store_writes_total",
            "Sessions written to the context store.");
    COUNTER(context_store_write_failures, "context_store_write_failures_total",
            "Context store writes that failed.");
    COUNTER(context_store_dropped, "context_store_dropped_total",
            "Sessions not written because the write queue was full.");
    COUNTER(context_store_bytes_written, "context_store_bytes_written_total",
            "New chunk bytes written to the context store.");
    COUNTER(context_store_bytes_reused, "context_store_bytes_reused_total",
            "Chunk bytes a write found already stored and did not rewrite.");
    COUNTER(context_store_evicted, "context_store_evicted_total",
            "Sessions removed for space, age or because a newer state replaced them.");
    COUNTER(context_store_corrupt, "context_store_corrupt_total",
            "Sessions removed because they could not be read back intact.");
    GAUGE(context_store_restored, "context_store_restored_sessions",
          "Sessions restored into the cache at start-up.");
    GAUGE(context_store_restored_bytes, "context_store_restored_bytes",
          "Bytes of sessions restored at start-up.");
    GAUGE(context_store_restore_seconds, "context_store_restore_seconds",
          "Time spent restoring sessions at start-up.");
    COUNTER(context_store_hydrations, "context_store_hydrations_total",
            "Stored sessions read back into the cache for a request.");
    COUNTER(context_store_hydrated_tokens, "context_store_hydrated_tokens_total",
            "Prompt tokens requests resumed from a hydrated session instead of prefilling.");
    COUNTER(context_store_hydration_failures, "context_store_hydration_failures_total",
            "Hydrations that failed; the request was prefilled normally.");
    COUNTER(context_store_hydration_seconds, "context_store_hydration_seconds_total",
            "Time requests waited for stored sessions to be read back, failed reads included.");
    GAUGE(context_store_remote_images, "context_store_remote_images",
          "Sessions the remote bucket holds that the local directory does not hold in full.");
    COUNTER(context_store_remote_uploads, "context_store_remote_uploads_total",
            "Objects (chunks and manifests) uploaded to the remote bucket.");
    COUNTER(context_store_remote_upload_bytes, "context_store_remote_upload_bytes_total",
            "Bytes uploaded to the remote bucket.");
    COUNTER(context_store_remote_upload_failures, "context_store_remote_upload_failures_total",
            "Session uploads that failed; they are retried when the session is next written.");
    COUNTER(context_store_remote_downloads, "context_store_remote_downloads_total",
            "Chunks fetched from the remote bucket.");
    COUNTER(context_store_remote_download_bytes, "context_store_remote_download_bytes_total",
            "Bytes fetched from the remote bucket.");
    COUNTER(context_store_remote_download_failures,
            "context_store_remote_download_failures_total",
            "Remote listings or fetches that failed or returned damaged data.");
#undef GAUGE
#undef COUNTER

    header("context_transfer_bytes_total", "counter", "Completed context payload transfers.");
    const auto transfer_bytes = [&](const char* resource, const char* direction, auto current,
                                    auto baseline) {
        out << "ninfer_context_transfer_bytes_total{resource=\"" << resource << "\",direction=\""
            << direction << "\"} " << current - baseline << '\n';
    };
#define TRANSFER(resource, prefix, direction)                                                      \
    transfer_bytes(resource, #direction, stats.prefix##_##direction##_bytes,                       \
                   baseline_.prefix##_##direction##_bytes)
    TRANSFER("state", state, d2h);
    TRANSFER("state", state, h2d);
    TRANSFER("state", state, d2d);
    TRANSFER("main_kv", main_kv, d2h);
    TRANSFER("main_kv", main_kv, h2d);
    TRANSFER("main_kv", main_kv, d2d);
    TRANSFER("backend_kv", backend_kv, d2h);
    TRANSFER("backend_kv", backend_kv, h2d);
    TRANSFER("backend_kv", backend_kv, d2d);
#undef TRANSFER
    counter("context_transfer_seconds_total", stats.actual_context_transfer_seconds,
            baseline_.actual_context_transfer_seconds,
            "Accumulated context transfer operation time.");
    header("host_work_seconds_total", "counter",
           "Exclusive Engine Host phases, excluding device waits.");
    const auto host = [&](std::string_view phase, std::uint64_t value, std::uint64_t baseline) {
        out << "ninfer_host_work_seconds_total{phase=\"" << phase << "\"} "
            << static_cast<double>(value - baseline) * 1e-9 << '\n';
    };
#define HOST(field) host(#field, stats.host_work.field##_ns, baseline_.host_work.field##_ns)
    HOST(engine_boundary);
    HOST(program_submit);
    HOST(program_post);
    HOST(engine_commit_output);
    HOST(engine_maintenance);
#undef HOST
    counter("device_wait_seconds_total", stats.host_work.device_wait_ns * 1e-9,
            baseline_.host_work.device_wait_ns * 1e-9,
            "Engine wall time waiting for device work, not kernel time.");
    // The sum of the host_work phases as one series: its rate against token throughput exposes a
    // host-bound Engine (the "healthy but slow" state of #208 keeps /health at 200).
    counter("engine_host_seconds_total", stats.host_work.active_ns() * 1e-9,
            baseline_.host_work.active_ns() * 1e-9,
            "Engine host-active seconds, device wait excluded.");
    counter("token_intervals_total", requests.token_intervals, 0U,
            "Gaps between consecutive output tokens of finished requests. The rate of "
            "ninfer_token_interval_seconds_total over this one is the mean inter-token time.");
    counter("token_interval_seconds_total", requests.token_interval_seconds, 0.0,
            "Wall seconds from first to last output token of finished requests.");
    if (requests.last_pace) {
        gauge("last_request_inter_token_seconds", requests.last_pace->inter_token_seconds(),
              "Mean seconds between output tokens of the last finished request that produced at "
              "least two. A healthy Engine stays in milliseconds; a fixed stall after the first "
              "token pushes a short answer into seconds.");
        gauge("last_request_decode_host_seconds", requests.last_pace->decode_host_seconds,
              "Decode-round host time exposed to the last finished request, device wait "
              "excluded.");
    }
    header("requests_total", "counter",
           "Generation attempts entering preparation, by terminal outcome.");
    for (const auto& [outcome, count] : std::array{
             std::pair{"completed", requests.completed}, std::pair{"cancelled", requests.cancelled},
             std::pair{"failed", requests.failed}, std::pair{"rejected", requests.rejected}}) {
        out << "ninfer_requests_total{outcome=\"" << outcome << "\"} " << count << '\n';
    }
    counter("response_failures_total", requests.response_failures, 0U,
            "Response rendering, storage or transport failures after generation settlement.");
    const auto histogram = [&](std::string_view name, const Histogram& value,
                               std::string_view help) {
        header(name, "histogram", help);
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < value.bins.size(); ++i) {
            count += value.bins[i];
            out << "ninfer_" << name << "_bucket{le=\"";
            if (i == kBuckets.size()) {
                out << "+Inf";
            } else {
                out << kBuckets[i];
            }
            out << "\"} " << count << '\n';
        }
        out << "ninfer_" << name << "_sum " << value.sum << '\n'
            << "ninfer_" << name << "_count " << count << '\n';
    };
    histogram("time_to_first_token_seconds", requests.ttft,
              "Preparation through first committed token; observed once, before completion.");
    histogram("request_duration_seconds", requests.duration,
              "Generation request duration through settlement, including cancellation.");
    histogram("request_queue_seconds", requests.queue,
              "Initial Engine queue wait of settled generation requests.");
    out << llamacpp_series(options_.max_concurrency, stats, admitted_requests);
    return out.str();
}

} // namespace ninfer::serve
