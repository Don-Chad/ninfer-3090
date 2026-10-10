#include "serve/metrics.h"

#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

namespace {

// A sample is a whole line: "name value".
bool has_line(const std::string& body, std::string_view line) {
    return ("\n" + body).find("\n" + std::string(line) + "\n") != std::string::npos;
}

} // namespace

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency                  = 2;
    options.context_cache.device_state_slots = 1;
    MemorySummary memory;
    memory.kv_capacity_page_groups     = 256;
    memory.host_context_capacity_bytes = 1048576;
    RuntimeStats baseline;
    baseline.generated_tokens         = 4;
    baseline.speculative_draft_tokens = 3;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, memory, baseline);
    RuntimeStats running = baseline;
    running.generated_tokens += 5;
    running.speculative_draft_tokens += 6;
    running.speculative_accepted_tokens = 3;
    running.running_requests            = 1;
    running.host_context_occupied_bytes = 4096;
    running.host_context_reserved_bytes = 1024;
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, true, 0);
    check(live.find("ninfer_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("ninfer_spec_decode_draft_tokens_total 6\n") != std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("ninfer_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("ninfer_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(metrics.render(running, true, 0) == live, "scrapes must not consume or reset counters");
    check(live.find("ninfer_host_context_used_bytes 4096\n") != std::string::npos,
          "reserved bytes must not be added to occupancy twice");

    GenerationOutcome outcome;
    outcome.finish_reason                            = FinishReason::OutputLimit;
    outcome.metrics.total_seconds                    = 2.0;
    outcome.metrics.engine_timing.queue_wait_seconds = 0.1;
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, false, 0);
    check(final.find("ninfer_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("ninfer_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("ninfer_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("ninfer_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("ninfer_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");

    // llama.cpp compatibility series: routers read these names and formats, from the Engine's
    // absolute totals rather than the post-warmup deltas.
    RuntimeStats compat = baseline;
    compat.computed_prefill_tokens = 1200;
    compat.committed_decode_tokens = 300;
    compat.prefill_seconds_total   = 1.5;
    compat.decode_seconds_total    = 4.25;
    const auto idle                = metrics.render(compat, true, 0);
    check(has_line(idle, "llamacpp:prompt_tokens_total 1200") &&
              has_line(idle, "llamacpp:prompt_seconds_total 1.500000") &&
              has_line(idle, "llamacpp:tokens_predicted_total 300") &&
              has_line(idle, "llamacpp:tokens_predicted_seconds_total 4.250000") &&
              has_line(idle, "llamacpp:requests_processing 0") &&
              has_line(idle, "llamacpp:requests_deferred 0"),
          "llamacpp token/seconds counters must come from live Engine totals");
    check(has_line(idle, "# TYPE llamacpp:prompt_tokens_total counter") &&
              has_line(idle, "# TYPE llamacpp:requests_processing gauge"),
          "llamacpp families must carry their Prometheus TYPE");
    const auto busy = metrics.render(compat, true, 5);
    check(has_line(busy, "llamacpp:requests_processing 2") &&
              has_line(busy, "llamacpp:requests_deferred 3"),
          "admitted requests beyond the lane count must be deferred, not processing");

    // Fork-only RuntimeStats series, as deltas since attachment like every ninfer_ counter.
    RuntimeStats fork                     = baseline;
    fork.pressure_spill_pages             = 9;
    fork.prefill_seconds_total            = 1.5;
    fork.decode_seconds_total             = 4.25;
    fork.waiting_cancelled_requests       = 4;
    fork.waiting_expired_requests         = 1;
    fork.waiting_abandoned_seconds        = 240.5;
    fork.cancelled_prefills               = 2;
    fork.cancelled_prefill_computed_tokens = 4096;
    fork.engine_recoveries                = 1;
    fork.context_store_images             = 5;
    fork.context_store_used_bytes         = 123456789;
    fork.context_store_writes             = 9;
    fork.context_store_bytes_reused       = 4000;
    fork.context_store_restored           = 3;
    fork.context_store_hydrations         = 2;
    fork.context_store_hydrated_tokens    = 70000;
    fork.context_store_remote_uploads     = 7;
    fork.context_store_remote_images      = 1;
    fork.main_kv_h2d_bytes                = 123456;
    fork.root_selections                  = 7;
    fork.checkpoint_selections            = 31;
    const auto series = metrics.render(fork, true, 0);
    check(has_line(series, "ninfer_context_pressure_spill_pages_total 9") &&
              has_line(series, "ninfer_prefill_seconds_total 1.5") &&
              has_line(series, "ninfer_decode_seconds_total 4.25") &&
              has_line(series, "ninfer_waiting_cancelled_requests_total 4") &&
              has_line(series, "ninfer_waiting_expired_requests_total 1") &&
              has_line(series, "ninfer_waiting_abandoned_seconds_total 240.5") &&
              has_line(series, "ninfer_cancelled_prefills_total 2") &&
              has_line(series, "ninfer_cancelled_prefill_computed_tokens_total 4096") &&
              has_line(series, "ninfer_engine_recoveries_total 1"),
          "abandoned, cancelled, recovery and phase-time series must be reported");
    check(has_line(series, "ninfer_context_store_images 5") &&
              has_line(series, "ninfer_context_store_used_bytes 123456789") &&
              has_line(series, "ninfer_context_store_writes_total 9") &&
              has_line(series, "ninfer_context_store_bytes_reused_total 4000") &&
              has_line(series, "ninfer_context_store_restored_sessions 3") &&
              has_line(series, "ninfer_context_store_hydrations_total 2") &&
              has_line(series, "ninfer_context_store_hydrated_tokens_total 70000") &&
              has_line(series, "ninfer_context_store_hydration_failures_total 0") &&
              has_line(series, "ninfer_context_store_remote_uploads_total 7") &&
              has_line(series, "ninfer_context_store_remote_images 1"),
          "context store series must be reported");
    check(has_line(series,
                   "ninfer_context_transfer_bytes_total{resource=\"main_kv\",direction=\"h2d\"} "
                   "123456") &&
              has_line(series, "ninfer_root_selections_total 7") &&
              has_line(series, "ninfer_checkpoint_selections_total 31"),
          "context transfer and selection series must be reported");
    check(series.find("ninfer:") == std::string::npos,
          "the colon-named series were renamed; recording-rule names must not be emitted");

    // Host time and token pace: a supervisor's signal for an Engine that stalls after every first
    // token while /health stays 200 (#208).
    RuntimeStats host                     = baseline;
    host.host_work.program_submit_ns      = 1'500'000'000;
    host.host_work.engine_maintenance_ns  = 500'000'000;
    const auto timed                      = metrics.render(host, true, 0);
    check(has_line(timed, "ninfer_engine_host_seconds_total 2"),
          "Engine host-active time must sum its phases, device wait excluded");
    check(!metrics.last_generation_pace() &&
              timed.find("ninfer_last_request_inter_token_seconds") == std::string::npos &&
              has_line(timed, "ninfer_token_intervals_total 0"),
          "no pace before a request with two or more tokens finished");
    GenerationOutcome slow;
    slow.finish_reason                                     = FinishReason::OutputLimit;
    slow.completion_tokens                                 = 24;
    slow.metrics.generation_wall_seconds                   = 5.75;
    slow.metrics.engine_timing.decode_host_exposed_seconds = 6.25;
    metrics.done(slow);
    GenerationOutcome single;
    single.finish_reason     = FinishReason::OutputLimit;
    single.completion_tokens = 1;
    metrics.done(single);
    const auto pace = metrics.last_generation_pace();
    check(pace && pace->completion_tokens == 24 && pace->inter_token_seconds() == 0.25,
          "a one-token answer must not replace the last multi-token pace");
    const auto paced = metrics.render(host, true, 0);
    check(has_line(paced, "ninfer_token_intervals_total 23") &&
              has_line(paced, "ninfer_token_interval_seconds_total 5.75") &&
              has_line(paced, "ninfer_last_request_inter_token_seconds 0.25") &&
              has_line(paced, "ninfer_last_request_decode_host_seconds 6.25"),
          "token pace series must be reported");

    outcome.constraint = ConstraintObservation{.complete          = true,
                                               .terminated        = false,
                                               .cache             = ConstraintCacheAccess::Built,
                                               .mask_positions    = 5,
                                               .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, false, 0);
    check(constrained.find(
              "ninfer_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("ninfer_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("ninfer_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    return failures == 0 ? 0 : 1;
}
