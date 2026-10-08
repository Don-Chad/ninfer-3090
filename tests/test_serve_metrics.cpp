#include "serve/serve_metrics.h"

#include <iostream>
#include <string>
#include <string_view>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool has_sample(const std::string& body, std::string_view sample) {
    // A sample is a whole line: "name value".
    const std::string line = "\n" + std::string(sample) + "\n";
    return ("\n" + body).find(line) != std::string::npos;
}

} // namespace

int main() {
    int failures = 0;
    ServeMetrics metrics;

    ninfer::RuntimeStats live;
    live.computed_prefill_tokens = 1200;
    live.committed_decode_tokens = 300;
    live.prefill_seconds_total   = 1.5;
    live.decode_seconds_total    = 4.25;

    const std::string idle = metrics.render(2, live, 0);
    failures += check(has_sample(idle, "llamacpp:prompt_tokens_total 1200") &&
                          has_sample(idle, "llamacpp:prompt_seconds_total 1.500000") &&
                          has_sample(idle, "llamacpp:tokens_predicted_total 300") &&
                          has_sample(idle, "llamacpp:tokens_predicted_seconds_total 4.250000"),
                      "llamacpp token/seconds counters do not come from live Engine totals");
    failures += check(has_sample(idle, "llamacpp:requests_processing 0") &&
                          has_sample(idle, "llamacpp:requests_deferred 0") &&
                          has_sample(idle, "ninfer:requests_total 0"),
                      "idle server did not report zero requests");
    failures += check(idle.find("# TYPE llamacpp:prompt_tokens_total counter") != std::string::npos &&
                          idle.find("# TYPE llamacpp:requests_processing gauge") != std::string::npos,
                      "metric families are missing their Prometheus TYPE");

    live.root_selections                 = 7;
    live.private_endpoint_selections     = 31;
    live.shared_stable_prefix_selections = 2;
    live.pressure_private_owners_evicted = 4;
    live.pressure_checkpoints_dropped    = 1;
    live.pressure_searches               = 9;
    live.main_kv_h2d_bytes               = 123456;
    live.state_d2h_bytes                 = 777;
    live.host_kv_occupied_bytes          = 1U << 20;
    live.device_state_occupied_slots     = 3;
    const std::string cache              = metrics.render(2, live, 0);
    failures += check(has_sample(cache, "ninfer:context_selections_total{source=\"root\"} 7") &&
                          has_sample(cache, "ninfer:context_selections_total{source=\"private_"
                                            "endpoint\"} 31") &&
                          has_sample(cache, "ninfer:context_selections_total{source=\"shared_"
                                            "stable_prefix\"} 2") &&
                          has_sample(cache, "ninfer:context_selections_total{source=\"private_"
                                            "long_anchor\"} 0"),
                      "context selections are not reported per source");
    failures += check(has_sample(cache, "ninfer:context_pressure_events_total{event=\"private_"
                                        "owner_evicted\"} 4") &&
                          has_sample(cache, "ninfer:context_pressure_events_total{event="
                                            "\"checkpoint_dropped\"} 1") &&
                          has_sample(cache, "ninfer:context_pressure_searches_total{result="
                                            "\"started\"} 9"),
                      "pressure events are not reported");
    failures += check(has_sample(cache, "ninfer:context_transfer_bytes_total{object=\"main_kv\","
                                        "direction=\"h2d\"} 123456") &&
                          has_sample(cache, "ninfer:context_transfer_bytes_total{object=\"state\","
                                            "direction=\"d2h\"} 777"),
                      "context transfer bytes are not reported by object and direction");
    failures += check(has_sample(cache, "ninfer:context_occupancy{pool=\"host_kv_bytes\"} 1048576") &&
                          has_sample(cache, "ninfer:context_occupancy{pool=\"device_state_slots\"} 3"),
                      "context occupancy gauges are not reported");
    failures += check(cache.find("# TYPE ninfer:context_selections_total counter") !=
                              std::string::npos &&
                          cache.find("# TYPE ninfer:context_occupancy gauge") != std::string::npos,
                      "context-cache families are missing their Prometheus TYPE");

    live.waiting_cancelled_requests        = 4;
    live.waiting_abandoned_seconds         = 240.5;
    live.cancelled_prefills                = 2;
    live.cancelled_prefills_retained       = 1;
    live.cancelled_prefill_retained_tokens = 16384;
    const std::string abandoned            = metrics.render(2, live, 0);
    failures += check(has_sample(abandoned, "ninfer:waiting_cancelled_requests_total 4") &&
                          has_sample(abandoned, "ninfer:cancelled_prefills_total 2") &&
                          has_sample(abandoned, "ninfer:cancelled_prefills_retained_total 1") &&
                          has_sample(abandoned, "ninfer:cancelled_prefill_retained_tokens_total 16384"),
                      "abandoned and cancelled request series are not reported");

    live.context_store_images        = 5;
    live.context_store_used_bytes    = 123456789;
    live.context_store_writes        = 9;
    live.context_store_bytes_written = 1000;
    live.context_store_bytes_reused  = 4000;
    live.context_store_restored      = 3;
    const std::string store          = metrics.render(2, live, 0);
    failures += check(has_sample(store, "ninfer:context_store_images 5") &&
                          has_sample(store, "ninfer:context_store_used_bytes 123456789") &&
                          has_sample(store, "ninfer:context_store_writes_total 9") &&
                          has_sample(store, "ninfer:context_store_bytes_reused_total 4000") &&
                          has_sample(store, "ninfer:context_store_restored_sessions 3"),
                      "context store series are not reported");

    // Admitted requests beyond the lane count are deferred, not processing.
    const std::string busy = metrics.render(2, live, 5);
    failures += check(has_sample(busy, "llamacpp:requests_processing 2") &&
                          has_sample(busy, "llamacpp:requests_deferred 3"),
                      "admitted requests were not split into processing and deferred");

    GenerationOutcome outcome;
    outcome.metrics.prefix_cache_hit_tokens     = 900;
    outcome.metrics.speculative_draft_tokens    = 60;
    outcome.metrics.speculative_accepted_tokens = 45;
    metrics.record_done(outcome);
    metrics.record_done(outcome);
    metrics.record_failure();
    metrics.record_rejection();
    metrics.record_rejection();
    const std::string after = metrics.render(2, live, 0);
    failures += check(has_sample(after, "ninfer:requests_total 2") &&
                          has_sample(after, "ninfer:requests_failed_total 1") &&
                          has_sample(after, "ninfer:requests_rejected_total 2") &&
                          has_sample(after, "ninfer:prefix_cache_hit_tokens_total 1800") &&
                          has_sample(after, "ninfer:draft_tokens_total 120") &&
                          has_sample(after, "ninfer:draft_accepted_tokens_total 90"),
                      "completed-request series did not accumulate");
    return failures == 0 ? 0 : 1;
}
