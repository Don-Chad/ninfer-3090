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
