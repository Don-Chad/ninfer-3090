#pragma once

// Cumulative counters behind GET /metrics, in the Prometheus text exposition format.
//
// The llamacpp:-prefixed series reproduce llama.cpp's --metrics semantics, so scrapers and
// dashboards built for llama.cpp read this server unchanged: computed prefill tokens (prefix-cache
// hits excluded) against prefill execution time, committed decode tokens against decode execution
// time. They come from the Engine's live RuntimeStats, so rates advance during a long request
// instead of jumping at its completion. The ninfer:-prefixed series report what llama.cpp has no
// name for: completed and failed requests, prefix-cache reuse and speculative acceptance.

#include "serve/generation_service.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

class ServeMetrics {
public:
    // Accumulates one completed request. Called from the request-done funnel, so every protocol
    // and both streaming modes count exactly once.
    void record_done(const GenerationOutcome& outcome);
    // Counts one request that ended in an error instead of an outcome.
    void record_failure();
    // Counts one request refused before it reached the Engine: overload, queue timeout, an
    // invalid or oversized prompt.
    void record_rejection();

    // One complete Prometheus text body without HTTP framing. `admitted_requests` counts requests
    // from admission to response release, so a request is visible while it waits for a lane.
    [[nodiscard]] std::string render(std::uint32_t max_concurrency,
                                     const ninfer::RuntimeStats& live,
                                     std::size_t admitted_requests) const;

private:
    mutable std::mutex mutex_;
    std::uint64_t requests_total_                    = 0;
    std::uint64_t requests_failed_total_             = 0;
    std::uint64_t requests_rejected_total_           = 0;
    std::uint64_t prefix_cache_hit_tokens_total_     = 0;
    std::uint64_t speculative_draft_tokens_total_    = 0;
    std::uint64_t speculative_accepted_tokens_total_ = 0;
};

} // namespace ninfer::serve
