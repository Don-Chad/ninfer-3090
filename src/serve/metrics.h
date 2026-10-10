#pragma once

#include "serve/generation_pace.h"
#include "serve/generation_service.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace ninfer::serve {

// Cumulative series behind GET /metrics, in the Prometheus text exposition format.
//
// Fixed-size request aggregates. Runtime counters remain owned and published by Engine; ninfer_*
// counters report the change since attachment (after warmup).
//
// The llamacpp:-prefixed series (this fork) reproduce llama.cpp's --metrics semantics, so scrapers
// and routers built for llama.cpp read this server unchanged: computed prefill tokens (prefix-cache
// hits excluded) against prefill execution time, committed decode tokens against decode execution
// time, and admitted requests split into processing and deferred. They are the Engine's absolute
// totals, as llama.cpp reports them.
class Metrics {
public:
    void configure(std::string model, const EngineOptions& options, const MemorySummary& memory,
                   RuntimeStats baseline);
    void first_token(const GenerationFirstTokenObservation& observation);
    void done(const GenerationOutcome& outcome);
    void rejected();
    void failed(bool cancelled);
    void response_failed();
    // The most recent finished request with at least two tokens, which is the only kind that has a
    // token interval to report. Empty until one finishes.
    [[nodiscard]] std::optional<GenerationPace> last_generation_pace() const;
    // `admitted_requests` counts requests from admission to response release, so a request is
    // visible while it waits for a lane.
    [[nodiscard]] std::string render(const RuntimeStats& runtime, bool ready,
                                     std::size_t admitted_requests) const;

private:
    struct Histogram {
        // Disjoint bins while recording; cumulative buckets are produced only when scraped.
        std::array<std::uint64_t, 21> bins{};
        double sum = 0.0;
        void observe(double seconds);
    };

    struct Requests {
        std::uint64_t completed         = 0;
        std::uint64_t cancelled         = 0;
        std::uint64_t failed            = 0;
        std::uint64_t rejected          = 0;
        std::uint64_t response_failures = 0;
        std::array<std::uint64_t, 3> constraint_outcomes{};
        std::array<std::uint64_t, 3> constraint_cache{};
        double constraint_prepare_seconds     = 0;
        double constraint_mask_seconds        = 0;
        double constraint_matcher_seconds     = 0;
        std::uint64_t constraint_positions    = 0;
        std::uint64_t constraint_upload_bytes = 0;
        // Output-token gaps of finished requests and their first-to-last-token wall time.
        std::uint64_t token_intervals        = 0;
        double token_interval_seconds        = 0;
        std::optional<GenerationPace> last_pace;
        Histogram ttft;
        Histogram duration;
        Histogram queue;
    };

    std::string model_;
    EngineOptions options_;
    MemorySummary memory_;
    RuntimeStats baseline_;
    double start_seconds_ = 0.0;
    mutable std::mutex mutex_;
    Requests requests_;
};

} // namespace ninfer::serve
