#pragma once

// GET /v1/load: a cheap, pollable view of serving capacity, current load, and the Engine's
// monotonic token counters, for load balancers and gateways that schedule across servers.

#include "ninfer/types.h"
#include "serve/generation_pace.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::serve {

// Capacity facts that are fixed once the Engine is ready. Captured at attach time because
// Engine::memory_summary() takes the execution lock and must not be called per poll.
struct LoadCapacity {
    std::string model_id;
    std::uint32_t max_concurrency      = 0;
    std::size_t max_pending_requests   = 0;
    std::uint32_t max_context          = 0;
    std::uint32_t kv_capacity_tokens   = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_pages    = 0; // Main KV page groups backing kv_capacity_tokens.
    std::uint32_t device_state_slots   = 0;
    // The one shared Host context budget (state images, KV, pause snapshots), as actually pinned.
    std::size_t host_context_capacity_bytes = 0;
};

// One poll: ingress occupancy plus the Engine's published runtime snapshot.
struct LoadSample {
    double uptime_seconds = 0.0;
    // Requests holding server ingress capacity: preparing, waiting, running, or releasing their
    // response. Admission rejects with 429 once this reaches max_concurrency +
    // max_pending_requests.
    std::size_t admitted_requests = 0;
    ninfer::RuntimeStats stats;
    // Pace of the last finished request with two or more tokens; empty before the first one.
    std::optional<GenerationPace> last_generation;
};

[[nodiscard]] LoadCapacity make_load_capacity(std::string model_id,
                                              const ninfer::EngineOptions& engine,
                                              const ninfer::MemorySummary& memory);

// Renders the /v1/load JSON body.
[[nodiscard]] std::string make_load_report(const LoadCapacity& capacity, const LoadSample& sample);

} // namespace ninfer::serve
