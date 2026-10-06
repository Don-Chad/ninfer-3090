#pragma once

// Verification seam for the worker's failure recovery (engine-architecture §7.4). A host-side
// worker failure is a defect, and once a defect is fixed no public input reaches it any more, so
// the recovery contract cannot be exercised through the public API alone. Arming N failures makes
// the next N prefill units throw after the Program has executed them, so each recovery releases a
// lane that holds live KV pages and state. Disarmed, the cost is one relaxed atomic load per
// prefill unit.

#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace ninfer::runtime {

inline std::atomic<std::uint32_t> armed_worker_failures{0};

inline void arm_worker_failures(std::uint32_t count) noexcept {
    armed_worker_failures.store(count, std::memory_order_relaxed);
}

// The same for admission planning: arming N makes the next N inspections of a waiting request
// throw, so a test can show that the failure is delivered to that request alone and that the
// requests already running are untouched. Disarmed, one relaxed atomic load per inspection.
inline std::atomic<std::uint32_t> armed_planning_failures{0};

inline void arm_planning_failures(std::uint32_t count) noexcept {
    armed_planning_failures.store(count, std::memory_order_relaxed);
}

// Worker-only: throws once per armed planning failure.
inline void consume_armed_planning_failure() {
    std::uint32_t armed = armed_planning_failures.load(std::memory_order_relaxed);
    while (armed != 0 && !armed_planning_failures.compare_exchange_weak(
                             armed, armed - 1U, std::memory_order_relaxed)) {}
    if (armed != 0) { throw std::runtime_error("injected planning failure"); }
}

// Worker-only: throws once per armed failure.
inline void consume_armed_worker_failure() {
    std::uint32_t armed = armed_worker_failures.load(std::memory_order_relaxed);
    while (armed != 0 && !armed_worker_failures.compare_exchange_weak(
                             armed, armed - 1U, std::memory_order_relaxed)) {}
    if (armed != 0) { throw std::runtime_error("injected worker failure"); }
}

} // namespace ninfer::runtime
