#pragma once

// Verification seams for the worker's failure recovery (engine-architecture §6, "Worker 失败恢复").
// A host-side worker failure is a defect, and once a defect is fixed no public input reaches it
// any more, so the recovery contract cannot be exercised through the public API alone. Each seam
// is armed with a count and fires once per count; disarmed, the cost is one relaxed atomic load at
// its site.

#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace ninfer::runtime {

namespace detail {

inline bool consume_armed(std::atomic<std::uint32_t>& armed) noexcept {
    std::uint32_t count = armed.load(std::memory_order_relaxed);
    while (count != 0 &&
           !armed.compare_exchange_weak(count, count - 1U, std::memory_order_relaxed)) {}
    return count != 0;
}

} // namespace detail

// Arming N makes the next N prefill units throw after the Program has executed them, so each
// recovery releases a lane that holds live KV pages and state (and, on the unit that completes
// the prompt, an open pending batch).
inline std::atomic<std::uint32_t> armed_worker_failures{0};

inline void arm_worker_failures(std::uint32_t count) noexcept {
    armed_worker_failures.store(count, std::memory_order_relaxed);
}

// The same for decode rounds: the next N throw after the Program has executed them and before
// their commit, leaving the round's pending batch open. A test arms it while a request is paused,
// so the failure lands on the resident lanes alone.
inline std::atomic<std::uint32_t> armed_decode_failures{0};

inline void arm_decode_failures(std::uint32_t count) noexcept {
    armed_decode_failures.store(count, std::memory_order_relaxed);
}

// Arming N makes the next N admission plans of a waiting request throw, so a test can show that
// the failure is delivered to that request alone and that the requests already running are
// untouched.
inline std::atomic<std::uint32_t> armed_planning_failures{0};

inline void arm_planning_failures(std::uint32_t count) noexcept {
    armed_planning_failures.store(count, std::memory_order_relaxed);
}

// Arming N makes the worker's device health probe report a fault on its next N uses, standing in
// for a sticky CUDA error (a real one would poison the test process's context). Only the probe
// that follows a host-side failure consults it, so it changes nothing until one occurs.
inline std::atomic<std::uint32_t> armed_device_faults{0};

inline void arm_device_faults(std::uint32_t count) noexcept {
    armed_device_faults.store(count, std::memory_order_relaxed);
}

// Worker-only: throws once per armed planning failure.
inline void consume_armed_planning_failure() {
    if (detail::consume_armed(armed_planning_failures)) {
        throw std::runtime_error("injected planning failure");
    }
}

// Worker-only: throws once per armed failure.
inline void consume_armed_worker_failure() {
    if (detail::consume_armed(armed_worker_failures)) {
        throw std::runtime_error("injected worker failure");
    }
}

// Worker-only: throws once per armed decode failure.
inline void consume_armed_decode_failure() {
    if (detail::consume_armed(armed_decode_failures)) {
        throw std::runtime_error("injected decode failure");
    }
}

// Worker-only: true once per armed device fault.
[[nodiscard]] inline bool consume_armed_device_fault() noexcept {
    return detail::consume_armed(armed_device_faults);
}

} // namespace ninfer::runtime
