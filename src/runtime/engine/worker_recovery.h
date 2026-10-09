#pragma once

// Bookkeeping the Engine worker keeps around host-side failure recovery. Header-only and free of
// CUDA so its policy is unit-testable without a device.

#include <chrono>
#include <cstdint>
#include <exception>
#include <string>

namespace ninfer::runtime {

// Counts worker failures recovered in a row and decides when one more must latch the Engine.
//
// A failure is recoverable while fewer than `maximum - 1` recoveries precede it, so the
// `maximum`-th consecutive failure latches. The streak ends in two ways:
//   * a request finishes with a published, uncancelled result (`record_success`) -- a cancelled
//     request, including one cancelled while still queued and never admitted, says nothing
//     about the Engine's health and must not call this;
//   * `healthy_window` passes since the last failure with no further failure. Each recovery is
//     verified (device synchronized without error, physical usage back at the quiescent startup
//     baseline), so a gap that long means the failures are not a tight loop, which is what the
//     cap exists to stop. This is
//     what keeps one client that resends the same failing request, every request of which ends
//     cancelled by its own timeout, from latching an Engine that recovers cleanly each time.
class RecoveryStreak {
public:
    using Clock = std::chrono::steady_clock;

    RecoveryStreak(std::uint32_t maximum, Clock::duration healthy_window) noexcept
        : maximum_(maximum), healthy_window_(healthy_window) {}

    // Whether a failure observed at `now` may be recovered from. Ends a stale streak first.
    [[nodiscard]] bool permits_recovery(Clock::time_point now) noexcept {
        if (count_ != 0 && now - last_failure_ >= healthy_window_) { count_ = 0; }
        return count_ + 1U < maximum_;
    }

    // The failure at `now` was recovered from.
    void record_recovery(Clock::time_point now) noexcept {
        ++count_;
        last_failure_ = now;
    }

    void record_success() noexcept { count_ = 0; }

    // Recoveries in the current streak.
    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }
    [[nodiscard]] std::uint32_t maximum() const noexcept { return maximum_; }

private:
    std::uint32_t maximum_;
    Clock::duration healthy_window_;
    std::uint32_t count_ = 0;
    Clock::time_point last_failure_{};
};

// Why a host-side worker failure must latch the Engine before any cleanup is attempted, or
// nullptr when recovery may proceed. A device fault comes first and always latches: a sticky CUDA
// error leaves the context unusable, so no cleanup on it can be verified. It is not a host-side
// failure, so it neither consults nor advances the streak. Otherwise the streak decides.
[[nodiscard]] inline const char* recovery_refusal(bool device_healthy, RecoveryStreak& streak,
                                                  RecoveryStreak::Clock::time_point now) noexcept {
    if (!device_healthy) { return "device fault"; }
    if (!streak.permits_recovery(now)) {
        return "failures repeated with no request completing between them";
    }
    return nullptr;
}

// Whether the Program's physical usage after recovery cleanup is the quiescent baseline taken
// once startup finished: every Device State slot and KV page (allocated or reserved), every Host
// State slot, Host KV byte and Host context byte the failed requests, their transactions and the
// cleared context cache held has been returned. Capacities are fixed at startup and not compared.
template <class Usage>
[[nodiscard]] bool returned_to_baseline(const Usage& now, const Usage& baseline) noexcept {
    return now.occupied.state_slots == baseline.occupied.state_slots &&
           now.occupied.main_kv_pages == baseline.occupied.main_kv_pages &&
           now.occupied.backend_kv_pages == baseline.occupied.backend_kv_pages &&
           now.occupied.host_bytes == baseline.occupied.host_bytes &&
           now.host_state_slots == baseline.host_state_slots &&
           now.host_kv_bytes == baseline.host_kv_bytes;
}

// "state 3/1 main_kv 40/0 ..." for the latch reason: occupied now / baseline, for each class.
template <class Usage>
[[nodiscard]] std::string describe_usage_difference(const Usage& now, const Usage& baseline) {
    std::string out;
    const auto add = [&](const char* name, auto value, auto expected) {
        if (value == expected) { return; }
        if (!out.empty()) { out += ", "; }
        out += name;
        out += ' ';
        out += std::to_string(value);
        out += '/';
        out += std::to_string(expected);
    };
    add("device_state_slots", now.occupied.state_slots, baseline.occupied.state_slots);
    add("main_kv_pages", now.occupied.main_kv_pages, baseline.occupied.main_kv_pages);
    add("backend_kv_pages", now.occupied.backend_kv_pages, baseline.occupied.backend_kv_pages);
    add("host_context_bytes", now.occupied.host_bytes, baseline.occupied.host_bytes);
    add("host_state_slots", now.host_state_slots, baseline.host_state_slots);
    add("host_kv_bytes", now.host_kv_bytes, baseline.host_kv_bytes);
    return out;
}

// The text of an in-flight exception for the operator log. Never throws: building the text
// allocates, so if that fails the result is an empty string (which never allocates) rather than
// a std::terminate from this noexcept function. Callers treat empty as "no detail".
[[nodiscard]] inline std::string exception_text(std::exception_ptr error) noexcept {
    try {
        try {
            if (error) { std::rethrow_exception(error); }
        } catch (const std::exception& exception) {
            return exception.what();
        } catch (...) {}
        return "unknown exception";
    } catch (...) {}
    return {};
}

} // namespace ninfer::runtime
