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
//     verified (device synchronized, empty physical baseline, grafts reinstalled), so a gap that
//     long means the failures are not a tight loop, which is what the cap exists to stop. This is
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
