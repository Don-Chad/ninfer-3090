#include "runtime/engine/worker_recovery.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using ninfer::runtime::describe_usage_difference;
using ninfer::runtime::exception_text;
using ninfer::runtime::recovery_refusal;
using ninfer::runtime::RecoveryStreak;
using ninfer::runtime::returned_to_baseline;
using Clock = RecoveryStreak::Clock;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// Drives one failure the way the worker does: ask, then record when permitted.
bool fail(RecoveryStreak& streak, Clock::time_point at) {
    if (!streak.permits_recovery(at)) { return false; }
    streak.record_recovery(at);
    return true;
}

} // namespace

int main() {
    using std::chrono::seconds;
    const Clock::time_point t0{seconds(1000)};
    int failures = 0;

    // A tight loop of failures latches on the third, as before.
    RecoveryStreak loop(3, seconds(30));
    failures += check(fail(loop, t0) && fail(loop, t0 + seconds(1)) && !fail(loop, t0 + seconds(2)),
                      "third failure in a tight loop did not latch");
    failures += check(loop.count() == 2, "refused failure was counted as a recovery");

    // A completed request ends the streak.
    RecoveryStreak served(3, seconds(30));
    (void)fail(served, t0);
    (void)fail(served, t0 + seconds(1));
    served.record_success();
    failures += check(served.count() == 0 && fail(served, t0 + seconds(2)),
                      "successful request did not reset the streak");

    // One client resending the same failing request every ~2 minutes, each earlier attempt
    // cancelled (so none completes) never latches.
    RecoveryStreak repeating(3, seconds(30));
    bool latched = false;
    for (int i = 0; i < 20; ++i) { latched = latched || !fail(repeating, t0 + seconds(120 * i)); }
    failures += check(!latched && repeating.count() == 1,
                      "failures spaced beyond the healthy window latched the Engine");

    // The window is measured from the last failure, not the first: a failure just inside it keeps
    // the streak alive and extends it.
    RecoveryStreak spaced(3, seconds(30));
    (void)fail(spaced, t0);
    (void)fail(spaced, t0 + seconds(29));
    failures += check(!fail(spaced, t0 + seconds(58)), "failures inside the window did not latch");
    RecoveryStreak boundary(3, seconds(30));
    (void)fail(boundary, t0);
    (void)fail(boundary, t0 + seconds(29));
    failures += check(fail(boundary, t0 + seconds(59)),
                      "a gap of the full window did not end the streak");

    // The refused failure that latches leaves the count alone for the event report.
    failures += check(loop.maximum() == 3, "maximum changed");

    // A device fault latches on the very first failure, and being no host-side failure it neither
    // consumes nor advances the streak. Otherwise the streak decides, as above.
    RecoveryStreak device(3, seconds(30));
    failures += check(recovery_refusal(false, device, t0) != nullptr && device.count() == 0,
                      "a device fault did not latch on the first failure");
    failures += check(recovery_refusal(true, device, t0) == nullptr,
                      "a healthy device refused the first host-side failure");
    device.record_recovery(t0);
    device.record_recovery(t0 + seconds(1));
    failures += check(recovery_refusal(true, device, t0 + seconds(2)) != nullptr &&
                          std::string_view(recovery_refusal(false, device, t0 + seconds(2))) ==
                              "device fault",
                      "streak refusal or device precedence wrong");

    // The quiescent-baseline check compares every occupied class and ignores capacity.
    struct Resources {
        std::uint32_t state_slots = 0, main_kv_pages = 0, backend_kv_pages = 0;
        std::size_t host_bytes = 0;
    };
    struct Usage {
        Resources occupied, capacity;
        std::uint32_t host_state_slots = 0;
        std::size_t host_kv_bytes      = 0;
    };
    Usage baseline;
    baseline.occupied.host_bytes = 4096;
    baseline.capacity.state_slots = 4;
    Usage same = baseline;
    same.capacity.state_slots = 9;
    failures += check(returned_to_baseline(same, baseline) &&
                          describe_usage_difference(same, baseline).empty(),
                      "equal occupancy was reported as a leak");
    const auto leaks = [&](auto mutate) {
        Usage leaked = baseline;
        mutate(leaked);
        return !returned_to_baseline(leaked, baseline) &&
               !describe_usage_difference(leaked, baseline).empty();
    };
    failures += check(leaks([](Usage& u) { u.occupied.state_slots = 1; }) &&
                          leaks([](Usage& u) { u.occupied.main_kv_pages = 1; }) &&
                          leaks([](Usage& u) { u.occupied.backend_kv_pages = 1; }) &&
                          leaks([](Usage& u) { u.occupied.host_bytes = 0; }) &&
                          leaks([](Usage& u) { u.host_state_slots = 1; }) &&
                          leaks([](Usage& u) { u.host_kv_bytes = 1; }),
                      "a leaked resource class passed the baseline check");
    Usage pages = baseline;
    pages.occupied.main_kv_pages = 40;
    failures += check(describe_usage_difference(pages, baseline) == "main_kv_pages 40/0",
                      "baseline difference text wrong");

    failures += check(exception_text(std::make_exception_ptr(std::runtime_error("boom"))) == "boom",
                      "std::exception text missing");
    failures += check(exception_text(std::make_exception_ptr(42)) == "unknown exception",
                      "non-std exception not reported as unknown");
    failures += check(exception_text(nullptr) == "unknown exception",
                      "null exception not reported as unknown");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
