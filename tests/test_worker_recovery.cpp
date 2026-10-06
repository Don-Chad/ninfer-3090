#include "runtime/engine/worker_recovery.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::runtime::exception_text;
using ninfer::runtime::RecoveryStreak;
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

    failures += check(exception_text(std::make_exception_ptr(std::runtime_error("boom"))) == "boom",
                      "std::exception text missing");
    failures += check(exception_text(std::make_exception_ptr(42)) == "unknown exception",
                      "non-std exception not reported as unknown");
    failures += check(exception_text(nullptr) == "unknown exception",
                      "null exception not reported as unknown");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
