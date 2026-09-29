#include "runtime/engine/slot_spill_guard.h"

#include <iostream>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

// An explicit save or restore sets the per-path mark, a spill may not fall
// below it, equal is allowed, a deeper spill raises it, an unknown path is never blocked.
int main() {
    int failures = 0;
    ninfer::SlotSpillGuard guard;
    const std::string f = "/slots/session.bin";

    failures += check(!guard.blocks(f, 31'505), "an unknown path must never block a spill");

    guard.note_authoritative(f, 78'020); // explicit save of the live copy
    const auto stale = guard.blocks(f, 31'505);
    failures += check(stale && *stale == 78'020,
                      "a stale shallower spill was not refused after an explicit save");
    failures += check(!guard.blocks(f, 78'020), "an equal-depth spill must be allowed");
    failures += check(!guard.blocks(f, 80'426), "a deeper spill must be allowed");

    guard.note_spilled(f, 80'426); // the deeper spill landed
    failures += check(guard.blocks(f, 78'020).value_or(0) == 80'426,
                      "a successful deeper spill did not raise the mark");
    guard.note_spilled(f, 40'000); // never lowers
    failures += check(guard.blocks(f, 79'000).value_or(0) == 80'426,
                      "note_spilled lowered the mark");

    guard.note_authoritative(f, 20'000); // client saved a rewound session: authoritative
    failures += check(!guard.blocks(f, 25'000),
                      "an explicit save did not lower the mark for a rewound session");
    failures += check(guard.blocks(f, 19'999).value_or(0) == 20'000,
                      "a spill below the rewound mark was not refused");

    guard.note_authoritative("/slots/other.bin", 5); // paths are independent
    failures += check(!guard.blocks(f, 20'000) && guard.blocks("/slots/other.bin", 4).has_value(),
                      "marks bled between paths");

    // A spill queued before an explicit claim of its path is superseded; one queued after is not.
    const std::string g         = "/slots/claimed.bin";
    const auto queued_before    = guard.generation(g);
    guard.claim(g);
    const auto queued_after     = guard.generation(g);
    failures += check(queued_before != guard.generation(g),
                      "a claim did not supersede a spill queued before it");
    failures += check(queued_after == guard.generation(g),
                      "a spill queued after the claim was superseded");
    failures += check(guard.generation(f) == 0U, "a claim advanced another path's generation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
