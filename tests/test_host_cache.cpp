#include "runtime/engine/host_cache.h"

#include <iostream>
#include <stdexcept>

namespace {

using ninfer::ContextCacheOptions;
using ninfer::runtime::resolve_host_cache;

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kGiB = 1ULL << 30;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ContextCacheOptions requested(std::uint32_t device_state_slots) {
    ContextCacheOptions options;
    options.auto_host_cache  = true;
    options.device_state_slots = device_state_slots;
    options.max_long_anchors_per_continuation = 2;
    return options;
}

} // namespace

int main() {
    int failures = 0;

    // 64 GiB free: 60 GiB above the 4 GiB reserve, 48 GiB budget, 12 GiB for 100 MiB states.
    const ContextCacheOptions large = resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0);
    failures += check(!large.auto_host_cache && large.host_state_slots == 122 &&
                          large.host_kv_capacity_bytes == (49152 - 12200) * kMiB,
                      "64 GiB host did not split 48 GiB into 122 states and the remaining KV");
    failures += check(large.max_private_continuations == 138 && large.max_shared_prefixes == 32,
                      "catalogs did not follow the resident state count");
    failures += check(large.device_state_slots == 8 && large.max_long_anchors_per_continuation == 2,
                      "device-side fields were not carried over");

    // Injected grafts hold shared-prefix slots of their own, on top of the sized catalog.
    const ContextCacheOptions grafted = resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 2);
    failures += check(grafted.max_shared_prefixes == 34 && grafted.max_private_continuations == 138,
                      "pinned graft prefixes were not added to the shared catalog");

    // At or under the reserve nothing is pinned and the catalogs fall back to their floors.
    const ContextCacheOptions starved = resolve_host_cache(requested(8), 3 * kGiB, 100 * kMiB, 8, 0);
    failures += check(starved.host_state_slots == 0 && starved.host_kv_capacity_bytes == 0 &&
                          starved.max_private_continuations == 16 &&
                          starved.max_shared_prefixes == 8,
                      "a host under the reserve still pinned memory or shrank the catalogs");

    // A very large host is bounded by the state-slot cap, and KV takes the rest of the budget.
    const ContextCacheOptions huge = resolve_host_cache(requested(8), 1024 * kGiB, 10 * kMiB, 8, 0);
    failures += check(huge.host_state_slots == 128 &&
                          huge.host_kv_capacity_bytes == (835584 - 1280) * kMiB &&
                          huge.max_private_continuations == 144 && huge.max_shared_prefixes == 32,
                      "state-slot cap or the KV remainder is wrong on a very large host");

    // A state larger than a quarter of the budget is not pinned, and KV gets the whole budget:
    // 8 GiB free leaves 4 GiB above the reserve, 80% of which is 3,435,973,836 bytes.
    const ContextCacheOptions oversized = resolve_host_cache(requested(8), 8 * kGiB, 3 * kGiB, 1, 0);
    failures += check(oversized.host_state_slots == 0 &&
                          oversized.host_kv_capacity_bytes == 3435973836ULL,
                      "a state slot larger than its share was still pinned");

    // Where pinned memory is charged to the GPU, the device headroom bounds the whole budget before
    // the split. 30 GiB free: (30 - 1) / 2 = 14.5 GiB = 14,848 MiB, so 37 states (3,700 MiB) and the
    // rest KV, instead of 48 GiB of budget.
    const ContextCacheOptions wddm =
        resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0, 30 * kGiB);
    failures += check(wddm.host_state_slots == 37 &&
                          wddm.host_kv_capacity_bytes == (14848 - 3700) * kMiB &&
                          wddm.max_private_continuations == 53 && wddm.max_shared_prefixes == 13,
                      "device headroom did not bound the whole host budget before the split");
    // A full card: 1.5 GiB free leaves a 256 MiB budget, too small for one 100 MiB state's share.
    const ContextCacheOptions full_card =
        resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0, 1536 * kMiB);
    failures += check(full_card.host_state_slots == 0 && full_card.host_kv_capacity_bytes == 256 * kMiB,
                      "a nearly full card still pinned state slots it could not afford");
    // At or under the 1 GiB device floor nothing is pinned at all.
    const ContextCacheOptions no_headroom =
        resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0, 1 * kGiB);
    failures += check(no_headroom.host_state_slots == 0 && no_headroom.host_kv_capacity_bytes == 0,
                      "a card with no headroom still pinned host memory");

    bool unflagged_rejected = false;
    try {
        ContextCacheOptions plain = requested(8);
        plain.auto_host_cache     = false;
        (void)resolve_host_cache(plain, 64 * kGiB, 100 * kMiB, 8, 0);
    } catch (const std::logic_error&) { unflagged_rejected = true; }
    failures += check(unflagged_rejected, "sizing ran without auto_host_cache");

    const auto available = ninfer::runtime::available_host_memory_bytes();
    failures += check(available.has_value() && *available > 0,
                      "the host memory probe returned nothing on this platform");

    return failures;
}
