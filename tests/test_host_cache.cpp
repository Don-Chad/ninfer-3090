#include "runtime/engine/host_cache.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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

void write(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

// A limit may sit on any ancestor of the process's cgroup, not the mount root, and the tightest
// level decides. Each case is a fabricated tree under a scratch directory.
int check_cgroup_probe() {
    using ninfer::runtime::cgroup_remaining_bytes;
    int failures = 0;
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "ninfer_host_cache_cgroup_test";
    std::filesystem::remove_all(scratch);

    // v2: root unlimited, an ancestor limited to 10 GiB with 2 GiB charged of which 1 GiB is
    // reclaimable page cache, so 9 GiB remain; the process sits two levels below it.
    const std::filesystem::path v2 = scratch / "v2";
    write(v2 / "memory.max", "max\n");
    write(v2 / "app.slice" / "memory.max", "10737418240\n");
    write(v2 / "app.slice" / "memory.current", "2147483648\n");
    write(v2 / "app.slice" / "memory.stat", "anon 1\ninactive_file 1073741824\n");
    write(v2 / "app.slice" / "c1" / "memory.max", "max\n");
    write(v2 / "app.slice" / "c1" / "leaf" / "memory.max", "max\n");
    failures += check(cgroup_remaining_bytes(v2, "0::/app.slice/c1/leaf\n") == 9 * kGiB,
                      "an ancestor's memory limit was not applied to the process's cgroup");
    // A tighter child wins over the ancestor: 4 GiB limit, 1 GiB charged -> 3 GiB.
    write(v2 / "app.slice" / "c2" / "memory.max", "4294967296\n");
    write(v2 / "app.slice" / "c2" / "memory.current", "1073741824\n");
    failures += check(cgroup_remaining_bytes(v2, "0::/app.slice/c2\n") == 3 * kGiB,
                      "the tightest limit along the cgroup path did not win");
    // No limit anywhere on the path: no answer rather than a guess.
    failures += check(!cgroup_remaining_bytes(v2, "0::/other/place\n").has_value(),
                      "an unlimited cgroup path reported a limit");
    // A deleted cgroup is still resolved by its path.
    failures += check(cgroup_remaining_bytes(v2, "0::/app.slice/c2 (deleted)\n") == 3 * kGiB,
                      "a deleted cgroup suffix broke path resolution");
    // Inside a cgroup namespace the process sees itself at the mount root.
    const std::filesystem::path ns = scratch / "ns";
    write(ns / "memory.max", "5368709120\n");
    write(ns / "memory.current", "1073741824\n");
    failures += check(cgroup_remaining_bytes(ns, "0::/\n") == 4 * kGiB,
                      "a limit at the namespace root was not applied");

    // v1 (memory controller listed among others) and hybrid, where a v2 line names no controllers.
    const std::filesystem::path v1 = scratch / "v1";
    write(v1 / "memory" / "grp" / "memory.limit_in_bytes", "8589934592\n");
    write(v1 / "memory" / "grp" / "memory.usage_in_bytes", "3221225472\n");
    write(v1 / "memory" / "grp" / "memory.stat", "total_inactive_file 1073741824\n");
    failures += check(
        cgroup_remaining_bytes(v1, "12:cpu,cpuacct:/x\n4:memory:/grp\n0::/\n") == 6 * kGiB,
        "a cgroup v1 memory limit was not applied");
    write(v1 / "memory" / "free" / "memory.limit_in_bytes", "9223372036854771712\n");
    write(v1 / "memory" / "free" / "memory.usage_in_bytes", "1\n");
    failures += check(!cgroup_remaining_bytes(v1, "4:memory:/free\n").has_value(),
                      "a cgroup v1 'no limit' value was treated as a limit");

    std::filesystem::remove_all(scratch);
    return failures;
}

} // namespace

int main() {
    int failures = 0;

    // 64 GiB free less the 1 GiB default reserve is a 63 GiB (64,512 MiB) budget; an eighth,
    // 8,064 MiB, buys 80 of the 100 MiB states.
    const ContextCacheOptions large = resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0);
    failures += check(!large.auto_host_cache && large.host_state_slots == 80 &&
                          large.host_kv_capacity_bytes == (64512 - 8000) * kMiB,
                      "64 GiB host did not split 63 GiB into 80 states and the remaining KV");
    failures += check(large.max_private_continuations == 96 && large.max_shared_prefixes == 24,
                      "catalogs did not follow the resident state count");

    // The reserve is a setting: 512 MiB leaves a 65,024 MiB budget and 81 states.
    ContextCacheOptions slim_reserve   = requested(8);
    slim_reserve.host_cache_reserve_bytes = 512 * kMiB;
    const ContextCacheOptions slim = resolve_host_cache(slim_reserve, 64 * kGiB, 100 * kMiB, 8, 0);
    failures += check(slim.host_state_slots == 81 &&
                          slim.host_kv_capacity_bytes == (65024 - 8100) * kMiB,
                      "the configured reserve did not set the budget");
    failures += check(large.device_state_slots == 8 && large.max_long_anchors_per_continuation == 2,
                      "device-side fields were not carried over");

    // Injected grafts hold shared-prefix slots of their own, on top of the sized catalog.
    const ContextCacheOptions grafted = resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 2);
    failures += check(grafted.max_shared_prefixes == 26 && grafted.max_private_continuations == 96,
                      "pinned graft prefixes were not added to the shared catalog");

    // At or under the reserve nothing is pinned and the catalogs fall back to their floors.
    const ContextCacheOptions starved = resolve_host_cache(requested(8), 1 * kGiB, 100 * kMiB, 8, 0);
    failures += check(starved.host_state_slots == 0 && starved.host_kv_capacity_bytes == 0 &&
                          starved.max_private_continuations == 16 &&
                          starved.max_shared_prefixes == 8,
                      "a host under the reserve still pinned memory or shrank the catalogs");

    // A very large host is bounded by the state-slot cap, and KV takes the rest of the budget.
    const ContextCacheOptions huge = resolve_host_cache(requested(8), 1024 * kGiB, 10 * kMiB, 8, 0);
    failures += check(huge.host_state_slots == 128 &&
                          huge.host_kv_capacity_bytes == (1047552 - 1280) * kMiB &&
                          huge.max_private_continuations == 144 && huge.max_shared_prefixes == 32,
                      "state-slot cap or the KV remainder is wrong on a very large host");

    // A state larger than an eighth of the budget is not pinned, and KV gets the whole budget:
    // 8 GiB free less the 1 GiB reserve is 7 GiB, whose eighth is under one 3 GiB state.
    const ContextCacheOptions oversized = resolve_host_cache(requested(8), 8 * kGiB, 3 * kGiB, 1, 0);
    failures += check(oversized.host_state_slots == 0 &&
                          oversized.host_kv_capacity_bytes == 7 * kGiB,
                      "a state slot larger than its share was still pinned");

    // Where pinned memory is charged to the GPU, the device headroom bounds the whole budget before
    // the split. 30 GiB free: (30 - 1) / 2 = 14.5 GiB = 14,848 MiB, so 18 states (1,800 MiB) and the
    // rest KV, instead of 48 GiB of budget.
    const ContextCacheOptions wddm =
        resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0, 30 * kGiB);
    failures += check(wddm.host_state_slots == 18 &&
                          wddm.host_kv_capacity_bytes == (14848 - 1800) * kMiB &&
                          wddm.max_private_continuations == 34 && wddm.max_shared_prefixes == 8,
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

    failures += check_cgroup_probe();

    const auto available = ninfer::runtime::available_host_memory_bytes();
    failures += check(available.has_value() && *available > 0,
                      "the host memory probe returned nothing on this platform");

    return failures;
}
