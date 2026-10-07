#include "runtime/engine/host_cache.h"
#include "runtime/engine/model_instance.h"

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

    // v2: root unlimited, an ancestor limited to 10 GiB with 2 GiB charged of which 1 GiB inactive
    // and 512 MiB active file cache is reclaimable, so 9.5 GiB remain; the process sits two levels
    // below it.
    const std::filesystem::path v2 = scratch / "v2";
    write(v2 / "memory.max", "max\n");
    write(v2 / "app.slice" / "memory.max", "10737418240\n");
    write(v2 / "app.slice" / "memory.current", "2147483648\n");
    write(v2 / "app.slice" / "memory.stat",
          "anon 1\nactive_file 536870912\ninactive_file 1073741824\n");
    write(v2 / "app.slice" / "c1" / "memory.max", "max\n");
    write(v2 / "app.slice" / "c1" / "leaf" / "memory.max", "max\n");
    failures += check(cgroup_remaining_bytes(v2, "0::/app.slice/c1/leaf\n") ==
                          9 * kGiB + 512 * kMiB,
                      "an ancestor's limit, less active and inactive file cache, was not applied");
    // Each list alone: 6 GiB limit, 3 GiB charged, 1 GiB of one list reclaimable -> 4 GiB.
    write(v2 / "inactive" / "memory.max", "6442450944\n");
    write(v2 / "inactive" / "memory.current", "3221225472\n");
    write(v2 / "inactive" / "memory.stat", "inactive_file 1073741824\n");
    failures += check(cgroup_remaining_bytes(v2, "0::/inactive\n") == 4 * kGiB,
                      "inactive file cache alone was not treated as reclaimable");
    write(v2 / "active" / "memory.max", "6442450944\n");
    write(v2 / "active" / "memory.current", "3221225472\n");
    write(v2 / "active" / "memory.stat", "active_file 1073741824\n");
    failures += check(cgroup_remaining_bytes(v2, "0::/active\n") == 4 * kGiB,
                      "active file cache alone was not treated as reclaimable");
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
    write(v1 / "memory" / "grp" / "memory.stat",
          "total_active_file 536870912\ntotal_inactive_file 1073741824\n");
    // 8 GiB limit, 3 GiB charged, 1.5 GiB of it reclaimable file cache -> 6.5 GiB.
    failures += check(
        cgroup_remaining_bytes(v1, "12:cpu,cpuacct:/x\n4:memory:/grp\n0::/\n") ==
            6 * kGiB + 512 * kMiB,
        "a cgroup v1 memory limit, less active and inactive file cache, was not applied");
    write(v1 / "memory" / "free" / "memory.limit_in_bytes", "9223372036854771712\n");
    write(v1 / "memory" / "free" / "memory.usage_in_bytes", "1\n");
    failures += check(!cgroup_remaining_bytes(v1, "4:memory:/free\n").has_value(),
                      "a cgroup v1 'no limit' value was treated as a limit");

    // The total is the limit itself, not what remains of it: the tightest level on the path.
    using ninfer::runtime::cgroup_limit_bytes;
    failures += check(cgroup_limit_bytes(v2, "0::/app.slice/c1/leaf\n") == 10 * kGiB &&
                          cgroup_limit_bytes(v2, "0::/app.slice/c2\n") == 4 * kGiB &&
                          !cgroup_limit_bytes(v2, "0::/other/place\n").has_value(),
                      "the cgroup limit was not the tightest limit on the path");
    failures += check(cgroup_limit_bytes(v1, "4:memory:/grp\n") == 8 * kGiB &&
                          !cgroup_limit_bytes(v1, "4:memory:/free\n").has_value(),
                      "a cgroup v1 limit was misread");

    std::filesystem::remove_all(scratch);
    return failures;
}

// The Engine normalizes its options once before sizing. Automatic mode must accept the options as a
// caller writes them (catalogs unset), refuse explicit catalogs, and refuse a disabled cache.
int check_automatic_normalization() {
    using ninfer::runtime::normalize_engine_options;
    int failures = 0;
    ninfer::EngineOptions options;
    options.max_concurrency               = 4;
    options.context_cache.auto_host_cache = true;

    bool accepted = true;
    ninfer::EngineOptions normalized;
    try {
        normalized = normalize_engine_options(options);
    } catch (const std::exception&) { accepted = false; }
    failures += check(accepted && normalized.context_cache.auto_host_cache &&
                          normalized.context_cache.max_private_continuations == 8 &&
                          normalized.context_cache.max_shared_prefixes == 4,
                      "automatic mode rejected, or did not default, options with unset catalogs");

    const auto rejected = [&](auto&& edit) {
        ninfer::EngineOptions bad = options;
        edit(bad.context_cache);
        try {
            (void)normalize_engine_options(bad);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(rejected([](ContextCacheOptions& cache) { cache.max_private_continuations = 9; }),
                      "automatic mode accepted an explicit private-continuation capacity");
    failures += check(rejected([](ContextCacheOptions& cache) { cache.max_shared_prefixes = 9; }),
                      "automatic mode accepted an explicit shared-prefix capacity");
    failures += check(rejected([](ContextCacheOptions& cache) { cache.enabled = false; }),
                      "automatic mode was accepted with the context cache disabled");
    return failures;
}

// A scoring Engine ignores the generation-only prefill scheduling options, so values it never uses
// are normalized away instead of failing construction.
int check_scoring_normalization() {
    using ninfer::runtime::normalize_engine_options;
    ninfer::EngineOptions options;
    options.purpose           = ninfer::EnginePurpose::CausalScoring;
    options.max_prefill_lanes = 4;
    options.prefill_max_skip  = 0;
    bool accepted             = true;
    ninfer::EngineOptions normalized;
    try {
        normalized = normalize_engine_options(options);
    } catch (const std::exception&) { accepted = false; }
    return check(accepted && normalized.max_prefill_lanes == 1 &&
                     normalized.prefill_max_skip == ninfer::EngineOptions{}.prefill_max_skip,
                 "a CausalScoring Engine rejected or kept its generation-only prefill options");
}

} // namespace

int main() {
    int failures = 0;
    failures += check_automatic_normalization();
    failures += check_scoring_normalization();

    // 64 GiB free less the 3 GiB default reserve is a 61 GiB (62,464 MiB) budget; an eighth,
    // 7,808 MiB, buys 78 of the 100 MiB states.
    const ContextCacheOptions large = resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0);
    failures += check(!large.auto_host_cache && large.host_state_slots == 78 &&
                          large.host_kv_capacity_bytes == (62464 - 7800) * kMiB,
                      "64 GiB host did not split 61 GiB into 78 states and the remaining KV");
    failures += check(large.max_private_continuations == 94 && large.max_shared_prefixes == 23,
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

    // A cap bounds what is pinned however much is free, after the reserve: 10 GiB (10,240 MiB) of
    // the 61 GiB budget buys an eighth, 1,280 MiB, so 12 of the 100 MiB states and the rest KV.
    ContextCacheOptions capped_request   = requested(8);
    capped_request.host_cache_max_bytes = 10 * kGiB;
    const ContextCacheOptions capped = resolve_host_cache(capped_request, 64 * kGiB, 100 * kMiB, 8, 0);
    failures += check(capped.host_state_slots == 12 &&
                          capped.host_kv_capacity_bytes == (10240 - 1200) * kMiB,
                      "the cap did not bound the pinned budget");
    failures += check(capped.host_cache_max_bytes == 10 * kGiB, "the cap was not carried over");
    // A cap above the budget changes nothing, and so does leaving it unset.
    ContextCacheOptions loose_request   = requested(8);
    loose_request.host_cache_max_bytes = 100 * kGiB;
    const ContextCacheOptions loose = resolve_host_cache(loose_request, 64 * kGiB, 100 * kMiB, 8, 0);
    failures += check(loose.host_state_slots == large.host_state_slots &&
                          loose.host_kv_capacity_bytes == large.host_kv_capacity_bytes,
                      "a cap above the available budget changed the sizing");
    // The reserve still applies first: a small machine is not pushed up to the cap.
    const ContextCacheOptions small_machine =
        resolve_host_cache(capped_request, 8 * kGiB, 100 * kMiB, 8, 0);
    failures += check(small_machine.host_kv_capacity_bytes + small_machine.host_state_slots * 100 * kMiB ==
                          5 * kGiB,
                      "an 8 GiB machine did not keep its 3 GiB reserve under the cap");
    // A cap of zero turns the host tier off.
    ContextCacheOptions zero_request   = requested(8);
    zero_request.host_cache_max_bytes = 0;
    const ContextCacheOptions none = resolve_host_cache(zero_request, 64 * kGiB, 100 * kMiB, 8, 0);
    failures += check(none.host_state_slots == 0 && none.host_kv_capacity_bytes == 0,
                      "a zero cap did not leave no host cache");

    // Injected grafts hold shared-prefix slots of their own, on top of the sized catalog.
    const ContextCacheOptions grafted = resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 2);
    failures += check(grafted.max_shared_prefixes == 25 && grafted.max_private_continuations == 94,
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
                          huge.host_kv_capacity_bytes == (1045504 - 1280) * kMiB &&
                          huge.max_private_continuations == 144 && huge.max_shared_prefixes == 32,
                      "state-slot cap or the KV remainder is wrong on a very large host");

    // A state larger than an eighth of the budget is not pinned, and KV gets the whole budget:
    // 8 GiB free less the 3 GiB reserve is 5 GiB, whose eighth is under one 3 GiB state.
    const ContextCacheOptions oversized = resolve_host_cache(requested(8), 8 * kGiB, 3 * kGiB, 1, 0);
    failures += check(oversized.host_state_slots == 0 &&
                          oversized.host_kv_capacity_bytes == 5 * kGiB,
                      "a state slot larger than its share was still pinned");

    // Where pinned memory is charged to the GPU, the device headroom bounds the whole budget before
    // the split. 30 GiB free: (30 - 1) / 2 = 14.5 GiB = 14,848 MiB, so 18 states (1,800 MiB) and the
    // rest KV, instead of 48 GiB of budget.
    const ContextCacheOptions wddm =
        resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0, {.device_free_after_startup = 30 * kGiB});
    failures += check(wddm.host_state_slots == 18 &&
                          wddm.host_kv_capacity_bytes == (14848 - 1800) * kMiB &&
                          wddm.max_private_continuations == 34 && wddm.max_shared_prefixes == 8,
                      "device headroom did not bound the whole host budget before the split");
    // A full card: 1.5 GiB free leaves a 256 MiB budget, too small for one 100 MiB state's share.
    const ContextCacheOptions full_card =
        resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0, {.device_free_after_startup = 1536 * kMiB});
    failures += check(full_card.host_state_slots == 0 && full_card.host_kv_capacity_bytes == 256 * kMiB,
                      "a nearly full card still pinned state slots it could not afford");
    // At or under the 1 GiB device floor nothing is pinned at all.
    const ContextCacheOptions no_headroom =
        resolve_host_cache(requested(8), 64 * kGiB, 100 * kMiB, 8, 0, {.device_free_after_startup = 1 * kGiB});
    failures += check(no_headroom.host_state_slots == 0 && no_headroom.host_kv_capacity_bytes == 0,
                      "a card with no headroom still pinned host memory");

    // A share of the machine's memory bounds the budget however much is free. 50% of a 64 GiB host
    // is 32 GiB (32,768 MiB): an eighth, 4,096 MiB, buys 40 of the 100 MiB states.
    ContextCacheOptions half_request   = requested(8);
    half_request.host_cache_percent = 50;
    const ContextCacheOptions half =
        resolve_host_cache(half_request, 64 * kGiB, 100 * kMiB, 8, 0, {.total_host_bytes = 64 * kGiB});
    failures += check(half.host_state_slots == 40 &&
                          half.host_kv_capacity_bytes == (32768 - 4000) * kMiB &&
                          half.host_cache_percent == 50,
                      "50% of a 64 GiB host did not pin 32 GiB");
    // It only lowers the budget: when little is free the reserve still wins, and 100% of a host
    // whose memory is mostly taken is the free memory less the reserve, not the whole host.
    ContextCacheOptions all_request   = requested(8);
    all_request.host_cache_percent = 100;
    const ContextCacheOptions all_of_busy =
        resolve_host_cache(all_request, 20 * kGiB, 100 * kMiB, 8, 0, {.total_host_bytes = 64 * kGiB});
    const ContextCacheOptions busy = resolve_host_cache(requested(8), 20 * kGiB, 100 * kMiB, 8, 0);
    failures += check(all_of_busy.host_state_slots == busy.host_state_slots &&
                          all_of_busy.host_kv_capacity_bytes == busy.host_kv_capacity_bytes &&
                          busy.host_kv_capacity_bytes + busy.host_state_slots * 100 * kMiB == 17 * kGiB,
                      "a share larger than what is free pinned past the reserve");
    // The percent, the cap and the free memory combine by taking the smallest.
    ContextCacheOptions both_request   = half_request;
    both_request.host_cache_max_bytes = 10 * kGiB;
    const ContextCacheOptions both =
        resolve_host_cache(both_request, 64 * kGiB, 100 * kMiB, 8, 0, {.total_host_bytes = 64 * kGiB});
    failures += check(both.host_kv_capacity_bytes + both.host_state_slots * 100 * kMiB == 10 * kGiB,
                      "the cap did not apply beneath the percent");
    bool percent_needs_total = false;
    try {
        (void)resolve_host_cache(half_request, 64 * kGiB, 100 * kMiB, 8, 0);
    } catch (const std::invalid_argument&) { percent_needs_total = true; }
    failures += check(percent_needs_total, "a percent was applied without the machine's total memory");
    for (const std::uint32_t bad : {0U, 101U}) {
        ContextCacheOptions out_of_range   = requested(8);
        out_of_range.host_cache_percent = bad;
        bool refused = false;
        try {
            (void)resolve_host_cache(out_of_range, 64 * kGiB, 100 * kMiB, 8, 0,
                                     {.total_host_bytes = 64 * kGiB});
        } catch (const std::invalid_argument&) { refused = true; }
        failures += check(refused, "an out-of-range host cache percent was accepted");
    }

    // Memory that grows after sizing and is bounded by options (the media caches) is reserved on top
    // of the base reserve: 3 + 3 GiB of 64 GiB leaves 58 GiB pinned, never more than free less both.
    const ContextCacheOptions with_media = resolve_host_cache(
        requested(8), 64 * kGiB, 100 * kMiB, 8, 0, {.extra_reserve_bytes = 3 * kGiB});
    failures += check(with_media.host_kv_capacity_bytes + with_media.host_state_slots * 100 * kMiB ==
                          58 * kGiB,
                      "the extra reserve did not come out of the pinned budget");
    // Whatever the settings, the pinned total never exceeds the memory free less the reserves.
    for (const std::uint64_t free_gib : {4ULL, 9ULL, 64ULL, 300ULL}) {
        const ContextCacheOptions r = resolve_host_cache(
            requested(8), free_gib * kGiB, 75 * kMiB, 8, 0, {.extra_reserve_bytes = 3 * kGiB});
        const std::uint64_t pinned = r.host_kv_capacity_bytes + r.host_state_slots * 75 * kMiB;
        failures += check(pinned + 6 * kGiB <= std::max(free_gib * kGiB, 6 * kGiB) &&
                              (free_gib * kGiB > 6 * kGiB || pinned == 0),
                          "the pinned tier left less than the reserves free");
    }

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
    const auto total = ninfer::runtime::total_host_memory_bytes();
    failures += check(total.has_value() && *total >= *available,
                      "the total host memory probe returned nothing, or less than is available");

    return failures;
}
