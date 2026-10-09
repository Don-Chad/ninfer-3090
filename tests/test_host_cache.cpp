#include "runtime/engine/host_cache.h"
#include "runtime/engine/model_instance.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using ninfer::ContextCacheOptions;
using ninfer::runtime::resolve_host_capacity_bytes;

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kGiB = 1ULL << 30;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ContextCacheOptions requested() {
    ContextCacheOptions options;
    options.auto_host_cache = true;
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

// The Engine normalizes its options once before sizing. Automatic mode sizes the one Host context
// capacity, so it refuses an explicit capacity beside it, and the percent and cap only qualify it.
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
                          !normalized.context_cache.host_capacity_bytes,
                      "automatic mode rejected, or resolved early, options with no capacity");

    const auto rejected = [&](auto&& edit) {
        ninfer::EngineOptions bad = options;
        edit(bad.context_cache);
        try {
            (void)normalize_engine_options(bad);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(rejected([](ContextCacheOptions& cache) { cache.host_capacity_bytes = kGiB; }),
                      "automatic mode accepted an explicit Host context capacity");
    failures += check(rejected([](ContextCacheOptions& cache) {
                          cache.auto_host_cache    = false;
                          cache.host_cache_percent = 50;
                      }),
                      "a host cache percent was accepted without automatic mode");
    failures += check(rejected([](ContextCacheOptions& cache) {
                          cache.auto_host_cache      = false;
                          cache.host_cache_max_bytes = kGiB;
                      }),
                      "a host cache cap was accepted without automatic mode");
    failures += check(rejected([](ContextCacheOptions& cache) { cache.host_cache_percent = 0; }) &&
                          rejected([](ContextCacheOptions& cache) { cache.host_cache_percent = 101; }),
                      "an out-of-range host cache percent was accepted");
    return failures;
}

// Fork features this build does not carry are refused at Engine construction rather than dropped.
int check_unavailable_options_rejected() {
    using ninfer::runtime::normalize_engine_options;
    const auto rejected = [](auto&& edit) {
        ninfer::EngineOptions options;
        edit(options);
        try {
            (void)normalize_engine_options(options);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    int failures = 0;
    failures += check(rejected([](ninfer::EngineOptions& o) { o.devices = {0, 1}; }) &&
                          rejected([](ninfer::EngineOptions& o) { o.stage_layers = {32}; }),
                      "a multi-GPU pipeline configuration was accepted");
    failures += check(rejected([](ninfer::EngineOptions& o) {
                          o.enable_vision    = true;
                          o.vision_residency = ninfer::VisionResidency::Overlay;
                      }),
                      "overlay vision residency was accepted");
    failures += check(rejected([](ninfer::EngineOptions& o) {
                          o.grafts.push_back({.name = "g", .path = "g.safetensors"});
                      }),
                      "a prompt graft was accepted");
    failures += check(rejected([](ninfer::EngineOptions& o) {
                          o.context_store.directory = "store";
                      }),
                      "the context store was accepted");
    failures += check(!rejected([](ninfer::EngineOptions& o) { o.devices = {0}; }),
                      "a single-entry device list was refused");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_automatic_normalization();
    failures += check_unavailable_options_rejected();

    // Everything free but the 3 GiB default reserve is pinned.
    failures += check(resolve_host_capacity_bytes(requested(), 64 * kGiB) == 61 * kGiB,
                      "64 GiB free did not leave the 3 GiB default reserve");

    // The reserve is a setting.
    ContextCacheOptions slim_reserve      = requested();
    slim_reserve.host_cache_reserve_bytes = 512 * kMiB;
    failures += check(resolve_host_capacity_bytes(slim_reserve, 64 * kGiB) ==
                          64 * kGiB - 512 * kMiB,
                      "the configured reserve did not set the capacity");

    // A cap bounds what is pinned however much is free, after the reserve.
    ContextCacheOptions capped_request  = requested();
    capped_request.host_cache_max_bytes = 10 * kGiB;
    failures += check(resolve_host_capacity_bytes(capped_request, 64 * kGiB) == 10 * kGiB,
                      "the cap did not bound the pinned capacity");
    // A cap above the budget changes nothing.
    ContextCacheOptions loose_request  = requested();
    loose_request.host_cache_max_bytes = 100 * kGiB;
    failures += check(resolve_host_capacity_bytes(loose_request, 64 * kGiB) == 61 * kGiB,
                      "a cap above the available budget changed the sizing");
    // The reserve still applies first: a small machine is not pushed up to the cap.
    failures += check(resolve_host_capacity_bytes(capped_request, 8 * kGiB) == 5 * kGiB,
                      "an 8 GiB machine did not keep its 3 GiB reserve under the cap");
    // A cap of zero turns the host tier off.
    ContextCacheOptions zero_request  = requested();
    zero_request.host_cache_max_bytes = 0;
    failures += check(resolve_host_capacity_bytes(zero_request, 64 * kGiB) == 0,
                      "a zero cap did not leave no host cache");
    // At or under the reserve nothing is pinned.
    failures += check(resolve_host_capacity_bytes(requested(), 3 * kGiB) == 0 &&
                          resolve_host_capacity_bytes(requested(), 1 * kGiB) == 0,
                      "a host under the reserve still pinned memory");

    // A share of the machine's memory bounds the capacity however much is free: 50% of 64 GiB.
    ContextCacheOptions half_request = requested();
    half_request.host_cache_percent  = 50;
    failures += check(resolve_host_capacity_bytes(half_request, 64 * kGiB,
                                                  {.total_host_bytes = 64 * kGiB}) == 32 * kGiB,
                      "50% of a 64 GiB host did not pin 32 GiB");
    // It only lowers the budget: 100% of a busy host is the free memory less the reserve.
    ContextCacheOptions all_request = requested();
    all_request.host_cache_percent  = 100;
    failures += check(resolve_host_capacity_bytes(all_request, 20 * kGiB,
                                                  {.total_host_bytes = 64 * kGiB}) == 17 * kGiB,
                      "a share larger than what is free pinned past the reserve");
    // The percent, the cap and the free memory combine by taking the smallest.
    ContextCacheOptions both_request  = half_request;
    both_request.host_cache_max_bytes = 10 * kGiB;
    failures += check(resolve_host_capacity_bytes(both_request, 64 * kGiB,
                                                  {.total_host_bytes = 64 * kGiB}) == 10 * kGiB,
                      "the cap did not apply beneath the percent");
    bool percent_needs_total = false;
    try {
        (void)resolve_host_capacity_bytes(half_request, 64 * kGiB);
    } catch (const std::invalid_argument&) { percent_needs_total = true; }
    failures += check(percent_needs_total, "a percent was applied without the machine's total memory");
    for (const std::uint32_t bad : {0U, 101U}) {
        ContextCacheOptions out_of_range = requested();
        out_of_range.host_cache_percent  = bad;
        bool refused                     = false;
        try {
            (void)resolve_host_capacity_bytes(out_of_range, 64 * kGiB,
                                              {.total_host_bytes = 64 * kGiB});
        } catch (const std::invalid_argument&) { refused = true; }
        failures += check(refused, "an out-of-range host cache percent was accepted");
    }

    // Memory that grows after sizing and is bounded by options (the media caches) is reserved on
    // top of the base reserve: 3 + 3 GiB of 64 GiB leaves 58 GiB pinned.
    failures += check(resolve_host_capacity_bytes(requested(), 64 * kGiB,
                                                  {.extra_reserve_bytes = 3 * kGiB}) == 58 * kGiB,
                      "the extra reserve did not come out of the pinned capacity");
    // A huge extra reserve saturates instead of wrapping into a large budget.
    failures += check(resolve_host_capacity_bytes(
                          requested(), 64 * kGiB,
                          {.extra_reserve_bytes = std::numeric_limits<std::uint64_t>::max()}) == 0,
                      "an overflowing reserve wrapped into a nonzero capacity");

    bool unflagged_rejected = false;
    try {
        ContextCacheOptions plain = requested();
        plain.auto_host_cache     = false;
        (void)resolve_host_capacity_bytes(plain, 64 * kGiB);
    } catch (const std::logic_error&) { unflagged_rejected = true; }
    failures += check(unflagged_rejected, "sizing ran without auto_host_cache");
    bool explicit_rejected = false;
    try {
        ContextCacheOptions both_set    = requested();
        both_set.host_capacity_bytes    = kGiB;
        (void)resolve_host_capacity_bytes(both_set, 64 * kGiB);
    } catch (const std::invalid_argument&) { explicit_rejected = true; }
    failures += check(explicit_rejected, "sizing ran beside an explicit Host context capacity");

    failures += check_cgroup_probe();

    const auto available = ninfer::runtime::available_host_memory_bytes();
    failures += check(available.has_value() && *available > 0,
                      "the host memory probe returned nothing on this platform");
    const auto total = ninfer::runtime::total_host_memory_bytes();
    failures += check(total.has_value() && *total >= *available,
                      "the total host memory probe returned nothing, or less than is available");

    return failures;
}
