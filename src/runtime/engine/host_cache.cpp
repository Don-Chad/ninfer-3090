#include "runtime/engine/host_cache.h"

#include "core/host_kv_clamp.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ninfer::runtime {
namespace {

// Of the budget, at most this share holds StateImage slots; KV pages take the rest. A state is
// ~75 MiB on the 27B beside ~17.5 KiB of KV per token, so a resident 100k-token conversation costs
// ~1.7 GiB of KV and one state; an eighth gives each conversation its tail plus two anchors with room
// to spare, and leaves the rest of the budget for the KV that actually bounds how many survive.
constexpr std::uint64_t kStateShareDenominator = 8;
constexpr std::uint64_t kMaximumHostStateSlots = 128;
constexpr std::uint32_t kMaximumSharedPrefixes = 32;

#if defined(__linux__)
std::optional<std::uint64_t> read_u64_file(const char* path) {
    std::ifstream file(path);
    std::string text;
    if (!file || !(file >> text) || text == "max") { return std::nullopt; }
    try {
        return std::stoull(text);
    } catch (...) { return std::nullopt; }
}

std::optional<std::uint64_t> meminfo_available_bytes() {
    std::ifstream file("/proc/meminfo");
    std::string key;
    std::uint64_t kib = 0;
    std::string unit;
    while (file >> key >> kib) {
        std::getline(file, unit);
        if (key == "MemAvailable:") { return kib * 1024ULL; }
    }
    return std::nullopt;
}

// Page cache charged to the cgroup is reclaimable, so it does not count against what remains.
std::uint64_t cgroup_inactive_file_bytes(const char* path, const char* key) {
    std::ifstream file(path);
    std::string name;
    std::uint64_t value = 0;
    while (file >> name >> value) {
        if (name == key) { return value; }
    }
    return 0;
}

std::optional<std::uint64_t> cgroup_remaining_bytes() {
    if (const auto limit = read_u64_file("/sys/fs/cgroup/memory.max")) {
        const auto current = read_u64_file("/sys/fs/cgroup/memory.current");
        if (!current) { return std::nullopt; }
        const std::uint64_t reclaimable =
            cgroup_inactive_file_bytes("/sys/fs/cgroup/memory.stat", "inactive_file");
        const std::uint64_t used = *current > reclaimable ? *current - reclaimable : 0;
        return *limit > used ? *limit - used : 0;
    }
    if (const auto limit = read_u64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")) {
        // cgroup v1 reports "no limit" as a value near the top of the page-aligned 63-bit range.
        if (*limit >= (1ULL << 60)) { return std::nullopt; }
        const auto usage = read_u64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
        if (!usage) { return std::nullopt; }
        const std::uint64_t reclaimable =
            cgroup_inactive_file_bytes("/sys/fs/cgroup/memory/memory.stat", "total_inactive_file");
        const std::uint64_t used = *usage > reclaimable ? *usage - reclaimable : 0;
        return *limit > used ? *limit - used : 0;
    }
    return std::nullopt;
}
#endif

} // namespace

std::optional<std::uint64_t> available_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_available_bytes();
        const std::optional<std::uint64_t> cgroup = cgroup_remaining_bytes();
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

ContextCacheOptions resolve_host_cache(const ContextCacheOptions& requested,
                                       std::uint64_t available_host_bytes,
                                       std::uint64_t state_image_bytes,
                                       std::uint32_t max_concurrency,
                                       std::uint32_t pinned_shared_prefixes,
                                       std::optional<std::uint64_t> device_free_after_startup) {
    if (!requested.auto_host_cache) {
        throw std::logic_error("host cache sizing was requested without auto_host_cache");
    }
    if (!requested.device_state_slots) {
        throw std::logic_error("host cache sizing needs normalized context cache options");
    }
    if (state_image_bytes == 0 || max_concurrency == 0) {
        throw std::invalid_argument("host cache sizing needs a StateImage size and concurrency");
    }

    // The machine serves only this process, so everything but a fixed reserve is spent. The reserve
    // covers what grows after this point: request buffers, the response store, graph
    // instantiation and module loads. Pinned pages cannot be reclaimed, so it is not a fraction.
    const std::uint64_t reserve = requested.host_cache_reserve_bytes;
    std::uint64_t budget = available_host_bytes > reserve ? available_host_bytes - reserve : 0;
    // Where pinned host memory is charged against the GPU (Windows), the same clamp the Program
    // applies to its KV buffer bounds the whole budget, before it is split. The Program clamps the
    // KV buffer against the memory left after the state slots are pinned; with the state slots
    // inside this budget that limit is always the larger, so the KV share is never cut again.
    if (device_free_after_startup) {
        budget = clamp_host_kv_reservation_bytes(budget, *device_free_after_startup, 1);
    }

    const std::uint64_t state_slots = std::min(budget / kStateShareDenominator / state_image_bytes,
                                               kMaximumHostStateSlots);
    const std::uint64_t state_bytes = state_slots * state_image_bytes;
    const std::uint64_t kv_bytes    = budget - state_bytes;

    const std::uint64_t resident_states =
        static_cast<std::uint64_t>(max_concurrency) + *requested.device_state_slots + state_slots;
    const std::uint64_t private_continuations =
        std::max<std::uint64_t>(2ULL * max_concurrency, resident_states);
    const std::uint64_t shared_floor = std::max<std::uint64_t>(
        max_concurrency, static_cast<std::uint64_t>(kMaximumExplicitPromptCacheMarkers));
    const std::uint64_t shared_prefixes =
        std::clamp<std::uint64_t>(private_continuations / 4, shared_floor,
                                  std::max<std::uint64_t>(kMaximumSharedPrefixes, shared_floor));

    ContextCacheOptions resolved       = requested;
    resolved.auto_host_cache           = false;
    resolved.host_state_slots          = static_cast<std::uint32_t>(state_slots);
    resolved.host_kv_capacity_bytes    = static_cast<std::size_t>(kv_bytes);
    resolved.max_private_continuations = static_cast<std::uint32_t>(private_continuations);
    resolved.max_shared_prefixes =
        static_cast<std::uint32_t>(shared_prefixes + pinned_shared_prefixes);
    return resolved;
}

} // namespace ninfer::runtime
