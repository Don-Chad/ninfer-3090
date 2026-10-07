#include "runtime/engine/host_cache.h"

#include "core/host_kv_clamp.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

std::optional<std::uint64_t> read_u64_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string text;
    if (!file || !(file >> text) || text == "max") { return std::nullopt; }
    try {
        return std::stoull(text);
    } catch (...) { return std::nullopt; }
}

// File-backed page cache charged to a cgroup (active and inactive lists) is reclaimable, so it does
// not count against what remains. Counting only the inactive list would undercount what is free
// right after a large artifact was read, because pages read more than once sit on the active list.
std::uint64_t reclaimable_file_bytes(const std::filesystem::path& path, const char* active_key,
                                     const char* inactive_key) {
    std::ifstream file(path);
    std::string name;
    std::uint64_t value = 0;
    std::uint64_t total = 0;
    while (file >> name >> value) {
        if (name == active_key || name == inactive_key) { total += value; }
    }
    return total;
}

// What one cgroup directory still allows: its limit less its non-reclaimable use. Empty when the
// directory sets no limit (or does not exist), which leaves the decision to its ancestors.
std::optional<std::uint64_t> limit_under(const std::filesystem::path& dir, bool v2) {
    const auto limit = read_u64_file(dir / (v2 ? "memory.max" : "memory.limit_in_bytes"));
    // cgroup v1 reports "no limit" as a value near the top of the page-aligned 63-bit range.
    if (!limit || (!v2 && *limit >= (1ULL << 60))) { return std::nullopt; }
    return limit;
}

enum class CgroupMeasure { Remaining, Limit };

std::optional<std::uint64_t> measure_under(const std::filesystem::path& dir, bool v2,
                                           CgroupMeasure measure) {
    const auto limit = limit_under(dir, v2);
    if (!limit || measure == CgroupMeasure::Limit) { return limit; }
    const auto used = read_u64_file(dir / (v2 ? "memory.current" : "memory.usage_in_bytes"));
    if (!used) { return std::nullopt; }
    const std::uint64_t reclaimable = reclaimable_file_bytes(
        dir / "memory.stat", v2 ? "active_file" : "total_active_file",
        v2 ? "inactive_file" : "total_inactive_file");
    const std::uint64_t charged = *used > reclaimable ? *used - reclaimable : 0;
    return *limit > charged ? *limit - charged : 0;
}

std::vector<std::string> path_components(std::string path) {
    if (const auto deleted = path.find(" (deleted)"); deleted != std::string::npos) {
        path.erase(deleted);
    }
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(path);
    while (std::getline(stream, part, '/')) {
        if (!part.empty()) { parts.push_back(part); }
    }
    return parts;
}

// The tightest remainder (or limit) over `base/<path>` and every ancestor up to `base`.
std::optional<std::uint64_t> tightest(const std::filesystem::path& base,
                                      const std::vector<std::string>& parts, bool v2,
                                      CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    for (std::size_t depth = parts.size() + 1; depth-- > 0;) {
        std::filesystem::path dir = base;
        for (std::size_t index = 0; index < depth; ++index) { dir /= parts[index]; }
        if (const auto remaining = measure_under(dir, v2, measure)) {
            best = best ? std::min(*best, *remaining) : *remaining;
        }
    }
    return best;
}

#if defined(__linux__)
std::optional<std::uint64_t> meminfo_bytes(std::string_view wanted) {
    std::ifstream file("/proc/meminfo");
    std::string key;
    std::uint64_t kib = 0;
    std::string unit;
    while (file >> key >> kib) {
        std::getline(file, unit);
        if (key == wanted) { return kib * 1024ULL; }
    }
    return std::nullopt;
}
#endif

} // namespace

namespace {

std::optional<std::uint64_t> cgroup_tightest(const std::filesystem::path& cgroup_root,
                                             std::string_view proc_self_cgroup,
                                             CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    const auto take = [&](std::optional<std::uint64_t> remaining) {
        if (remaining) { best = best ? std::min(*best, *remaining) : *remaining; }
    };
    std::istringstream lines{std::string(proc_self_cgroup)};
    std::string line;
    while (std::getline(lines, line)) {
        // "<hierarchy>:<controllers>:<path>"; the path may itself contain colons.
        const std::size_t first = line.find(':');
        const std::size_t second = first == std::string::npos ? first : line.find(':', first + 1);
        if (second == std::string::npos) { continue; }
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::vector<std::string> parts = path_components(line.substr(second + 1));
        if (line.compare(0, first, "0") == 0 && controllers.empty()) {
            take(tightest(cgroup_root, parts, true, measure));
        } else if ((',' + controllers + ',').find(",memory,") != std::string::npos) {
            take(tightest(cgroup_root / "memory", parts, false, measure));
        }
    }
    return best;
}

} // namespace

std::optional<std::uint64_t> cgroup_remaining_bytes(const std::filesystem::path& cgroup_root,
                                                    std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Remaining);
}

std::optional<std::uint64_t> cgroup_limit_bytes(const std::filesystem::path& cgroup_root,
                                                std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Limit);
}

std::optional<std::uint64_t> available_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemAvailable:");
        std::ifstream self("/proc/self/cgroup");
        std::ostringstream self_text;
        self_text << self.rdbuf();
        const std::optional<std::uint64_t> cgroup =
            cgroup_remaining_bytes("/sys/fs/cgroup", self_text.str());
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

std::optional<std::uint64_t> total_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullTotalPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemTotal:");
        std::ifstream self("/proc/self/cgroup");
        std::ostringstream self_text;
        self_text << self.rdbuf();
        const std::optional<std::uint64_t> cgroup =
            cgroup_limit_bytes("/sys/fs/cgroup", self_text.str());
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
                                       const HostCacheEnvironment& environment) {
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
    const std::uint64_t reserve = requested.host_cache_reserve_bytes + environment.extra_reserve_bytes;
    std::uint64_t budget = available_host_bytes > reserve ? available_host_bytes - reserve : 0;
    // The reserve leaves memory for what still grows; a cap leaves memory for the machine's other
    // users, which `available_host_bytes` cannot know will want it.
    if (requested.host_cache_max_bytes) {
        budget = std::min<std::uint64_t>(budget, *requested.host_cache_max_bytes);
    }
    if (requested.host_cache_percent) {
        if (*requested.host_cache_percent == 0 || *requested.host_cache_percent > 100) {
            throw std::invalid_argument("host cache percent must be in [1,100]");
        }
        if (!environment.total_host_bytes) {
            throw std::invalid_argument("host cache percent needs the machine's total memory");
        }
        // total * percent / 100 without overflow, split so the remainder is not lost to rounding.
        const std::uint64_t total   = *environment.total_host_bytes;
        const std::uint64_t percent = *requested.host_cache_percent;
        budget = std::min<std::uint64_t>(budget, total / 100 * percent + total % 100 * percent / 100);
    }
    // Where pinned host memory is charged against the GPU (Windows), the same clamp the Program
    // applies to its KV buffer bounds the whole budget, before it is split. The Program clamps the
    // KV buffer against the memory left after the state slots are pinned; with the state slots
    // inside this budget that limit is always the larger, so the KV share is never cut again.
    if (environment.device_free_after_startup) {
        budget = clamp_host_kv_reservation_bytes(budget, *environment.device_free_after_startup, 1);
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
