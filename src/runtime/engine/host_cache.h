#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace ninfer::runtime {

// Memory this process may still take from the host: the smaller of the system's available memory and
// what remains under the container's cgroup limit. Empty when the platform gives no answer.
[[nodiscard]] std::optional<std::uint64_t> available_host_memory_bytes() noexcept;

// The machine's total memory as this process may use it: the smaller of physical memory and the
// tightest cgroup limit above the process. Empty when the platform gives no answer.
[[nodiscard]] std::optional<std::uint64_t> total_host_memory_bytes() noexcept;

// The tightest memory limit (not the remainder) that any cgroup level above the process sets, read
// as `cgroup_remaining_bytes` reads them. Empty when no level sets a limit.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_limit_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

// What the machine says about itself when the host tier is sized, beyond the memory still free.
struct HostCacheEnvironment {
    // Total memory as `total_host_memory_bytes` reports it; required for `host_cache_percent`.
    std::optional<std::uint64_t> total_host_bytes;
    // Host memory that grows after sizing beyond `host_cache_reserve_bytes` and that the caller can
    // bound (the media caches), added to the reserve.
    std::uint64_t extra_reserve_bytes = 0;
};

// What the memory cgroup of this process still allows, from the contents of `/proc/self/cgroup`
// and the cgroup mount `cgroup_root` (normally /sys/fs/cgroup). Each level from the process's own
// cgroup up to the root is read and the tightest remainder wins, because a limit may sit on any
// ancestor (a container runtime's or a systemd scope's cgroup), not at the mount root. Handles
// cgroup v2 and v1 (and hybrid); reclaimable page cache is not counted as use. Empty when no level
// sets a limit. Exposed so it can be exercised against a fabricated cgroup tree.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_remaining_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

// Sizes the pinned Host context capacity (ContextCacheOptions::host_capacity_bytes, the one quota
// shared by Host StateImages, Host KV, pause snapshots and in-flight destinations) from
// `available_host_bytes`, the host memory still free once the model is loaded.
//
// The machine is assumed to serve only this process, so all but the reserve is spent; the reserve
// covers what still grows afterwards (request buffers, the response store, graph instantiation).
// The capacity is the smallest of: available memory less the reserve
// (`host_cache_reserve_bytes` plus `environment.extra_reserve_bytes`), `host_cache_max_bytes`, and
// `host_cache_percent` of `environment.total_host_bytes`.
//
// Where pinned host memory is charged against the GPU (Windows/WDDM) the Program clamps the
// capacity again against the device memory left when it pins it; that clamp is not applied here.
//
// `requested` must carry `auto_host_cache` and no explicit `host_capacity_bytes`.
[[nodiscard]] std::size_t resolve_host_capacity_bytes(const ContextCacheOptions& requested,
                                                      std::uint64_t available_host_bytes,
                                                      const HostCacheEnvironment& environment = {});

} // namespace ninfer::runtime
