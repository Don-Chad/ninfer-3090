#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace ninfer::runtime {

// Memory this process may still take from the host: the smaller of the system's available memory and
// what remains under the container's cgroup limit. Empty when the platform gives no answer.
[[nodiscard]] std::optional<std::uint64_t> available_host_memory_bytes() noexcept;

// What the memory cgroup of this process still allows, from the contents of `/proc/self/cgroup`
// and the cgroup mount `cgroup_root` (normally /sys/fs/cgroup). Each level from the process's own
// cgroup up to the root is read and the tightest remainder wins, because a limit may sit on any
// ancestor (a container runtime's or a systemd scope's cgroup), not at the mount root. Handles
// cgroup v2 and v1 (and hybrid); reclaimable page cache is not counted as use. Empty when no level
// sets a limit. Exposed so it can be exercised against a fabricated cgroup tree.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_remaining_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

// Sizes the pinned host tier of the context cache from `available_host_bytes`, the host memory still
// free once the model and every other startup allocation are in place.
//
// The machine is assumed to serve only this process, so all but `requested.host_cache_reserve_bytes`
// is spent; the reserve covers what still grows afterwards (request buffers, the response store,
// graph instantiation). Of what is spent, an eighth goes to StateImage
// slots (each a whole Gated DeltaNet snapshot, `state_image_bytes`) and the rest to KV pages. The
// private and shared catalogs are sized to the number of states that can be resident, since a
// retained continuation is only worth keeping while it has a state to resume from.
// `pinned_shared_prefixes` (injected grafts) are held outside that sizing and added to the shared
// catalog on top.
//
// `device_free_after_startup` is set where pinned host memory is charged against the GPU (Windows):
// the free device memory left once the Program's planned reservation is taken. The whole budget is
// clamped against it as the Program clamps its KV buffer, before the split, so the sizes reported
// here are the ones that get pinned. Empty elsewhere.
//
// `requested` must carry `auto_host_cache`; the returned options have it cleared and every
// other field (including the device-side ones) carried over.
[[nodiscard]] ContextCacheOptions
resolve_host_cache(const ContextCacheOptions& requested, std::uint64_t available_host_bytes,
                   std::uint64_t state_image_bytes, std::uint32_t max_concurrency,
                   std::uint32_t pinned_shared_prefixes,
                   std::optional<std::uint64_t> device_free_after_startup = std::nullopt);

} // namespace ninfer::runtime
