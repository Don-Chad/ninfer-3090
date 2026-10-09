#pragma once

#include "runtime/contract/resources.h"
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace ninfer {
struct HostKVPageLayout;
}

namespace ninfer::models::qwen3_5 {
struct StateImageHostLayout;
}

namespace ninfer::models::qwen3_5::detail {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point started) noexcept;

std::int32_t checked_i32(std::uint32_t value, const char* label);

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept;

std::size_t context_resource_index(runtime::ContextResourceClass resource);

runtime::ContextTransferRequirement
state_transfer_requirement(const StateImageHostLayout& layout,
                           runtime::ContextTransferDirection direction,
                           bool dflash_local_only = false);

runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextResourceClass resource,
                        runtime::ContextTransferDirection direction, const HostKVPageLayout& layout,
                        std::uint32_t pages, std::uint32_t contiguous_runs = 1);

} // namespace ninfer::models::qwen3_5::detail
