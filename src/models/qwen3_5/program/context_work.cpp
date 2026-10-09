#include "models/qwen3_5/program/context_work.h"
#include "core/host_kv_arena.h"
#include "models/qwen3_5/state/state_image.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

std::uint64_t elapsed_ns(Clock::time_point started) noexcept {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept {
    return frontier == 0 ? 0U : 1U + (frontier - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::size_t context_resource_index(runtime::ContextResourceClass resource) {
    switch (resource) {
    case runtime::ContextResourceClass::State:
        return 0;
    case runtime::ContextResourceClass::MainKV:
        return 1;
    case runtime::ContextResourceClass::BackendKV:
        return 2;
    }
    throw std::logic_error("unknown context resource class");
}

runtime::ContextTransferRequirement
state_transfer_requirement(const StateImageHostLayout& layout,
                           runtime::ContextTransferDirection direction, bool dflash_local_only) {
    return runtime::ContextTransferRequirement{
        .resource   = runtime::ContextResourceClass::State,
        .direction  = direction,
        .units      = 1,
        .page_count = 0,
        .work       = dflash_local_only ? dflash_local_transfer_work(layout)
                                        : state_image_transfer_work(layout),
    };
}

runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextResourceClass resource,
                        runtime::ContextTransferDirection direction, const HostKVPageLayout& layout,
                        std::uint32_t pages, std::uint32_t contiguous_runs) {
    const TransferWork work = direction == runtime::ContextTransferDirection::DeviceToDevice
                                  ? plan_device_kv_copy_work(layout, pages)
                                  : plan_host_kv_transfer_work(layout, pages, contiguous_runs);
    return runtime::ContextTransferRequirement{
        .resource   = resource,
        .direction  = direction,
        .units      = work.payload_bytes,
        .page_count = pages,
        .work       = work,
    };
}

} // namespace ninfer::models::qwen3_5::detail
