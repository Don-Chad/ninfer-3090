#pragma once

#include "runtime/contract/request.h"
#include "core/transfer_work.h"
#include "core/wide_math.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ninfer::runtime {

enum class CheckpointRole : std::uint8_t { InputReplay, LongAnchor, SharedPrefix, Continuation };

// Physical quantities are sampled from stores, never a scheduler-owned occupancy ledger.
struct ContextResourceUsage {
    std::uint32_t state_slots                                                   = 0;
    std::uint32_t main_kv_pages                                                 = 0;
    std::uint32_t backend_kv_pages                                              = 0;
    std::size_t host_bytes                                                      = 0;
    friend bool operator==(ContextResourceUsage, ContextResourceUsage) noexcept = default;
};

struct ResourceReservation {
    bool reserved = false;
    ContextResourceUsage shortage;

    explicit operator bool() const noexcept { return reserved; }
};

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
struct PrefillWork {
    std::uint64_t chunks          = 0;
    std::uint64_t tokens          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    const core::Uint128 linear = core::wide_multiply(prefix_tokens, suffix_tokens);
    // suffix*(suffix+1)/2 without a 128-bit divide. Exactly one of the two factors is even, so
    // halving that one first keeps both operands inside uint64 even at suffix_tokens ==
    // UINT64_MAX, where suffix_tokens + 1 would otherwise wrap.
    const core::Uint128 triangular =
        (suffix_tokens & 1U) == 0U
            ? core::wide_multiply(suffix_tokens >> 1U, suffix_tokens + 1U)
            : core::wide_multiply(suffix_tokens, (suffix_tokens >> 1U) + 1U);
    bool overflowed               = false;
    const core::Uint128 attention = core::wide_add(linear, triangular, overflowed);
    result.attention_pairs        = overflowed ? std::numeric_limits<std::uint64_t>::max()
                                               : core::wide_clamp_to_uint64(attention);
    return result;
}

enum class ContextResourceClass : std::uint8_t {
    State,
    MainKV,
    BackendKV,
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ContextTransferObservation {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count           = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

struct ContextTransferRequirement {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0;
    std::uint32_t page_count           = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t pressure_spill_pages   = 0;
    std::uint64_t partial_tail_cow_pages = 0;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis.
//
// The primary device's curve is the flat fields below. A model split into pipeline stages has the
// same curve for every further device in `extra_ranks`: each holds its own layers' KV, so a page
// group costs each device a different number of bytes, and the capacity that fits is the smallest
// any one of them allows. A device with no KV cost (stride zero) does not constrain it. Empty on a
// single device (the multi-GPU pipeline is parked on this build).
struct RankCapacityCurve {
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
};

struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    // Pages held for the Engine's life by installed context (direct grafts). Included in the
    // minimum and maximum; an explicit token capacity is granted on top of them.
    std::uint32_t pinned_main_page_groups            = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::vector<RankCapacityCurve> extra_ranks;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    // Bytes further device `extra_rank` (0-based, so device 1 is index 0) must hold.
    [[nodiscard]] std::size_t extra_rank_reservation_bytes(std::size_t extra_rank,
                                                           std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    // What each further device reserves (device 1 first), and which device bound the capacity.
    std::vector<std::size_t> extra_rank_reservation_bytes;
    std::size_t binding_rank                         = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
