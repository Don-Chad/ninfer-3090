#pragma once

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {

// VMM-backed persistent arena whose KV payload prefix can lend individual granules for a bounded
// window. Every store binds at the offsets it uses today, so plane pointers, block tables and the
// captured decode graphs stay valid: the home mapping never moves and only granules that lie
// entirely inside free KV pages are ever unmapped.
//
// Unlike the weight pool there is no host mirror. A lent granule holds free pages only, and no
// page is read before the append kernel writes it, so closing a lease remaps home and copies
// nothing.
//
// Neither lease() nor close() quiesces a stream: the caller guarantees that every lent granule
// lies entirely inside free KV pages, so no in-flight work addresses it and decode continues on
// the main stream while a window is open.
class EvictableKVPool {
public:
    struct Config {
        std::size_t arena_bytes           = 0; // persistent arena capacity
        std::size_t lendable_prefix_bytes = 0; // KV payload prefix eligible for lending
        std::size_t window_capacity_bytes = 0; // largest extent one lease may hand out
    };

    struct LeaseStats {
        double lease_seconds     = 0.0;
        double return_seconds    = 0.0;
        std::size_t mapped_bytes = 0;
    };

    class Transaction {
    public:
        Transaction() noexcept = default;
        ~Transaction();

        Transaction(const Transaction&)            = delete;
        Transaction& operator=(const Transaction&) = delete;
        Transaction(Transaction&& other) noexcept;
        Transaction& operator=(Transaction&& other) noexcept;

        [[nodiscard]] bool open() const noexcept { return pool_ != nullptr; }
        [[nodiscard]] DeviceSpan leased() const noexcept { return leased_; }
        [[nodiscard]] const LeaseStats& stats() const noexcept { return stats_; }

        // Remaps the lent granules home. Never throws: a failure poisons the pool, which the
        // owner must treat as fatal for the KV cache.
        void close() noexcept;

    private:
        friend class EvictableKVPool;

        Transaction(EvictableKVPool& pool, DeviceSpan leased, cudaStream_t stream,
                    double lease_seconds) noexcept;

        EvictableKVPool* pool_ = nullptr;
        DeviceSpan leased_{};
        cudaStream_t stream_ = nullptr;
        LeaseStats stats_{};
    };

    [[nodiscard]] static bool supported(const DeviceContext& device);
    // VMM allocation granularity of the device, or zero when virtual memory management is
    // unavailable. Lets a caller decide whether a window fits the lendable prefix before it
    // commits to building a pool.
    [[nodiscard]] static std::size_t device_granularity(const DeviceContext& device);

    EvictableKVPool(DeviceContext& device, const Config& config);
    ~EvictableKVPool();

    EvictableKVPool(const EvictableKVPool&)            = delete;
    EvictableKVPool& operator=(const EvictableKVPool&) = delete;

    // Stable home mapping backing the persistent arena for the process lifetime.
    [[nodiscard]] DeviceSpan arena() const noexcept;
    [[nodiscard]] std::size_t granularity() const noexcept;
    // Granules covering the lendable prefix; indices beyond this are permanently resident.
    [[nodiscard]] std::size_t lendable_granules() const noexcept;
    [[nodiscard]] std::size_t window_capacity_bytes() const noexcept;
    [[nodiscard]] bool lease_open() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;

    // Granule containing an arena offset, and the arena range one granule covers.
    [[nodiscard]] std::size_t granule_of(std::size_t offset) const;
    [[nodiscard]] DeviceSpan granule_bytes(std::size_t index) const;

    // Lends the given granules, mapped consecutively at the overlay range in the order supplied.
    // Indices must be strictly increasing and inside the lendable prefix.
    [[nodiscard]] Transaction lease(std::span<const std::size_t> granules, cudaStream_t stream);

private:
    struct Impl;

    void give_back(Transaction& transaction) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
