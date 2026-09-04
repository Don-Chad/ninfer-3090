#pragma once

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

namespace ninfer {

// VMM-backed weight arena whose evict-ranked tail can be borrowed for a bounded window. The
// physical chunks behind the arena suffix are unmapped from their stable home addresses and
// remapped into a reserved overlay range, so a transaction hands out device memory without a single
// allocation and captured graphs stay valid because the home mapping never moves. Closing the
// transaction remaps the chunks home and re-uploads only the dirtied extent from a pinned mirror.
//
// evict() quiesces both streams of the owning DeviceContext before remapping. The caller
// guarantees that no other stream touches the arena while a transaction is open.
class EvictableWeightPool {
public:
    static constexpr std::size_t kChunkBytes = 16ULL * 1024ULL * 1024ULL;

    struct Config {
        std::size_t arena_bytes           = 0; // weights arena capacity
        std::size_t evictable_tail_bytes  = 0; // arena suffix holding evict-ranked tensors
        std::size_t window_capacity_bytes = 0; // largest extent one transaction may borrow
    };

    struct TransactionStats {
        double evict_seconds     = 0.0;
        double restore_seconds   = 0.0;
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
        [[nodiscard]] const TransactionStats& stats() const noexcept { return stats_; }

        // Remaps the borrowed chunks home and re-uploads the dirtied extent. Never throws: a
        // failure poisons the pool, which the owner must treat as fatal for the weights.
        void close() noexcept;

    private:
        friend class EvictableWeightPool;

        Transaction(EvictableWeightPool& pool, DeviceSpan leased, cudaStream_t stream,
                    double evict_seconds) noexcept;

        EvictableWeightPool* pool_ = nullptr;
        DeviceSpan leased_{};
        cudaStream_t stream_ = nullptr;
        TransactionStats stats_{};
    };

    [[nodiscard]] static bool supported(const DeviceContext& device);

    EvictableWeightPool(DeviceContext& device, const Config& config);
    ~EvictableWeightPool();

    EvictableWeightPool(const EvictableWeightPool&)            = delete;
    EvictableWeightPool& operator=(const EvictableWeightPool&) = delete;

    // Stable home mapping backing the weights arena for the process lifetime.
    [[nodiscard]] DeviceSpan arena() const noexcept;
    [[nodiscard]] std::size_t evictable_tail_bytes() const noexcept;
    [[nodiscard]] std::size_t window_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t mirror_bytes() const noexcept;
    [[nodiscard]] bool mirror_captured() const noexcept;
    [[nodiscard]] bool transaction_open() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;

    // Pins a copy of every tail byte a window can dirty. Call exactly once after the weights
    // landed and the upload stream drained.
    void capture_window_mirror(cudaStream_t stream);

    // Borrows ceil(bytes / kChunkBytes) chunks from the arena end, mapped contiguously at the
    // overlay range. Rejects requests beyond the window capacity and nested transactions.
    [[nodiscard]] Transaction evict(std::size_t bytes, cudaStream_t stream);

private:
    struct Impl;

    void restore(Transaction& transaction) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
