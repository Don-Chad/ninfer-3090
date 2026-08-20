#pragma once

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

namespace ninfer {

// VMM-backed weight arena whose tail region can be temporarily evicted: the physical
// chunks behind the arena suffix are unmapped from their stable home addresses and
// remapped into a reserved overlay range, giving the caller device memory without a
// single new allocation. restore() remaps the chunks home and re-uploads the evicted
// extent from a pinned host mirror captured after the weights landed.
//
// The caller owns quiescing: evict()/restore() require that no GPU work touching the
// arena is in flight. The transaction is deterministic — every byte it hands out is
// backed by pages this pool already owns, so restore cannot fail with an allocation
// error.
class EvictableWeightPool {
public:
    struct Config {
        std::size_t arena_bytes          = 0;   // weights arena capacity
        std::size_t evictable_tail_bytes = 0;   // chunk-aligned suffix that may be evicted
        std::size_t overlay_bytes        = 0;   // staging the window needs mapped at once
        int device                       = 0;
    };

    static constexpr std::size_t kChunkBytes = 16ULL * 1024ULL * 1024ULL;

    [[nodiscard]] static bool supported(int device);

    explicit EvictableWeightPool(const Config& config);
    ~EvictableWeightPool();

    EvictableWeightPool(const EvictableWeightPool&)            = delete;
    EvictableWeightPool& operator=(const EvictableWeightPool&) = delete;
    EvictableWeightPool(EvictableWeightPool&&) noexcept;
    EvictableWeightPool& operator=(EvictableWeightPool&&) noexcept;

    // Stable home mapping backing the weights arena for the process lifetime.
    [[nodiscard]] DeviceSpan arena() const noexcept;
    [[nodiscard]] std::size_t chunk_bytes() const noexcept { return kChunkBytes; }
    [[nodiscard]] std::size_t evictable_tail_bytes() const noexcept;
    [[nodiscard]] std::size_t overlay_capacity() const noexcept;

    // Capture the pinned mirror of the evictable tail after the weights upload
    // completed. Must be called exactly once before the first evict().
    void capture_tail_mirror(cudaStream_t stream);

    // Unmap ceil(bytes / chunk) chunks from the arena end and map the same physical
    // pages contiguously (home order) at overlay_base(). Returns the mapped extent.
    // Requires a captured mirror and no in-flight GPU work on the arena.
    std::byte* evict(std::size_t bytes, std::size_t* mapped_bytes);

    // Remap every evicted chunk home and re-upload the evicted extent from the
    // mirror; synchronizes the stream. Safe to call when nothing is evicted.
    void restore(cudaStream_t stream);

    [[nodiscard]] bool evicted() const noexcept;

    // Wall-clock costs of the most recent transaction, for telemetry.
    [[nodiscard]] double last_evict_seconds() const noexcept;
    [[nodiscard]] double last_restore_seconds() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
