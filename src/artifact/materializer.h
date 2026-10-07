#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
    // Set when the stored row-split Q8 object is requantized at load: `bytes` is then the target
    // encoding's size and the device parent carries the target geometry (see artifact/transcode.h).
    std::optional<QType> transcode;
    // Which pipeline rank's device arena holds the object; `offset` is relative to that arena.
    // Rank 0 is the primary device, and is the only rank a single-device load ever uses.
    std::size_t rank = 0;
};

// Offset inside the one page-locked Host block (Residency::Pinned).
struct PinnedPlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

struct MaterializationPlan {
    const Reader* source           = nullptr;
    std::size_t object_count       = 0;
    std::uint64_t prior_read_bytes = 0;
    // One device arena per pipeline rank. Entry 0 is the primary device and always exists, so a
    // single-device plan has exactly one entry and rank 0 owns every device object.
    std::vector<std::uint64_t> device_capacity_by_rank{0};
    // Evict-ranked objects occupy [evictable_tail_offset, device_capacity(0)) on rank 0; the offset
    // is aligned as Binder::finish was asked. Both are zero without ranked objects. The evictable
    // tail is a rank-0 feature: an offloaded rank holds nothing a Vision window may borrow.
    std::uint64_t evictable_tail_offset = 0;
    std::uint64_t evictable_tail_bytes  = 0;
    std::uint64_t pinned_capacity_bytes = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<PinnedPlacement> pinned_objects;
    std::vector<HostPlacement> host_objects;

    [[nodiscard]] std::size_t device_rank_count() const noexcept {
        return device_capacity_by_rank.empty() ? 1 : device_capacity_by_rank.size();
    }

    [[nodiscard]] std::uint64_t device_capacity(std::size_t rank = 0) const noexcept {
        return rank < device_capacity_by_rank.size() ? device_capacity_by_rank[rank] : 0;
    }
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0; // primary device (rank 0)
    // Expert-offload split only: the weight arenas held by the ranks past the primary device.
    std::uint64_t offloaded_device_capacity_bytes = 0;
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t pinned_bytes          = 0; // page-locked Host block (Residency::Pinned)
    std::uint64_t peak_staging_bytes    = 0;
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0;
    double upload_seconds               = 0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;
    [[nodiscard]] const WeightParent& pinned_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> pinned_block() const noexcept;
    // Present when the device backing was supplied by an eviction pool.
    [[nodiscard]] EvictableWeightPool* weight_pool() const noexcept { return pool_.get(); }

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&, DeviceContext&,
                                            const StartupObserver*,
                                            std::unique_ptr<EvictableWeightPool>);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> pinned;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
    };

    // The pool owns the physical memory behind a pool-backed arena; destroy the arena first.
    std::unique_ptr<EvictableWeightPool> pool_;
    // One arena per pipeline rank, allocated on that rank's device. Index 0 is the primary device;
    // a single-device load holds exactly one entry, which is the only one an offload-free model
    // ever touches.
    std::vector<std::unique_ptr<DeviceArena>> arenas_;
    std::unique_ptr<PinnedHostBuffer> pinned_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

// `backing`, when given, supplies the device arena (its arena must cover the plan) and is owned by
// the result. Its window mirror is captured later by the owner, once the window is known.
[[nodiscard]] MaterializedArtifact
materialize(const Reader& reader, MaterializationPlan&& plan, DeviceContext& device,
            const StartupObserver* startup_observer      = nullptr,
            std::unique_ptr<EvictableWeightPool> backing = nullptr);

} // namespace ninfer::artifact
