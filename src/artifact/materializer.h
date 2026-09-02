#pragma once

#include "artifact/binder.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

struct LoadProgress {
    std::function<void(std::string_view, std::uint64_t, std::uint64_t)> callback;
};

struct MaterializationStats {
    std::uint64_t file_bytes              = 0;
    std::uint64_t h2d_bytes               = 0;
    std::uint64_t device_capacity_bytes   = 0;
    std::uint64_t retained_resource_bytes = 0;
    std::uint64_t pinned_weight_bytes     = 0;
    std::uint64_t peak_staging_bytes      = 0;
    std::size_t tensor_count              = 0;
    std::size_t resource_count            = 0;
    double upload_seconds                 = 0.0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    void* device_data(ObjectHandle handle) const;
    // Device pointer for device objects, pinned-host pointer for HostPinned objects.
    void* storage_data(ObjectHandle handle) const;
    [[nodiscard]] bool is_host_pinned(ObjectHandle handle) const noexcept;
    std::size_t pinned_offset(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> pinned_block() const noexcept;
    std::span<const std::byte> resource_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_resource_bytes(ObjectHandle handle);

    const MaterializationStats& stats() const noexcept { return stats_; }

    DeviceArena& device_arena();
    // Present only when the arena is backed by an eviction pool (overlay vision residency).
    [[nodiscard]] EvictableWeightPool* eviction_pool() const noexcept { return pool_.get(); }

private:
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                            DeviceContext&, LoadProgress*,
                                            std::unique_ptr<EvictableWeightPool>);

    struct ObjectStorage {
        void* device              = nullptr;
        void* pinned              = nullptr;
        std::size_t pinned_offset = 0;
        std::vector<std::byte> resource;
    };

    std::unique_ptr<EvictableWeightPool> pool_;
    std::unique_ptr<DeviceArena> device_arena_;
    std::unique_ptr<PinnedHostBuffer> pinned_block_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

// backing_pool, when provided, supplies the device arena storage and is owned by the returned
// artifact. Its window mirror is not captured here; the caller captures it once the upload stream
// is synchronized.
MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress = nullptr,
                                 std::unique_ptr<EvictableWeightPool> backing_pool = nullptr);

} // namespace ninfer::artifact
