#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
    // Permanent pinned-host materialization: the tensor never receives device memory at load;
    // its payload lands in one contiguous pinned block laid out with the same alignment rules as
    // the device arena, so the block can be uploaded wholesale and addressed by offsets.
    HostPinned,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct PinnedMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct MaterializationPlan {
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    // Evict-ranked tensors form the arena suffix [evictable_tail_offset, device_capacity_bytes);
    // the offset is aligned to the alignment passed to Binder::finish so an eviction pool can
    // borrow whole chunks without touching resident objects. Both are zero without ranked tensors.
    std::uint64_t evictable_tail_offset = 0;
    std::uint64_t evictable_tail_bytes  = 0;
    std::uint64_t pinned_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
    std::vector<PinnedMaterialization> pinned_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] bool has_object(std::string_view name) const noexcept;

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    // evict_rank 0 keeps the tensor resident for the process lifetime. A nonzero rank moves it
    // into the arena's evictable tail; higher ranks land closer to the arena end and are evicted
    // first.
    void materialize_on_device(ObjectHandle handle, std::uint32_t evict_rank = 0);
    void materialize_on_host_pinned(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    // evictable_alignment aligns the evictable-tail begin (the eviction pool chunk size, or 1).
    MaterializationPlan finish(std::uint64_t evictable_alignment = 1);

private:
    struct PendingEvictable {
        ObjectHandle handle;
        std::uint64_t bytes     = 0;
        std::uint64_t alignment = 0;
        std::uint32_t rank      = 0;
    };

    ObjectHandle find_unconsumed(std::string_view name);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    std::vector<PendingEvictable> evictable_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
