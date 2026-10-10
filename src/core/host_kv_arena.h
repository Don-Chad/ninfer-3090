#pragma once

#include "core/host_context_arena.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {

struct HostKVPlaneLayout {
    std::size_t offset             = 0;
    std::size_t page_payload_bytes = 0;
    std::size_t head_payload_bytes = 0;

    friend bool operator==(const HostKVPlaneLayout&, const HostKVPlaneLayout&) = default;
};

struct HostKVPageLayout {
    KVPageGeometry geometry;
    std::vector<HostKVPlaneLayout> planes;
    std::size_t page_stride = 0;

    friend bool operator==(const HostKVPageLayout&, const HostKVPageLayout&) = default;
};

[[nodiscard]] HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry);

class HostKVArena;

class HostKVAllocationHandle {
public:
    HostKVAllocationHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(HostKVAllocationHandle,
                                         HostKVAllocationHandle) noexcept = default;

private:
    friend class HostKVArena;
    friend class HostKVAllocation;
    friend class HostKVAllocationView;
    friend class HostKVAllocationConstView;

    HostKVAllocationHandle(const HostKVArena* owner, std::uint32_t descriptor,
                           std::uint32_t generation) noexcept
        : owner_(owner), descriptor_(descriptor), generation_(generation) {}

    const HostKVArena* owner_ = nullptr;
    std::uint32_t descriptor_ = 0;
    std::uint32_t generation_ = 0;
};

class HostKVAllocationView {
public:
    HostKVAllocationView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }

    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }

    [[nodiscard]] const HostKVPageLayout& layout() const;
    [[nodiscard]] HostKVAllocationView subview(std::uint32_t begin, std::uint32_t count) const;

private:
    friend class HostKVArena;
    friend class HostKVAllocationConstView;

    HostKVAllocationView(HostKVAllocationHandle handle, std::byte* data,
                         const HostKVPageLayout* layout, std::uint32_t page_count) noexcept
        : handle_(handle), data_(data), layout_(layout), page_count_(page_count) {}

    HostKVAllocationHandle handle_;
    std::byte* data_                = nullptr;
    const HostKVPageLayout* layout_ = nullptr;
    std::uint32_t page_count_       = 0;
};

class HostKVAllocationConstView {
public:
    HostKVAllocationConstView() noexcept = default;

    HostKVAllocationConstView(HostKVAllocationView view) noexcept
        : handle_(view.handle_), data_(view.data_), layout_(view.layout_),
          page_count_(view.page_count_) {}

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }

    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }

    [[nodiscard]] const HostKVPageLayout& layout() const;
    [[nodiscard]] HostKVAllocationConstView subview(std::uint32_t begin, std::uint32_t count) const;

private:
    friend class HostKVArena;

    HostKVAllocationConstView(HostKVAllocationHandle handle, const std::byte* data,
                              const HostKVPageLayout* layout, std::uint32_t page_count) noexcept
        : handle_(handle), data_(data), layout_(layout), page_count_(page_count) {}

    HostKVAllocationHandle handle_;
    const std::byte* data_          = nullptr;
    const HostKVPageLayout* layout_ = nullptr;
    std::uint32_t page_count_       = 0;
};

class HostKVAllocation {
public:
    HostKVAllocation() noexcept = default;
    ~HostKVAllocation();

    HostKVAllocation(const HostKVAllocation&)            = delete;
    HostKVAllocation& operator=(const HostKVAllocation&) = delete;
    HostKVAllocation(HostKVAllocation&& other) noexcept;
    HostKVAllocation& operator=(HostKVAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] HostKVAllocationHandle handle() const noexcept;
    [[nodiscard]] std::uint32_t page_count() const noexcept;
    void publish() noexcept;
    bool release() noexcept;

private:
    friend class HostKVArena;

    HostKVAllocation(HostKVArena& owner, std::uint32_t descriptor,
                     std::uint32_t generation) noexcept
        : owner_(&owner), descriptor_(descriptor), generation_(generation) {}

    void disarm() noexcept;

    HostKVArena* owner_       = nullptr;
    std::uint32_t descriptor_ = 0;
    std::uint32_t generation_ = 0;
};

class HostKVArena {
public:
    HostKVArena(HostContextArena& arena, std::span<const HostKVPageLayout> supported_layouts);

    HostKVArena(const HostKVArena&)            = delete;
    HostKVArena& operator=(const HostKVArena&) = delete;
    HostKVArena(HostKVArena&&)                 = delete;
    HostKVArena& operator=(HostKVArena&&)      = delete;

    // Shared backing capacity; never add this to another typed pool's capacity.
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return arena_->capacity_bytes(); }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    [[nodiscard]] std::size_t free_bytes() const noexcept { return arena_->free_bytes(); }

    [[nodiscard]] const HostKVPageLayout* layout_for(const KVPageGeometry& geometry) const noexcept;

    [[nodiscard]] bool can_allocate(const HostKVPageLayout& layout,
                                    std::uint32_t pages) const noexcept;
    [[nodiscard]] std::optional<HostKVAllocation> allocate(const HostKVPageLayout& layout,
                                                           std::uint32_t pages) noexcept;

    [[nodiscard]] std::pair<HostKVAllocation, HostKVAllocation> split(HostKVAllocation&& allocation,
                                                                      std::uint32_t page_offset);

    [[nodiscard]] HostKVAllocationView writable_view(HostKVAllocation& allocation);
    [[nodiscard]] HostKVAllocationConstView view(const HostKVAllocation& allocation) const;

private:
    friend class HostKVAllocation;
    friend class HostKVAllocationView;
    friend class HostKVAllocationConstView;

    struct Descriptor {
        HostContextAllocation storage;
        std::uint32_t layout     = 0;
        std::uint32_t pages      = 0;
        std::uint32_t generation = 1;
        bool active              = false;
    };

    [[nodiscard]] std::optional<std::uint32_t>
    find_layout(const HostKVPageLayout& layout) const noexcept;
    [[nodiscard]] bool valid_handle(HostKVAllocationHandle handle) const noexcept;
    [[nodiscard]] std::uint32_t take_descriptor() noexcept;
    bool release_descriptor(std::uint32_t descriptor, std::uint32_t generation) noexcept;
    [[nodiscard]] std::byte* allocation_data(const Descriptor& descriptor) const noexcept;

    HostContextArena* arena_    = nullptr;
    std::size_t occupied_bytes_ = 0;
    std::vector<HostKVPageLayout> layouts_;
    std::vector<Descriptor> descriptors_;
    std::vector<std::uint32_t> free_descriptors_;
};

} // namespace ninfer
