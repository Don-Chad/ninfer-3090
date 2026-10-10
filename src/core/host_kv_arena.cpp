#include "core/host_kv_arena.h"

#include "core/dtype.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

constexpr std::size_t kHostKVAlignment = 256;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
    }
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(std::string(label) + " alignment overflow");
    }
    return (value + mask) & ~mask;
}

void increment_generation(std::uint32_t& generation) noexcept {
    ++generation;
    if (generation == 0) { ++generation; }
}

} // namespace

HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry) {
    if (geometry.page_tokens == 0 || geometry.planes.empty()) {
        throw std::invalid_argument("Host KV page geometry is empty");
    }

    HostKVPageLayout out;
    out.geometry = geometry;
    out.planes.reserve(geometry.planes.size());
    std::size_t cursor = 0;
    for (const KVPlaneGeometry& plane : geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Host KV plane geometry must be positive");
        }
        // Device slab alignment is not part of the canonical packed Host representation.
        cursor = align_up(cursor, kHostKVAlignment, "Host KV plane");
        const std::size_t head_bytes =
            checked_mul(checked_mul(static_cast<std::size_t>(plane.leading_extent),
                                    geometry.page_tokens, "Host KV head payload overflow"),
                        dtype_size(plane.dtype), "Host KV head payload overflow");
        const std::size_t page_bytes =
            checked_mul(head_bytes, static_cast<std::size_t>(plane.head_extent),
                        "Host KV plane payload overflow");
        out.planes.push_back(HostKVPlaneLayout{
            .offset             = cursor,
            .page_payload_bytes = page_bytes,
            .head_payload_bytes = head_bytes,
        });
        cursor = checked_add(cursor, page_bytes, "Host KV page payload overflow");
    }
    out.page_stride = align_up(cursor, kHostKVAlignment, "Host KV page record");
    return out;
}

bool HostKVAllocationView::valid() const noexcept {
    return handle_.owner_ != nullptr && handle_.owner_->valid_handle(handle_);
}

const HostKVPageLayout& HostKVAllocationView::layout() const {
    if (!valid() || layout_ == nullptr) { throw std::logic_error("Host KV view is stale"); }
    return *layout_;
}

HostKVAllocationView HostKVAllocationView::subview(std::uint32_t begin, std::uint32_t count) const {
    if (!valid() || count == 0 || begin > page_count_ || count > page_count_ - begin) {
        throw std::out_of_range("Host KV subview is outside its allocation");
    }
    return HostKVAllocationView(
        handle_, data_ + static_cast<std::size_t>(begin) * layout_->page_stride, layout_, count);
}

bool HostKVAllocationConstView::valid() const noexcept {
    return handle_.owner_ != nullptr && handle_.owner_->valid_handle(handle_);
}

const HostKVPageLayout& HostKVAllocationConstView::layout() const {
    if (!valid() || layout_ == nullptr) { throw std::logic_error("Host KV view is stale"); }
    return *layout_;
}

HostKVAllocationConstView HostKVAllocationConstView::subview(std::uint32_t begin,
                                                             std::uint32_t count) const {
    if (!valid() || count == 0 || begin > page_count_ || count > page_count_ - begin) {
        throw std::out_of_range("Host KV subview is outside its allocation");
    }
    return HostKVAllocationConstView(
        handle_, data_ + static_cast<std::size_t>(begin) * layout_->page_stride, layout_, count);
}

HostKVAllocation::~HostKVAllocation() { (void)release(); }

HostKVAllocation::HostKVAllocation(HostKVAllocation&& other) noexcept
    : owner_(other.owner_), descriptor_(other.descriptor_), generation_(other.generation_) {
    other.disarm();
}

HostKVAllocation& HostKVAllocation::operator=(HostKVAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_      = other.owner_;
    descriptor_ = other.descriptor_;
    generation_ = other.generation_;
    other.disarm();
    return *this;
}

HostKVAllocationHandle HostKVAllocation::handle() const noexcept {
    return valid() ? HostKVAllocationHandle(owner_, descriptor_, generation_)
                   : HostKVAllocationHandle();
}

std::uint32_t HostKVAllocation::page_count() const noexcept {
    if (!valid() || descriptor_ >= owner_->descriptors_.size()) { return 0; }
    const HostKVArena::Descriptor& descriptor = owner_->descriptors_[descriptor_];
    return descriptor.active && descriptor.generation == generation_ ? descriptor.pages : 0;
}

void HostKVAllocation::publish() noexcept {
    if (owner_ != nullptr && owner_->valid_handle(handle())) {
        owner_->descriptors_[descriptor_].storage.publish();
    }
}

bool HostKVAllocation::release() noexcept {
    if (!valid()) { return false; }
    const bool released = owner_->release_descriptor(descriptor_, generation_);
    disarm();
    return released;
}

void HostKVAllocation::disarm() noexcept {
    owner_      = nullptr;
    descriptor_ = 0;
    generation_ = 0;
}

HostKVArena::HostKVArena(HostContextArena& arena,
                         std::span<const HostKVPageLayout> supported_layouts)
    : arena_(&arena), layouts_(supported_layouts.begin(), supported_layouts.end()) {
    for (std::size_t index = 0; index < layouts_.size(); ++index) {
        const HostKVPageLayout planned = plan_host_kv_page_layout(layouts_[index].geometry);
        if (planned != layouts_[index]) {
            throw std::invalid_argument("Host KV arena received an inconsistent page layout");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (layouts_[previous] == layouts_[index]) {
                throw std::invalid_argument("Host KV arena contains a duplicate page layout");
            }
        }
    }
    if (arena_->capacity_bytes() == 0) { return; }
    if (layouts_.empty()) {
        throw std::invalid_argument("Non-empty Host KV arena requires supported page layouts");
    }

    const auto smallest = std::min_element(
        layouts_.begin(), layouts_.end(), [](const HostKVPageLayout& a, const HostKVPageLayout& b) {
            return a.page_stride < b.page_stride;
        });
    if (smallest->page_stride < arena_->minimum_allocation_bytes()) {
        throw std::invalid_argument("Host KV geometry is below the shared arena minimum");
    }
    const std::size_t maximum_descriptors = arena_->capacity_bytes() / smallest->page_stride;
    if (maximum_descriptors > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host KV arena descriptor capacity exceeds uint32");
    }
    descriptors_.resize(maximum_descriptors);
    free_descriptors_.reserve(maximum_descriptors);
    for (std::size_t index = maximum_descriptors; index > 0; --index) {
        free_descriptors_.push_back(static_cast<std::uint32_t>(index - 1));
    }
}

std::optional<std::uint32_t>
HostKVArena::find_layout(const HostKVPageLayout& layout) const noexcept {
    const auto it = std::find(layouts_.begin(), layouts_.end(), layout);
    if (it == layouts_.end()) { return std::nullopt; }
    return static_cast<std::uint32_t>(it - layouts_.begin());
}

const HostKVPageLayout* HostKVArena::layout_for(const KVPageGeometry& geometry) const noexcept {
    const auto layout =
        std::find_if(layouts_.begin(), layouts_.end(), [&](const HostKVPageLayout& candidate) {
            return candidate.geometry == geometry;
        });
    return layout == layouts_.end() ? nullptr : &*layout;
}

bool HostKVArena::can_allocate(const HostKVPageLayout& layout, std::uint32_t pages) const noexcept {
    if (pages == 0 || free_descriptors_.empty() || !find_layout(layout) ||
        layout.page_stride > std::numeric_limits<std::size_t>::max() / pages) {
        return false;
    }
    return arena_->can_allocate(layout.page_stride * static_cast<std::size_t>(pages));
}

std::optional<HostKVAllocation> HostKVArena::allocate(const HostKVPageLayout& layout,
                                                      std::uint32_t pages) noexcept {
    const std::optional<std::uint32_t> layout_index = find_layout(layout);
    if (!layout_index || pages == 0 || free_descriptors_.empty() ||
        layout.page_stride > std::numeric_limits<std::size_t>::max() / pages) {
        return std::nullopt;
    }
    const std::size_t bytes = layout.page_stride * static_cast<std::size_t>(pages);
    auto storage            = arena_->allocate(bytes);
    if (!storage) { return std::nullopt; }
    const std::uint32_t descriptor_index = take_descriptor();
    Descriptor& descriptor               = descriptors_[descriptor_index];
    descriptor.storage                   = std::move(*storage);
    descriptor.layout                    = *layout_index;
    descriptor.pages                     = pages;
    descriptor.active                    = true;
    occupied_bytes_ += bytes;
    return HostKVAllocation(*this, descriptor_index, descriptor.generation);
}

std::pair<HostKVAllocation, HostKVAllocation> HostKVArena::split(HostKVAllocation&& allocation,
                                                                 std::uint32_t page_offset) {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot split a stale Host KV allocation");
    }
    Descriptor& original = descriptors_[allocation.descriptor_];
    if (page_offset == 0 || page_offset >= original.pages) {
        throw std::out_of_range("Host KV split must leave two non-empty allocations");
    }
    const std::uint32_t right_index = take_descriptor();
    if (right_index == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("Host KV descriptor capacity invariant was violated");
    }

    const std::size_t stride        = layouts_[original.layout].page_stride;
    const std::uint32_t right_pages = original.pages - page_offset;
    Descriptor& right               = descriptors_[right_index];
    auto pieces =
        arena_->split(std::move(original.storage), static_cast<std::size_t>(page_offset) * stride);
    original.storage = std::move(pieces.first);
    right.storage    = std::move(pieces.second);
    right.layout     = original.layout;
    right.pages      = right_pages;
    right.active     = true;

    increment_generation(original.generation);
    original.pages                       = page_offset;
    const std::uint32_t left_generation  = original.generation;
    const std::uint32_t right_generation = right.generation;
    const std::uint32_t left_index       = allocation.descriptor_;
    allocation.disarm();
    return {HostKVAllocation(*this, left_index, left_generation),
            HostKVAllocation(*this, right_index, right_generation)};
}

HostKVAllocationView HostKVArena::writable_view(HostKVAllocation& allocation) {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot view a stale Host KV allocation");
    }
    const Descriptor& descriptor = descriptors_[allocation.descriptor_];
    return HostKVAllocationView(allocation.handle(), allocation_data(descriptor),
                                &layouts_[descriptor.layout], descriptor.pages);
}

HostKVAllocationConstView HostKVArena::view(const HostKVAllocation& allocation) const {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot view a stale Host KV allocation");
    }
    const Descriptor& descriptor = descriptors_[allocation.descriptor_];
    return HostKVAllocationConstView(allocation.handle(), allocation_data(descriptor),
                                     &layouts_[descriptor.layout], descriptor.pages);
}

bool HostKVArena::valid_handle(HostKVAllocationHandle handle) const noexcept {
    if (handle.owner_ != this || handle.descriptor_ >= descriptors_.size()) { return false; }
    const Descriptor& descriptor = descriptors_[handle.descriptor_];
    return descriptor.active && descriptor.generation == handle.generation_;
}

std::uint32_t HostKVArena::take_descriptor() noexcept {
    if (free_descriptors_.empty()) { return std::numeric_limits<std::uint32_t>::max(); }
    const std::uint32_t out = free_descriptors_.back();
    free_descriptors_.pop_back();
    return out;
}

bool HostKVArena::release_descriptor(std::uint32_t descriptor_index,
                                     std::uint32_t generation) noexcept {
    if (descriptor_index >= descriptors_.size()) { return false; }
    Descriptor& descriptor = descriptors_[descriptor_index];
    if (!descriptor.active || descriptor.generation != generation) { return false; }

    occupied_bytes_ -= descriptor.storage.bytes();
    (void)descriptor.storage.release();
    descriptor.active = false;
    descriptor.pages  = 0;
    increment_generation(descriptor.generation);
    free_descriptors_.push_back(descriptor_index);
    return true;
}

std::byte* HostKVArena::allocation_data(const Descriptor& descriptor) const noexcept {
    return descriptor.storage.data();
}

} // namespace ninfer
