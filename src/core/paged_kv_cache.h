#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "core/transfer_work.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

/** Non-owning, single-sequence view consumed by growing-cache Ops. */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
};

/** Non-owning multi-sequence view consumed by batched growing-cache Ops. */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
};

/** Rebinds one checked single-sequence table row as a one-row batched view. */
[[nodiscard]] PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache);

// A plane is storage-only. Target code assigns K/V/layer meaning to plane indices.
struct KVPlaneGeometry {
    DType dtype                 = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t alignment       = 256;

    friend bool operator==(const KVPlaneGeometry&, const KVPlaneGeometry&) = default;
};

enum class PagedKVPlaneOrder : std::uint8_t {
    PageMajor,
    HeadMajor,
};

struct KVPageGeometry {
    std::uint32_t page_tokens            = kPagedKVPageSize;
    PagedKVPlaneOrder device_plane_order = PagedKVPlaneOrder::PageMajor;
    std::vector<KVPlaneGeometry> planes;

    friend bool operator==(const KVPageGeometry&, const KVPageGeometry&) = default;
};

struct DeviceKVPagePoolSpec {
    std::uint32_t page_group_count = 0;
    KVPageGeometry geometry;
};

struct KVExecutionTableSpec {
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
};

struct DeviceKVPlaneLayout {
    KVPlaneGeometry geometry;
    // Which rank's device memory holds the plane, and so which backing `storage` is a region of.
    std::size_t rank = 0;
    TensorRegion storage;
};

// Consecutive physical pages of one pool. The free list is kept sorted and coalesced, so a run
// is the natural unit for both allocation and lending.
struct KVPageRun {
    std::int32_t begin  = 0;
    std::uint32_t count = 0;
};

// Device bytes one plane devotes to a page run.
struct KVPlaneByteRange {
    const void* base  = nullptr;
    std::size_t bytes = 0;
};

struct DeviceKVPagePoolLayout {
    DeviceKVPagePoolSpec spec;
    std::vector<DeviceKVPlaneLayout> planes;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

struct KVExecutionTableLayout {
    KVExecutionTableSpec spec;
    // The rank whose backing holds this copy of the table.
    std::size_t rank = 0;
    TensorRegion block_tables;

    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

[[nodiscard]] DeviceKVPagePoolLayout plan_device_kv_page_pool(LayoutBuilder& builder,
                                                              const DeviceKVPagePoolSpec& spec);

// Plans a pool whose planes live on several ranks: plane `i` is laid out in
// `builders[plane_rank[i]]`, so each rank's planes are contiguous in that rank's own backing.
[[nodiscard]] DeviceKVPagePoolLayout
plan_device_kv_page_pool(std::span<LayoutBuilder* const> builders,
                         std::span<const std::size_t> plane_rank, const DeviceKVPagePoolSpec& spec);

// `rank` says whose backing `builder` lays out. A pool spanning several ranks plans one table per
// rank that runs attention, since a kernel cannot read another device's copy.
[[nodiscard]] KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                              const KVExecutionTableSpec& spec,
                                                              std::size_t rank = 0);

class DeviceKVPagePool;
class KVExecutionTablePool;
class HostKVAllocationView;
struct HostKVPageLayout;
class HostKVAllocationConstView;

/** Copyable, non-owning physical-page capability minted by one DeviceKVPagePool. */
class DeviceKVPageHandle {
public:
    DeviceKVPageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

private:
    friend class DeviceKVPagePool;
    friend class DeviceKVPageLease;
    friend class KVExecutionTablePool;

    DeviceKVPageHandle(const DeviceKVPagePool* owner, std::int32_t index,
                       std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const DeviceKVPagePool* owner_ = nullptr;
    std::int32_t index_            = -1;
    std::uint32_t generation_      = 0;
};

/** Move-only owner of one complete Device page-group replica. */
class DeviceKVPageLease {
public:
    DeviceKVPageLease() noexcept = default;
    ~DeviceKVPageLease();

    DeviceKVPageLease(const DeviceKVPageLease&)            = delete;
    DeviceKVPageLease& operator=(const DeviceKVPageLease&) = delete;
    DeviceKVPageLease(DeviceKVPageLease&& other) noexcept;
    DeviceKVPageLease& operator=(DeviceKVPageLease&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] DeviceKVPageHandle handle() const noexcept;
    [[nodiscard]] bool belongs_to(const DeviceKVPagePool& pool) const noexcept;
    bool release() noexcept;

private:
    friend class DeviceKVPagePool;

    DeviceKVPageLease(DeviceKVPagePool& owner, std::int32_t index,
                      std::uint32_t generation) noexcept
        : owner_(&owner), index_(index), generation_(generation) {}

    DeviceKVPagePool* owner_  = nullptr;
    std::int32_t index_       = -1;
    std::uint32_t generation_ = 0;
};

/** Move-only reservation of capacity not yet associated with physical page IDs. */
class DeviceKVPageReservation {
public:
    DeviceKVPageReservation() noexcept = default;
    ~DeviceKVPageReservation();

    DeviceKVPageReservation(const DeviceKVPageReservation&)            = delete;
    DeviceKVPageReservation& operator=(const DeviceKVPageReservation&) = delete;
    DeviceKVPageReservation(DeviceKVPageReservation&& other) noexcept;
    DeviceKVPageReservation& operator=(DeviceKVPageReservation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::uint32_t pages() const noexcept { return pages_; }

    [[nodiscard]] bool belongs_to(const DeviceKVPagePool& pool) const noexcept;
    void clear() noexcept;
    void release() noexcept;

private:
    friend class DeviceKVPagePool;

    DeviceKVPageReservation(DeviceKVPagePool& owner, std::uint32_t pages) noexcept
        : owner_(&owner), pages_(pages) {}

    DeviceKVPagePool* owner_ = nullptr;
    std::uint32_t pages_     = 0;
};

// One page-group allocator over planes that may be spread across ranks. The allocator, its
// generations and every reservation are rank-agnostic: a page group id names the same slice in every
// plane wherever the plane lives, which is what lets the layers above (admission, prefix reuse, the
// context cache) stay unaware of how many devices hold the cache.
class DeviceKVPagePool {
public:
    // `backings[r]` is rank r's device memory; a plane bound to rank r is a region of it.
    DeviceKVPagePool(std::span<const DeviceSpan> backings, const DeviceKVPagePoolLayout& layout);
    // Every plane on rank 0.
    DeviceKVPagePool(DeviceSpan backing, const DeviceKVPagePoolLayout& layout);

    DeviceKVPagePool(const DeviceKVPagePool&)            = delete;
    DeviceKVPagePool& operator=(const DeviceKVPagePool&) = delete;
    DeviceKVPagePool(DeviceKVPagePool&&)                 = delete;
    DeviceKVPagePool& operator=(DeviceKVPagePool&&)      = delete;

    [[nodiscard]] const KVPageGeometry& geometry() const noexcept { return spec_.geometry; }

    // Physical extent of the pool; page indices are validated against it and it never changes.
    [[nodiscard]] std::uint32_t capacity_pages() const noexcept;
    // Pages the pool may still hand out: the physical extent minus the pages lent away.
    [[nodiscard]] std::uint32_t usable_pages() const noexcept;
    [[nodiscard]] std::uint32_t allocated_pages() const noexcept;
    [[nodiscard]] std::uint32_t reserved_pages() const noexcept;
    [[nodiscard]] std::uint32_t lent_pages() const noexcept;
    [[nodiscard]] std::uint32_t available_pages() const noexcept;
    [[nodiscard]] std::size_t plane_count() const noexcept;
    [[nodiscard]] const Tensor& plane(std::size_t index) const;
    // Ranks the planes span (the highest plane rank plus one) and the rank holding plane `index`.
    [[nodiscard]] std::size_t rank_count() const noexcept { return rank_count_; }
    [[nodiscard]] std::size_t plane_rank(std::size_t index) const;
    [[nodiscard]] std::span<const KVPageRun> free_runs() const noexcept;
    // Device bytes one plane devotes to a run of consecutive pages. Page-major geometry only:
    // a run is one contiguous block there, which is what makes a run lendable. Lending exists for a
    // single device, so a pool spanning several ranks refuses it.
    [[nodiscard]] KVPlaneByteRange plane_page_range(std::size_t plane, std::int32_t first_page,
                                                    std::uint32_t count) const;

    // Removes a wholly free run from circulation so its device memory can be lent elsewhere, and
    // puts it back. The caller owns the memory only between these two calls.
    void lend_pages(std::int32_t begin, std::uint32_t count);
    void return_pages(std::int32_t begin, std::uint32_t count);
    [[nodiscard]] std::uint32_t
    contiguous_run_count(std::span<const DeviceKVPageHandle> pages) const;

    [[nodiscard]] std::optional<DeviceKVPageReservation> reserve(std::uint32_t pages) noexcept;

    [[nodiscard]] DeviceKVPageReservation make_empty_reservation() noexcept {
        return DeviceKVPageReservation(*this, 0);
    }

    [[nodiscard]] bool can_resize_reservation(const DeviceKVPageReservation& reservation,
                                              std::uint32_t new_reserved_pages) const noexcept;
    void resize_reservation(DeviceKVPageReservation& reservation, std::uint32_t new_reserved_pages);

    // Grows destination to target_page_count without host allocation or a second capacity check.
    void materialize(DeviceKVPageReservation& reservation, std::uint32_t target_page_count,
                     std::vector<DeviceKVPageLease>& destination,
                     std::optional<DeviceKVPageHandle> preferred_predecessor = std::nullopt);
    // Single-page forms for fixed-capacity logical stores which do not own a growable lease vector.
    [[nodiscard]] DeviceKVPageLease materialize_one(DeviceKVPageReservation& reservation);
    // Returns trailing leases to the reservation, keeping their capacity reserved.
    void dematerialize(DeviceKVPageReservation& reservation, std::uint32_t target_page_count,
                       std::vector<DeviceKVPageLease>& source);
    void dematerialize_one(DeviceKVPageReservation& reservation, DeviceKVPageLease&& page);

    // Data movement. Each plane's copy is issued on the stream of the rank that holds it, so the
    // work for a page group fans out across ranks and the caller fences on all of them. The host
    // image of a page keeps the full plane inventory: ranks write disjoint plane ranges of it.
    // Transfers return the physical work they issued (payload bytes and cudaMemcpy calls).
    void zero_pages(std::span<const DeviceKVPageHandle> pages, RankStreams streams = {}) const;
    TransferWork copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                           RankStreams streams = {}) const;

    // One run assumes contiguous Host records and contiguous Device page IDs. Before Device
    // allocation, sum this work over known Host runs as a nominal transfer estimate.
    [[nodiscard]] TransferWork host_transfer_run_work(std::uint32_t pages) const;
    [[nodiscard]] TransferWork device_copy_work(std::uint32_t pages) const;

    TransferWork copy_to_host(std::span<const DeviceKVPageHandle> source,
                              HostKVAllocationView destination, RankStreams streams = {}) const;
    TransferWork copy_from_host(HostKVAllocationConstView source,
                                std::span<const DeviceKVPageHandle> destination,
                                RankStreams streams = {}) const;
    // The same copies against caller-owned memory laid out by `host`, for images that do not live
    // in a Host KV arena (session snapshots). `host` must describe this pool's geometry.
    TransferWork copy_to_host(std::span<const DeviceKVPageHandle> source, std::byte* destination,
                              const HostKVPageLayout& host, RankStreams streams = {}) const;
    TransferWork copy_from_host(const std::byte* source, const HostKVPageLayout& host,
                                std::span<const DeviceKVPageHandle> destination,
                                RankStreams streams = {}) const;

private:
    friend class DeviceKVPageLease;
    friend class DeviceKVPageReservation;
    friend class KVExecutionTablePool;

    [[nodiscard]] bool valid_handle(DeviceKVPageHandle handle) const noexcept;
    [[nodiscard]] std::int32_t physical_index(DeviceKVPageHandle handle) const;
    void validate_distinct_pages(std::span<const DeviceKVPageHandle> pages,
                                 const char* duplicate_message) const;
    void consume_free_run(std::size_t run_index, std::int32_t begin, std::uint32_t count) noexcept;
    void release_free_page(std::int32_t index) noexcept;
    void release_page(std::int32_t index, std::uint32_t generation) noexcept;
    void release_reservation(std::uint32_t pages) noexcept;

    void initialize_host_transfer_plan();
    // True when `host` is exactly the layout the transfer plan was built from: the plan copies to
    // its own cached offsets, so a same-geometry layout with different plane offsets must be
    // refused rather than silently written to the canonical ones.
    [[nodiscard]] bool host_layout_is_canonical(const HostKVPageLayout& host) const noexcept;
    template <bool ToHost>
    TransferWork copy_host_pages(std::span<const DeviceKVPageHandle> pages,
                                 std::conditional_t<ToHost, std::byte*, const std::byte*> host,
                                 const RankStreams& streams) const;

    struct HostTransferPlane {
        std::size_t host_offset    = 0;
        std::size_t page_bytes     = 0;
        std::size_t head_bytes     = 0;
        std::uint32_t heads        = 0;
        bool page_copies_supported = false;

        [[nodiscard]] bool copy_by_page(std::size_t pages) const noexcept {
            return page_copies_supported && pages < heads;
        }
    };

    // Consecutive page-major planes with one Device and Host pitch, copied as one 2D operation
    // per page when a run is shorter than the group. A group never spans ranks: its planes share
    // one device allocation and one stream.
    struct HostTransferGroup {
        std::size_t first_plane  = 0;
        std::size_t plane_count  = 1;
        std::size_t device_pitch = 0;
        std::size_t host_pitch   = 0;

        [[nodiscard]] bool copy_by_page(std::size_t pages) const noexcept {
            return pages < plane_count;
        }
    };

    DeviceKVPagePoolSpec spec_;
    std::vector<Tensor> planes_;
    std::vector<std::size_t> plane_ranks_;
    std::size_t rank_count_ = 1;
    std::vector<HostTransferPlane> host_transfer_planes_;
    std::vector<HostTransferGroup> host_transfer_groups_;
    // Index is min(run pages, size - 1); the last entry covers the ordinary long-run route.
    std::vector<std::uint32_t> host_transfer_operations_;
    std::size_t host_page_stride_         = 0;
    std::uint64_t page_payload_bytes_     = 0;
    std::size_t maximum_host_plane_group_ = 1;
    std::vector<KVPageRun> free_page_runs_;
    std::vector<std::uint32_t> page_generations_;
    std::vector<bool> page_allocated_;
    mutable std::vector<std::uint32_t> validation_marks_;
    mutable std::uint32_t validation_stamp_ = 0;
    std::uint32_t allocated_pages_          = 0;
    std::uint32_t reserved_pages_           = 0;
    std::uint32_t lent_pages_               = 0;
};

struct DeviceKVPageReservationRequest {
    DeviceKVPagePool* pool = nullptr;
    std::uint32_t pages    = 0;
};

// Reserves every distinct physical pool or leaves every pool unchanged.
[[nodiscard]] std::vector<DeviceKVPageReservation>
reserve_device_kv_page_bundle(std::span<const DeviceKVPageReservationRequest> requests);

class KVExecutionRowHandle {
public:
    KVExecutionRowHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::int32_t row_index() const noexcept { return row_; }

private:
    friend class KVExecutionTablePool;
    friend class KVExecutionRowLease;

    KVExecutionRowHandle(const KVExecutionTablePool* owner, std::int32_t row,
                         std::uint32_t generation) noexcept
        : owner_(owner), row_(row), generation_(generation) {}

    const KVExecutionTablePool* owner_ = nullptr;
    std::int32_t row_                  = -1;
    std::uint32_t generation_          = 0;
};

class KVExecutionRowLease {
public:
    KVExecutionRowLease() noexcept = default;
    ~KVExecutionRowLease();

    KVExecutionRowLease(const KVExecutionRowLease&)            = delete;
    KVExecutionRowLease& operator=(const KVExecutionRowLease&) = delete;
    KVExecutionRowLease(KVExecutionRowLease&& other) noexcept;
    KVExecutionRowLease& operator=(KVExecutionRowLease&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] KVExecutionRowHandle handle() const noexcept;

    [[nodiscard]] std::int32_t row_index() const noexcept { return row_; }

    [[nodiscard]] bool belongs_to(const KVExecutionTablePool& pool) const noexcept;
    bool release() noexcept;

private:
    friend class KVExecutionTablePool;

    KVExecutionRowLease(KVExecutionTablePool& owner, std::int32_t row,
                        std::uint32_t generation) noexcept
        : owner_(&owner), row_(row), generation_(generation) {}

    KVExecutionTablePool* owner_ = nullptr;
    std::int32_t row_            = -1;
    std::uint32_t generation_    = 0;
};

// The logical-to-physical page map every attention layer reads. One host shadow is the source of
// truth; each rank that runs attention holds a device copy of its own, and publishing writes them
// all, so a mapping is visible on every rank once each rank's stream has passed the copy.
class KVExecutionTablePool {
public:
    // `backings[r]` is rank r's device memory; a layout with rank r is a region of it. One copy per
    // layout, at most one per rank.
    KVExecutionTablePool(std::span<const DeviceSpan> backings,
                         std::span<const KVExecutionTableLayout> layouts,
                         const DeviceKVPagePool& pages);
    // A single copy on rank 0.
    KVExecutionTablePool(DeviceSpan backing, const KVExecutionTableLayout& layout,
                         const DeviceKVPagePool& pages);

    KVExecutionTablePool(const KVExecutionTablePool&)            = delete;
    KVExecutionTablePool& operator=(const KVExecutionTablePool&) = delete;
    KVExecutionTablePool(KVExecutionTablePool&&)                 = delete;
    KVExecutionTablePool& operator=(KVExecutionTablePool&&)      = delete;

    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept;
    [[nodiscard]] std::int32_t row_count() const noexcept;
    [[nodiscard]] KVExecutionRowLease acquire(std::int32_t row);

    // Each copy is written on the stream of the rank that holds it.
    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageHandle> pages, RankStreams streams);
    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageLease> pages, RankStreams streams);
    void publish_repeated(KVExecutionRowHandle row, DeviceKVPageHandle page, std::uint32_t count,
                          RankStreams streams);

    // The row and matrix as rank `rank` sees them. Throws if that rank holds no copy.
    [[nodiscard]] Tensor row(KVExecutionRowHandle handle, std::size_t rank = 0) const;
    [[nodiscard]] const Tensor& matrix(std::size_t rank = 0) const;
    // Ranks that hold a copy, in the order the layouts were given.
    [[nodiscard]] std::span<const std::size_t> replica_ranks() const noexcept {
        return replica_ranks_;
    }

private:
    friend class KVExecutionRowLease;

    [[nodiscard]] bool valid_handle(KVExecutionRowHandle handle) const noexcept;
    bool release_row(std::int32_t row, std::uint32_t generation) noexcept;
    void publish_indices(KVExecutionRowHandle row, std::uint32_t logical_begin,
                         std::span<const std::int32_t> indices, RankStreams streams);
    [[nodiscard]] const Tensor& replica(std::size_t rank) const;

    KVExecutionTableSpec spec_;
    const DeviceKVPagePool* pages_ = nullptr;
    std::vector<Tensor> replicas_;
    std::vector<std::size_t> replica_ranks_;
    PinnedHostBuffer host_shadow_;
    std::vector<bool> row_in_use_;
    std::vector<std::uint32_t> row_generations_;
};

} // namespace ninfer
