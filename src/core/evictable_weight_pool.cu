#include "core/evictable_weight_pool.h"

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace {

using Clock = std::chrono::steady_clock;

void cu_check(CUresult result, const char* expr) {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(expr) + " failed: " +
                             (name != nullptr ? name : "unknown CUresult"));
}

#define NINFER_CU_CHECK(expr) ::ninfer::cu_check((expr), #expr)

void ensure_driver_initialized() {
    static std::once_flag once;
    std::call_once(once, [] { NINFER_CU_CHECK(cuInit(0)); });
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

struct EvictableWeightPool::Impl {
    DeviceContext& device;
    Config config{};
    std::size_t granularity     = 0;
    std::size_t arena_reserved  = 0; // chunk-aligned home reservation
    std::size_t window_reserved = 0; // chunk-aligned overlay reservation
    std::size_t tail_begin      = 0; // arena offset of the first evictable chunk
    std::size_t mirror_begin    = 0; // arena offset of the first mirrored byte
    std::size_t mirror_extent   = 0; // mirrored bytes: [mirror_begin, arena_bytes)
    CUdeviceptr home            = 0;
    CUdeviceptr overlay         = 0;
    // One handle per region piece: [0, tail_begin) in large pieces, the tail in kChunkBytes
    // chunks. Offsets are arena offsets; the vectors are parallel.
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
    std::size_t evicted_pieces = 0; // pieces currently mapped at the overlay range
    std::unique_ptr<PinnedHostBuffer> mirror;
    bool mirror_captured = false;
    bool poisoned        = false;

    explicit Impl(DeviceContext& context) : device(context) {}

    void set_access(CUdeviceptr va, std::size_t bytes) {
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = device.device;
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        NINFER_CU_CHECK(cuMemSetAccess(va, bytes, &access, 1));
    }

    void map_home(std::size_t piece) {
        NINFER_CU_CHECK(cuMemMap(home + offsets[piece], sizes[piece], 0, handles[piece], 0));
        set_access(home + offsets[piece], sizes[piece]);
    }

    void map_overlay(std::size_t piece, std::size_t rank) {
        NINFER_CU_CHECK(
            cuMemMap(overlay + rank * kChunkBytes, sizes[piece], 0, handles[piece], 0));
        set_access(overlay + rank * kChunkBytes, sizes[piece]);
    }
};

bool EvictableWeightPool::supported(const DeviceContext& device) {
    ensure_driver_initialized();
    CUdevice handle = 0;
    if (cuDeviceGet(&handle, device.device) != CUDA_SUCCESS) { return false; }
    int value = 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                             handle) != CUDA_SUCCESS) {
        return false;
    }
    return value != 0;
}

EvictableWeightPool::EvictableWeightPool(DeviceContext& device, const Config& config)
    : impl_(std::make_unique<Impl>(device)) {
    if (config.arena_bytes == 0) {
        throw std::invalid_argument("evictable pool arena must not be empty");
    }
    if (config.evictable_tail_bytes == 0 || config.evictable_tail_bytes > config.arena_bytes) {
        throw std::invalid_argument("evictable pool tail must be a nonempty arena suffix");
    }
    if (config.window_capacity_bytes == 0) {
        throw std::invalid_argument("evictable pool window capacity must be positive");
    }
    ensure_driver_initialized();
    Impl& impl  = *impl_;
    impl.config = config;

    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device.device;
    NINFER_CU_CHECK(cuMemGetAllocationGranularity(&impl.granularity, &prop,
                                                  CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (kChunkBytes % impl.granularity != 0) {
        throw std::runtime_error("evictable pool chunk is not a multiple of the VMM granularity");
    }

    impl.arena_reserved = align_up(config.arena_bytes, kChunkBytes);
    // Chunk-align into the evictable suffix so no chunk ever covers a resident object.
    impl.tail_begin      = align_up(config.arena_bytes - config.evictable_tail_bytes, kChunkBytes);
    impl.window_reserved = align_up(config.window_capacity_bytes, kChunkBytes);
    if (impl.window_reserved > impl.arena_reserved - impl.tail_begin) {
        throw std::invalid_argument("evictable pool window exceeds the evictable tail");
    }
    impl.mirror_begin  = impl.arena_reserved - impl.window_reserved;
    impl.mirror_extent = config.arena_bytes > impl.mirror_begin
                             ? config.arena_bytes - impl.mirror_begin
                             : 0;

    NINFER_CU_CHECK(cuMemAddressReserve(&impl.home, impl.arena_reserved, kChunkBytes, 0, 0));
    NINFER_CU_CHECK(cuMemAddressReserve(&impl.overlay, impl.window_reserved, kChunkBytes, 0, 0));

    constexpr std::size_t kPrefixPiece = 1024ULL * 1024ULL * 1024ULL;
    std::size_t offset                 = 0;
    while (offset < impl.tail_begin) {
        const std::size_t piece = std::min(kPrefixPiece, impl.tail_begin - offset);
        impl.offsets.push_back(offset);
        impl.sizes.push_back(piece);
        offset += piece;
    }
    while (offset < impl.arena_reserved) {
        impl.offsets.push_back(offset);
        impl.sizes.push_back(kChunkBytes);
        offset += kChunkBytes;
    }

    impl.handles.resize(impl.offsets.size());
    for (std::size_t piece = 0; piece < impl.offsets.size(); ++piece) {
        NINFER_CU_CHECK(cuMemCreate(&impl.handles[piece], impl.sizes[piece], &prop, 0));
        impl.map_home(piece);
    }
}

EvictableWeightPool::~EvictableWeightPool() {
    if (impl_ == nullptr || impl_->home == 0) { return; }
    Impl& impl               = *impl_;
    const std::size_t pieces = impl.handles.size();
    for (std::size_t piece = 0; piece < pieces; ++piece) {
        const bool away = impl.evicted_pieces != 0 && piece >= pieces - impl.evicted_pieces;
        if (away) {
            const std::size_t rank = piece - (pieces - impl.evicted_pieces);
            (void)cuMemUnmap(impl.overlay + rank * kChunkBytes, impl.sizes[piece]);
        } else {
            (void)cuMemUnmap(impl.home + impl.offsets[piece], impl.sizes[piece]);
        }
        (void)cuMemRelease(impl.handles[piece]);
    }
    (void)cuMemAddressFree(impl.home, impl.arena_reserved);
    (void)cuMemAddressFree(impl.overlay, impl.window_reserved);
}

DeviceSpan EvictableWeightPool::arena() const noexcept {
    return DeviceSpan{reinterpret_cast<void*>(impl_->home), impl_->config.arena_bytes};
}

std::size_t EvictableWeightPool::evictable_tail_bytes() const noexcept {
    return impl_->arena_reserved - impl_->tail_begin;
}

std::size_t EvictableWeightPool::window_capacity_bytes() const noexcept {
    return impl_->window_reserved;
}

std::size_t EvictableWeightPool::mirror_bytes() const noexcept { return impl_->mirror_extent; }

bool EvictableWeightPool::mirror_captured() const noexcept { return impl_->mirror_captured; }

bool EvictableWeightPool::transaction_open() const noexcept {
    return impl_->evicted_pieces != 0;
}

bool EvictableWeightPool::poisoned() const noexcept { return impl_->poisoned; }

void EvictableWeightPool::capture_window_mirror(cudaStream_t stream) {
    Impl& impl = *impl_;
    if (impl.mirror_captured) {
        throw std::logic_error("evictable pool window mirror was already captured");
    }
    if (impl.mirror_extent != 0) {
        impl.mirror = std::make_unique<PinnedHostBuffer>(impl.mirror_extent);
        CUDA_CHECK(cudaMemcpyAsync(impl.mirror->data(),
                                   reinterpret_cast<void*>(impl.home + impl.mirror_begin),
                                   impl.mirror_extent, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    impl.mirror_captured = true;
}

EvictableWeightPool::Transaction EvictableWeightPool::evict(std::size_t bytes,
                                                            cudaStream_t stream) {
    Impl& impl = *impl_;
    if (!impl.mirror_captured) {
        throw std::logic_error("evictable pool has no window mirror; capture it after load");
    }
    if (impl.poisoned) {
        throw std::runtime_error("evictable pool is poisoned by a failed restore");
    }
    if (impl.evicted_pieces != 0) {
        throw std::logic_error("evictable pool transaction is already open");
    }
    if (bytes == 0) { throw std::invalid_argument("evictable pool cannot evict zero bytes"); }
    const std::size_t chunks = (bytes + kChunkBytes - 1) / kChunkBytes;
    const std::size_t extent = chunks * kChunkBytes;
    if (extent > impl.window_reserved) {
        throw std::invalid_argument("evictable pool request exceeds the window capacity: " +
                                    std::to_string(bytes) + " bytes");
    }
    // The borrowed chunks may still be read by in-flight work on either stream.
    CUDA_CHECK(cudaStreamSynchronize(impl.device.stream));
    CUDA_CHECK(cudaStreamSynchronize(impl.device.transfer_stream));

    const auto start        = Clock::now();
    const std::size_t first = impl.handles.size() - chunks;
    for (std::size_t rank = 0; rank < chunks; ++rank) {
        const std::size_t piece = first + rank;
        NINFER_CU_CHECK(cuMemUnmap(impl.home + impl.offsets[piece], impl.sizes[piece]));
        impl.map_overlay(piece, rank);
    }
    impl.evicted_pieces = chunks;
    return Transaction(*this, DeviceSpan{reinterpret_cast<void*>(impl.overlay), extent}, stream,
                       seconds_since(start));
}

void EvictableWeightPool::restore(Transaction& transaction) noexcept {
    Impl& impl = *impl_;
    if (impl.evicted_pieces == 0) { return; }
    const auto start         = Clock::now();
    const std::size_t chunks = impl.evicted_pieces;
    const std::size_t first  = impl.handles.size() - chunks;
    try {
        for (std::size_t rank = 0; rank < chunks; ++rank) {
            const std::size_t piece = first + rank;
            NINFER_CU_CHECK(cuMemUnmap(impl.overlay + rank * kChunkBytes, impl.sizes[piece]));
            impl.map_home(piece);
        }
        // Only the borrowed chunks were dirtied, and only bytes below arena_bytes hold weights.
        const std::size_t evicted_begin = impl.offsets[first];
        const std::size_t evicted_end   = std::min(impl.config.arena_bytes, impl.arena_reserved);
        if (evicted_end > evicted_begin) {
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(impl.home + evicted_begin),
                static_cast<const std::byte*>(impl.mirror->data()) +
                    (evicted_begin - impl.mirror_begin),
                evicted_end - evicted_begin, cudaMemcpyHostToDevice, transaction.stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(transaction.stream_));
    } catch (const std::exception& error) {
        impl.poisoned = true;
        std::fprintf(stderr,
                     "ninfer: evictable weight pool restore failed (%s); the weights are no "
                     "longer trustworthy\n",
                     error.what());
    } catch (...) {
        impl.poisoned = true;
        std::fprintf(stderr, "ninfer: evictable weight pool restore failed; the weights are no "
                             "longer trustworthy\n");
    }
    impl.evicted_pieces                = 0;
    transaction.stats_.restore_seconds = seconds_since(start);
}

EvictableWeightPool::Transaction::Transaction(EvictableWeightPool& pool, DeviceSpan leased,
                                              cudaStream_t stream, double evict_seconds) noexcept
    : pool_(&pool), leased_(leased), stream_(stream) {
    stats_.evict_seconds = evict_seconds;
    stats_.mapped_bytes  = leased.bytes;
}

EvictableWeightPool::Transaction::~Transaction() { close(); }

EvictableWeightPool::Transaction::Transaction(Transaction&& other) noexcept
    : pool_(other.pool_), leased_(other.leased_), stream_(other.stream_), stats_(other.stats_) {
    other.pool_ = nullptr;
}

EvictableWeightPool::Transaction&
EvictableWeightPool::Transaction::operator=(Transaction&& other) noexcept {
    if (this != &other) {
        close();
        pool_       = other.pool_;
        leased_     = other.leased_;
        stream_     = other.stream_;
        stats_      = other.stats_;
        other.pool_ = nullptr;
    }
    return *this;
}

void EvictableWeightPool::Transaction::close() noexcept {
    if (pool_ == nullptr) { return; }
    pool_->restore(*this);
    pool_ = nullptr;
}

} // namespace ninfer
