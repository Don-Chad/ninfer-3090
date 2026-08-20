#include "core/evictable_weight_pool.h"

#include "core/device.h"

#include <cuda.h>

#include <algorithm>
#include <chrono>
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

} // namespace

struct EvictableWeightPool::Impl {
    Config config{};
    std::size_t granularity     = 0;
    std::size_t arena_reserved  = 0;
    std::size_t overlay_reserved = 0;
    std::size_t tail_begin      = 0;   // arena offset of the first evictable chunk
    CUdeviceptr home            = 0;
    CUdeviceptr overlay         = 0;
    // One handle per region piece: [0, tail_begin) in large pieces, the tail in
    // kChunkBytes chunks. Offsets are arena offsets; the vectors are parallel.
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
    std::size_t first_tail_piece = 0;
    std::size_t evicted_pieces   = 0;   // count of evicted pieces at the vector end
    std::unique_ptr<PinnedHostBuffer> mirror;
    double last_evict_seconds   = 0.0;
    double last_restore_seconds = 0.0;

    void map_home(std::size_t piece) {
        NINFER_CU_CHECK(cuMemMap(home + offsets[piece], sizes[piece], 0, handles[piece], 0));
        set_access(home + offsets[piece], sizes[piece]);
    }

    void set_access(CUdeviceptr va, std::size_t bytes) {
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = config.device;
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        NINFER_CU_CHECK(cuMemSetAccess(va, bytes, &access, 1));
    }
};

bool EvictableWeightPool::supported(int device) {
    ensure_driver_initialized();
    CUdevice handle = 0;
    if (cuDeviceGet(&handle, device) != CUDA_SUCCESS) { return false; }
    int value = 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                             handle) != CUDA_SUCCESS) {
        return false;
    }
    return value != 0;
}

EvictableWeightPool::EvictableWeightPool(const Config& config) : impl_(std::make_unique<Impl>()) {
    if (config.arena_bytes == 0) {
        throw std::invalid_argument("evictable pool arena must not be empty");
    }
    if (config.evictable_tail_bytes == 0 || config.evictable_tail_bytes > config.arena_bytes) {
        throw std::invalid_argument("evictable pool tail must be a nonempty arena suffix");
    }
    ensure_driver_initialized();
    Impl& impl  = *impl_;
    impl.config = config;

    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = config.device;
    NINFER_CU_CHECK(cuMemGetAllocationGranularity(&impl.granularity, &prop,
                                                  CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (kChunkBytes % impl.granularity != 0) {
        throw std::runtime_error("evictable pool chunk is not a multiple of the VMM granularity");
    }

    impl.arena_reserved = align_up(config.arena_bytes, kChunkBytes);
    // Chunk-align INTO the evictable suffix so no chunk ever covers non-evictable bytes.
    impl.tail_begin = align_up(config.arena_bytes - config.evictable_tail_bytes, kChunkBytes);
    // Address space is free: reserve overlay VA for the whole evictable tail so
    // a window may borrow any extent the ladder can supply, regardless of the
    // planning hint in config.overlay_bytes.
    impl.overlay_reserved = impl.arena_reserved - impl.tail_begin;

    NINFER_CU_CHECK(cuMemAddressReserve(&impl.home, impl.arena_reserved, kChunkBytes, 0, 0));
    NINFER_CU_CHECK(cuMemAddressReserve(&impl.overlay, impl.overlay_reserved, kChunkBytes, 0, 0));

    // Non-evictable prefix in large pieces, evictable tail in chunks.
    constexpr std::size_t kPrefixPiece = 1024ULL * 1024ULL * 1024ULL;
    std::size_t offset                 = 0;
    while (offset < impl.tail_begin) {
        const std::size_t piece = std::min(kPrefixPiece, impl.tail_begin - offset);
        impl.offsets.push_back(offset);
        impl.sizes.push_back(piece);
        offset += piece;
    }
    impl.first_tail_piece = impl.offsets.size();
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
    Impl& impl = *impl_;
    for (std::size_t piece = 0; piece < impl.handles.size(); ++piece) {
        const bool away = impl.evicted_pieces != 0 && piece >= impl.handles.size() -
                                                                  impl.evicted_pieces;
        if (away) {
            const std::size_t rank = piece - (impl.handles.size() - impl.evicted_pieces);
            (void)cuMemUnmap(impl.overlay + rank * kChunkBytes, impl.sizes[piece]);
        } else {
            (void)cuMemUnmap(impl.home + impl.offsets[piece], impl.sizes[piece]);
        }
        (void)cuMemRelease(impl.handles[piece]);
    }
    (void)cuMemAddressFree(impl.home, impl.arena_reserved);
    (void)cuMemAddressFree(impl.overlay, impl.overlay_reserved);
}

EvictableWeightPool::EvictableWeightPool(EvictableWeightPool&&) noexcept            = default;
EvictableWeightPool& EvictableWeightPool::operator=(EvictableWeightPool&&) noexcept = default;

DeviceSpan EvictableWeightPool::arena() const noexcept {
    return DeviceSpan{reinterpret_cast<void*>(impl_->home), impl_->config.arena_bytes};
}

std::size_t EvictableWeightPool::evictable_tail_bytes() const noexcept {
    return impl_->arena_reserved - impl_->tail_begin;
}

std::size_t EvictableWeightPool::overlay_capacity() const noexcept {
    return impl_->overlay_reserved;
}

void EvictableWeightPool::capture_tail_mirror(cudaStream_t stream) {
    Impl& impl = *impl_;
    if (impl.mirror != nullptr) {
        throw std::logic_error("evictable pool tail mirror was already captured");
    }
    const std::size_t mirrored = impl.config.arena_bytes - impl.tail_begin;
    impl.mirror                = std::make_unique<PinnedHostBuffer>(mirrored);
    CUDA_CHECK(cudaMemcpyAsync(impl.mirror->data(),
                               reinterpret_cast<void*>(impl.home + impl.tail_begin), mirrored,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::byte* EvictableWeightPool::evict(std::size_t bytes, std::size_t* mapped_bytes) {
    Impl& impl = *impl_;
    if (impl.mirror == nullptr) {
        throw std::logic_error("evictable pool has no tail mirror; capture it after load");
    }
    if (impl.evicted_pieces != 0) {
        throw std::logic_error("evictable pool transaction is already open");
    }
    if (bytes == 0) { throw std::invalid_argument("evictable pool cannot evict zero bytes"); }
    const std::size_t chunks = (bytes + kChunkBytes - 1) / kChunkBytes;
    const std::size_t extent = chunks * kChunkBytes;
    if (extent > impl.overlay_reserved || extent > impl.arena_reserved - impl.tail_begin) {
        throw std::invalid_argument("evictable pool request exceeds tail or overlay capacity: " +
                                    std::to_string(bytes) + " bytes");
    }
    const auto start        = Clock::now();
    const std::size_t first = impl.handles.size() - chunks;
    for (std::size_t rank = 0; rank < chunks; ++rank) {
        const std::size_t piece = first + rank;
        NINFER_CU_CHECK(cuMemUnmap(impl.home + impl.offsets[piece], impl.sizes[piece]));
        NINFER_CU_CHECK(
            cuMemMap(impl.overlay + rank * kChunkBytes, impl.sizes[piece], 0,
                     impl.handles[piece], 0));
        impl.set_access(impl.overlay + rank * kChunkBytes, impl.sizes[piece]);
    }
    impl.evicted_pieces     = chunks;
    impl.last_evict_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (mapped_bytes != nullptr) { *mapped_bytes = extent; }
    return reinterpret_cast<std::byte*>(impl.overlay);
}

void EvictableWeightPool::restore(cudaStream_t stream) {
    Impl& impl = *impl_;
    if (impl.evicted_pieces == 0) { return; }
    const auto start         = Clock::now();
    const std::size_t chunks = impl.evicted_pieces;
    const std::size_t first  = impl.handles.size() - chunks;
    for (std::size_t rank = 0; rank < chunks; ++rank) {
        const std::size_t piece = first + rank;
        NINFER_CU_CHECK(cuMemUnmap(impl.overlay + rank * kChunkBytes, impl.sizes[piece]));
        impl.map_home(piece);
    }
    // Only bytes below arena_bytes ever hold weights; the aligned slack stays untouched.
    const std::size_t evicted_begin = impl.offsets[first];
    const std::size_t evicted_end   = std::min(impl.config.arena_bytes, impl.arena_reserved);
    if (evicted_end > evicted_begin) {
        const std::size_t upload = evicted_end - evicted_begin;
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(impl.home + evicted_begin),
            static_cast<const std::byte*>(impl.mirror->data()) +
                (evicted_begin - impl.tail_begin),
            upload, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    impl.evicted_pieces       = 0;
    impl.last_restore_seconds = std::chrono::duration<double>(Clock::now() - start).count();
}

bool EvictableWeightPool::evicted() const noexcept { return impl_->evicted_pieces != 0; }

double EvictableWeightPool::last_evict_seconds() const noexcept {
    return impl_->last_evict_seconds;
}

double EvictableWeightPool::last_restore_seconds() const noexcept {
    return impl_->last_restore_seconds;
}

} // namespace ninfer
