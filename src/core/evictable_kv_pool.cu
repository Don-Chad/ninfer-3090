#include "core/evictable_kv_pool.h"

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

constexpr std::size_t kResidentPiece = 1024ULL * 1024ULL * 1024ULL;

} // namespace

struct EvictableKVPool::Impl {
    DeviceContext& device;
    Config config{};
    std::size_t granularity       = 0;
    std::size_t arena_reserved    = 0; // granularity-aligned home reservation
    std::size_t window_reserved   = 0; // granularity-aligned overlay reservation
    std::size_t lendable_granules = 0; // granules covering the KV payload prefix
    CUdeviceptr home              = 0;
    CUdeviceptr overlay           = 0;
    // One handle per piece: the lendable prefix in granules, the remainder in large pieces.
    // Offsets are arena offsets; the vectors are parallel and piece i of the prefix is granule i.
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
    std::vector<std::size_t> lent; // granule indices currently mapped at the overlay range
    bool poisoned = false;

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
        NINFER_CU_CHECK(cuMemMap(overlay + rank * granularity, sizes[piece], 0, handles[piece], 0));
        set_access(overlay + rank * granularity, sizes[piece]);
    }
};

bool EvictableKVPool::supported(const DeviceContext& device) {
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

std::size_t EvictableKVPool::device_granularity(const DeviceContext& device) {
    if (!supported(device)) { return 0; }
    CUmemAllocationProp prop{};
    prop.type            = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id     = device.device;
    std::size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
        CUDA_SUCCESS) {
        return 0;
    }
    return granularity;
}

EvictableKVPool::EvictableKVPool(DeviceContext& device, const Config& config)
    : impl_(std::make_unique<Impl>(device)) {
    if (config.arena_bytes == 0) {
        throw std::invalid_argument("evictable KV arena must not be empty");
    }
    if (config.lendable_prefix_bytes > config.arena_bytes) {
        throw std::invalid_argument("evictable KV lendable prefix exceeds the arena");
    }
    if (config.window_capacity_bytes == 0) {
        throw std::invalid_argument("evictable KV window capacity must be positive");
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
    if (impl.granularity == 0) {
        throw std::runtime_error("evictable KV pool read a zero VMM granularity");
    }
    if (kResidentPiece % impl.granularity != 0) {
        throw std::runtime_error("evictable KV resident piece is not a granularity multiple");
    }

    impl.arena_reserved = align_up(config.arena_bytes, impl.granularity);
    // Whole granules only: a granule that straddles the payload boundary stays resident.
    impl.lendable_granules = config.lendable_prefix_bytes / impl.granularity;
    impl.window_reserved   = align_up(config.window_capacity_bytes, impl.granularity);
    if (impl.window_reserved > impl.lendable_granules * impl.granularity) {
        throw std::invalid_argument("evictable KV window exceeds the lendable prefix");
    }

    NINFER_CU_CHECK(cuMemAddressReserve(&impl.home, impl.arena_reserved, impl.granularity, 0, 0));
    NINFER_CU_CHECK(
        cuMemAddressReserve(&impl.overlay, impl.window_reserved, impl.granularity, 0, 0));

    std::size_t offset = 0;
    for (std::size_t granule = 0; granule < impl.lendable_granules; ++granule) {
        impl.offsets.push_back(offset);
        impl.sizes.push_back(impl.granularity);
        offset += impl.granularity;
    }
    while (offset < impl.arena_reserved) {
        const std::size_t piece = std::min(kResidentPiece, impl.arena_reserved - offset);
        impl.offsets.push_back(offset);
        impl.sizes.push_back(piece);
        offset += piece;
    }

    impl.handles.resize(impl.offsets.size());
    for (std::size_t piece = 0; piece < impl.offsets.size(); ++piece) {
        NINFER_CU_CHECK(cuMemCreate(&impl.handles[piece], impl.sizes[piece], &prop, 0));
        impl.map_home(piece);
    }
}

EvictableKVPool::~EvictableKVPool() {
    if (impl_ == nullptr || impl_->home == 0) { return; }
    Impl& impl = *impl_;
    for (std::size_t rank = 0; rank < impl.lent.size(); ++rank) {
        (void)cuMemUnmap(impl.overlay + rank * impl.granularity, impl.granularity);
    }
    for (std::size_t piece = 0; piece < impl.handles.size(); ++piece) {
        const bool away = std::find(impl.lent.begin(), impl.lent.end(), piece) != impl.lent.end();
        if (!away) { (void)cuMemUnmap(impl.home + impl.offsets[piece], impl.sizes[piece]); }
        (void)cuMemRelease(impl.handles[piece]);
    }
    (void)cuMemAddressFree(impl.home, impl.arena_reserved);
    (void)cuMemAddressFree(impl.overlay, impl.window_reserved);
}

DeviceSpan EvictableKVPool::arena() const noexcept {
    return DeviceSpan{reinterpret_cast<void*>(impl_->home), impl_->config.arena_bytes};
}

std::size_t EvictableKVPool::granularity() const noexcept { return impl_->granularity; }

std::size_t EvictableKVPool::lendable_granules() const noexcept { return impl_->lendable_granules; }

std::size_t EvictableKVPool::window_capacity_bytes() const noexcept {
    return impl_->window_reserved;
}

bool EvictableKVPool::lease_open() const noexcept { return !impl_->lent.empty(); }

bool EvictableKVPool::poisoned() const noexcept { return impl_->poisoned; }

std::size_t EvictableKVPool::granule_of(std::size_t offset) const {
    if (offset >= impl_->config.arena_bytes) {
        throw std::out_of_range("evictable KV offset is outside the arena");
    }
    return offset / impl_->granularity;
}

DeviceSpan EvictableKVPool::granule_bytes(std::size_t index) const {
    if (index >= impl_->lendable_granules) {
        throw std::out_of_range("evictable KV granule is outside the lendable prefix");
    }
    return DeviceSpan{reinterpret_cast<void*>(impl_->home + index * impl_->granularity),
                      impl_->granularity};
}

EvictableKVPool::Transaction EvictableKVPool::lease(std::span<const std::size_t> granules,
                                                    cudaStream_t stream) {
    Impl& impl = *impl_;
    if (impl.poisoned) {
        throw std::runtime_error("evictable KV pool is poisoned by a failed return");
    }
    if (!impl.lent.empty()) { throw std::logic_error("evictable KV lease is already open"); }
    if (granules.empty()) { throw std::invalid_argument("evictable KV lease needs a granule"); }
    if (granules.size() * impl.granularity > impl.window_reserved) {
        throw std::invalid_argument("evictable KV lease exceeds the window capacity: " +
                                    std::to_string(granules.size()) + " granules");
    }
    for (std::size_t rank = 0; rank < granules.size(); ++rank) {
        if (granules[rank] >= impl.lendable_granules) {
            throw std::invalid_argument("evictable KV lease names a resident granule");
        }
        if (rank != 0 && granules[rank] <= granules[rank - 1]) {
            throw std::invalid_argument("evictable KV lease granules must strictly increase");
        }
    }
    // No stream is quiesced here, and that is the point of this pool: a lent granule lies
    // entirely inside free KV pages, so no in-flight kernel reads or writes it and other lanes
    // keep running while the window is open.
    const auto start = Clock::now();
    impl.lent.reserve(granules.size());
    for (std::size_t rank = 0; rank < granules.size(); ++rank) {
        const std::size_t piece = granules[rank];
        NINFER_CU_CHECK(cuMemUnmap(impl.home + impl.offsets[piece], impl.sizes[piece]));
        impl.map_overlay(piece, rank);
        impl.lent.push_back(piece);
    }
    return Transaction(*this,
                       DeviceSpan{reinterpret_cast<void*>(impl.overlay),
                                  granules.size() * impl.granularity},
                       stream, seconds_since(start));
}

void EvictableKVPool::give_back(Transaction& transaction) noexcept {
    Impl& impl = *impl_;
    if (impl.lent.empty()) { return; }
    const auto start = Clock::now();
    try {
        for (std::size_t rank = 0; rank < impl.lent.size(); ++rank) {
            NINFER_CU_CHECK(cuMemUnmap(impl.overlay + rank * impl.granularity, impl.granularity));
            impl.map_home(impl.lent[rank]);
        }
    } catch (const std::exception& error) {
        impl.poisoned = true;
        std::fprintf(stderr,
                     "ninfer: evictable KV pool return failed (%s); the KV cache is no longer "
                     "trustworthy\n",
                     error.what());
    } catch (...) {
        impl.poisoned = true;
        std::fprintf(stderr, "ninfer: evictable KV pool return failed; the KV cache is no longer "
                             "trustworthy\n");
    }
    impl.lent.clear();
    transaction.stats_.return_seconds = seconds_since(start);
}

EvictableKVPool::Transaction::Transaction(EvictableKVPool& pool, DeviceSpan leased,
                                          cudaStream_t stream, double lease_seconds) noexcept
    : pool_(&pool), leased_(leased), stream_(stream) {
    stats_.lease_seconds = lease_seconds;
    stats_.mapped_bytes  = leased.bytes;
}

EvictableKVPool::Transaction::~Transaction() { close(); }

EvictableKVPool::Transaction::Transaction(Transaction&& other) noexcept
    : pool_(other.pool_), leased_(other.leased_), stream_(other.stream_), stats_(other.stats_) {
    other.pool_ = nullptr;
}

EvictableKVPool::Transaction&
EvictableKVPool::Transaction::operator=(Transaction&& other) noexcept {
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

void EvictableKVPool::Transaction::close() noexcept {
    if (pool_ == nullptr) { return; }
    pool_->give_back(*this);
    pool_ = nullptr;
}

} // namespace ninfer
