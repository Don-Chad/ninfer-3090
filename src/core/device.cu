#include "core/device.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

// cudaDeviceScheduleAuto -- the default with no cudaSetDeviceFlags call -- spin-waits on every
// synchronize whenever the host has more CPU cores than active CUDA contexts, holding one core at
// 100% for as long as the GPU is busy. Spin keeps that behavior (and its low latency) as the
// explicit default; NINFER_CUDA_SYNC=blocking trades some decode latency for CPU headroom.
constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const char* value = std::getenv("NINFER_CUDA_SYNC");
    if (value == nullptr) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("NINFER_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

// cudaSetDeviceFlags fails once a device's primary context exists, and that context outlives any
// single DeviceContext -- release() tears down streams and events, not the primary context. A
// later DeviceContext on the same physical device must therefore not call cudaSetDeviceFlags
// again; it reuses the schedule already active there, validating that this construction is not
// silently asking for a different one it has no way to honor.
struct DeviceSyncState {
    std::mutex mutex;
    std::unordered_map<int, unsigned int> applied_flags;
};

DeviceSyncState& device_sync_state() {
    static DeviceSyncState state;
    return state;
}

void apply_sync_schedule_once(int device, unsigned int sync_flags) {
    DeviceSyncState& state = device_sync_state();
    const std::scoped_lock lock(state.mutex);
    const auto existing = state.applied_flags.find(device);
    if (existing != state.applied_flags.end()) {
        if (existing->second != sync_flags) {
            throw std::runtime_error(
                "NINFER_CUDA_SYNC schedule for CUDA device " + std::to_string(device) +
                " is already fixed by an earlier DeviceContext in this process and cannot change");
        }
        return;
    }
    const cudaError_t err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }
    state.applied_flags.emplace(device, sync_flags);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceBinding::DeviceBinding(int device) {
    const cudaError_t get = cudaGetDevice(&previous_);
    if (get != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDevice failed", get));
    }
    if (previous_ == device) { return; }
    const cudaError_t set = cudaSetDevice(device);
    if (set != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", set));
    }
    changed_ = true;
}

DeviceBinding::~DeviceBinding() noexcept {
    if (changed_) { log_cuda_error("cudaSetDevice", cudaSetDevice(previous_)); }
}

DeviceContext::DeviceContext(int device_id)
    : DeviceContext(std::span<const int>(&device_id, 1)) {}

DeviceContext::DeviceContext(std::span<const int> device_ids) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_ids.empty() || device_ids.size() > kMaxRanks) {
        throw std::invalid_argument("DeviceContext requires between one and " +
                                    std::to_string(kMaxRanks) + " CUDA devices");
    }
    device_ids_.assign(device_ids.begin(), device_ids.end());
    for (std::size_t i = 0; i < device_ids_.size(); ++i) {
        const int id = device_ids_[i];
        if (id < 0 || id >= count) {
            throw std::runtime_error("CUDA device " + std::to_string(id) + " does not exist: " +
                                     std::to_string(count) +
                                     (count == 1 ? " device is visible" : " devices are visible"));
        }
        // A repeated id is allowed on purpose. Several ranks on one card is not a useful
        // deployment -- it frees no memory -- but it exercises the entire multi-rank path
        // (per-rank binding and materialization, per-rank workspaces, and the cross-rank copies)
        // on a single-GPU machine. Every bug caught that way is one not caught by renting a
        // second card.
    }

    endpoints_.resize(device_ids_.size());
    try {
        for (std::size_t rank = 0; rank < device_ids_.size(); ++rank) {
            RankContext& endpoint = endpoints_[rank];
            endpoint.index        = rank;
            endpoint.device       = device_ids_[rank];
            err                = cudaSetDevice(endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
            }
            // Flags a physical device's schedule exactly once for the life of the process --
            // covers both a repeated device id within this construction (several ranks on one
            // GPU) and an earlier, already-destroyed DeviceContext that flagged it first.
            apply_sync_schedule_once(endpoint.device, sync_flags);
            err = cudaGetDeviceProperties(&endpoint.props, endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaGetDeviceProperties failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.transfer_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.vision_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(vision_stream) failed", err));
            }
            err = cudaEventCreateWithFlags(&endpoint.fence, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaEventCreateWithFlags(fence) failed", err));
            }
        }

        // Every rank must share one compute capability, so kernels built for it run on all of them.
        for (std::size_t rank = 1; rank < endpoints_.size(); ++rank) {
            if (endpoints_[rank].props.major != endpoints_[0].props.major ||
                endpoints_[rank].props.minor != endpoints_[0].props.minor) {
                throw std::invalid_argument(
                    "model-parallel CUDA devices must have the same compute capability");
            }
        }

        // peer_matrix_[from][to]: can `from` DMA into `to`. Ranks on one card need no peer setup:
        // copies between them are ordinary device-to-device, and enabling peer access to self is an
        // error, so they are recorded as reachable.
        //
        // Peer access is a performance capability, not a requirement. NVIDIA disables P2P over PCIe
        // on GeForce, so a pair of 3090s without an NVLink bridge reports can_access == 0 -- but
        // transfers still work there, staged through host memory. Refusing to start would rule out
        // the most common consumer multi-GPU box for no reason, so record the capability and let
        // callers choose a transport that suits it. (Neither a 2x 3090 nor a 2x A4000 PCIe box
        // measured so far reported any.)
        const std::size_t ranks = endpoints_.size();
        peer_matrix_.assign(ranks * ranks, 0);
        peer_access_ = true;
        for (std::size_t source = 0; source < ranks; ++source) {
            for (std::size_t destination = 0; destination < ranks; ++destination) {
                char& reachable = peer_matrix_[source * ranks + destination];
                if (endpoints_[source].device == endpoints_[destination].device) {
                    reachable = 1;
                    continue;
                }
                int can_access = 0;
                err = cudaDeviceCanAccessPeer(&can_access, endpoints_[source].device,
                                              endpoints_[destination].device);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        cuda_error_message("cudaDeviceCanAccessPeer failed", err));
                }
                if (can_access != 0) {
                    err = cudaSetDevice(endpoints_[source].device);
                    if (err != cudaSuccess) {
                        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
                    }
                    err = cudaDeviceEnablePeerAccess(endpoints_[destination].device, 0);
                    if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        (void)cudaGetLastError();
                        reachable = 1;
                    } else if (err != cudaSuccess) {
                        // Capability without a usable mapping: fall back rather than fail outright.
                        (void)cudaGetLastError();
                    } else {
                        reachable = 1;
                    }
                }
                if (reachable == 0) { peer_access_ = false; }
            }
        }
    } catch (...) {
        release();
        throw;
    }

    active_rank_ = 0;
    err          = cudaSetDevice(endpoints_[0].device);
    if (err != cudaSuccess) {
        release();
        throw std::runtime_error(cuda_error_message("cudaSetDevice(primary) failed", err));
    }
    refresh_active_aliases();
}

DeviceContext::~DeviceContext() { release(); }

void DeviceContext::release() noexcept {
    for (RankContext& endpoint : endpoints_) {
        if (endpoint.stream != nullptr || endpoint.transfer_stream != nullptr ||
            endpoint.vision_stream != nullptr || endpoint.fence != nullptr) {
            log_cuda_error("cudaSetDevice", cudaSetDevice(endpoint.device));
        }
        destroy_event(endpoint.fence);
        destroy_stream(endpoint.vision_stream);
        destroy_stream(endpoint.transfer_stream);
        destroy_stream(endpoint.stream);
    }
    endpoints_.clear();
    device_ids_.clear();
    peer_matrix_.clear();
    device          = 0;
    stream          = nullptr;
    transfer_stream = nullptr;
    vision_stream   = nullptr;
    props           = {};
    active_rank_    = 0;
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : endpoints_(std::move(other.endpoints_)), device_ids_(std::move(other.device_ids_)),
      peer_matrix_(std::move(other.peer_matrix_)), active_rank_(other.active_rank_),
      peer_access_(other.peer_access_) {
    refresh_active_aliases();
    other.device                  = 0;
    other.stream                  = nullptr;
    other.transfer_stream         = nullptr;
    other.vision_stream           = nullptr;
    other.props                   = {};
    other.active_rank_            = 0;
    other.peer_access_            = false;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    release();
    endpoints_              = std::move(other.endpoints_);
    device_ids_             = std::move(other.device_ids_);
    peer_matrix_            = std::move(other.peer_matrix_);
    active_rank_            = other.active_rank_;
    peer_access_            = other.peer_access_;
    refresh_active_aliases();
    other.device                  = 0;
    other.stream                  = nullptr;
    other.transfer_stream         = nullptr;
    other.vision_stream           = nullptr;
    other.props                   = {};
    other.active_rank_            = 0;
    other.peer_access_            = false;
    return *this;
}

void DeviceContext::refresh_active_aliases() noexcept {
    if (endpoints_.empty()) {
        device          = 0;
        stream          = nullptr;
        transfer_stream = nullptr;
        vision_stream   = nullptr;
        props           = {};
        return;
    }
    const RankContext& endpoint = endpoints_[active_rank_];
    device                   = endpoint.device;
    stream                   = endpoint.stream;
    transfer_stream          = endpoint.transfer_stream;
    vision_stream            = endpoint.vision_stream;
    props                    = endpoint.props;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

// Streaming-multiprocessor count of the active device. Distinct from compute_capability(): every
// sm_86 part shares capability 86 but not this count, so residency budgets must read this.
int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

std::size_t DeviceContext::size() const noexcept { return endpoints_.size(); }

bool DeviceContext::model_parallel() const noexcept { return endpoints_.size() > 1; }

const RankContext& DeviceContext::rank(std::size_t index) const {
    if (index >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[index];
}

bool DeviceContext::same_physical_device(std::size_t a, std::size_t b) const {
    return rank(a).device == rank(b).device;
}

bool DeviceContext::peer_access() const noexcept { return peer_access_; }

bool DeviceContext::peer_access(std::size_t from, std::size_t to) const {
    if (from >= endpoints_.size() || to >= endpoints_.size()) {
        throw std::out_of_range("CUDA device rank is out of range");
    }
    return peer_matrix_[from * endpoints_.size() + to] != 0;
}

std::size_t DeviceContext::active_rank() const noexcept { return active_rank_; }

const std::vector<int>& DeviceContext::device_ids() const noexcept { return device_ids_; }

cudaStream_t DeviceContext::stream_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].stream;
}

cudaStream_t DeviceContext::transfer_stream_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].transfer_stream;
}

cudaEvent_t DeviceContext::fence_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].fence;
}

void DeviceContext::activate_rank(std::size_t rank) {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    active_rank_ = rank;
    refresh_active_aliases();
}

void DeviceContext::synchronize_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    CUDA_CHECK(cudaStreamSynchronize(endpoints_[rank].stream));
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

void DeviceContext::synchronize() const {
    for (const RankContext& endpoint : endpoints_) {
        CUDA_CHECK(cudaSetDevice(endpoint.device));
        CUDA_CHECK(cudaStreamSynchronize(endpoint.stream));
    }
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

ScopedDeviceRank::ScopedDeviceRank(DeviceContext& context, std::size_t rank)
    : context_(context), previous_rank_(context.active_rank()) {
    context_.activate_rank(rank);
}

ScopedDeviceRank::~ScopedDeviceRank() noexcept {
    if (context_.active_rank() != previous_rank_) { context_.activate_rank(previous_rank_); }
}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx)
    : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    cudaError_t err = cudaSetDevice(ctx.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice(timer) failed", err));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    err               = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx)
    : CudaCompletionEvent(ctx.rank(ctx.active_rank())) {}

CudaCompletionEvent::CudaCompletionEvent(const RankContext& rank) : device_(rank.device) {
    // The event belongs to this rank's device, which is the one that will record it.
    DeviceBinding bind(rank.device);
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}

RankFenceSet::RankFenceSet(const DeviceContext& context) {
    events_.reserve(context.size());
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        events_.emplace_back(context.rank(rank));
    }
}

void RankFenceSet::record(RankStreams streams) {
    for (std::size_t rank = 0; rank < events_.size(); ++rank) { events_[rank].record(streams[rank]); }
}

void RankFenceSet::wait(RankStreams streams) const {
    for (std::size_t rank = 0; rank < events_.size(); ++rank) { events_[rank].wait(streams[rank]); }
}

bool RankFenceSet::ready() const {
    for (const CudaCompletionEvent& event : events_) {
        if (!event.ready()) { return false; }
    }
    return true;
}

void RankFenceSet::synchronize() const {
    for (const CudaCompletionEvent& event : events_) { event.synchronize(); }
}

} // namespace ninfer
