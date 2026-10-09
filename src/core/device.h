#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;

    [[nodiscard]] constexpr DeviceExecutionView
    on_stream(cudaStream_t target_stream) const noexcept {
        return {target_stream, multiprocessor_count};
    }
};

// CUDA function attributes are scoped to a device context. A process-wide `static` result
// therefore leaves the same kernel unconfigured the first time it launches on a second GPU, which
// shows up as a launch failure or silently wrong output rather than as a clear error. Give each
// launcher specialization a cheap, device-keyed cache instead.
//
// Single-GPU behaviour is unchanged: the map holds exactly one entry.
template <typename Configure>
void configure_cuda_device_once(Configure&& configure) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaError_t> results;

    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaError_t result = cudaSuccess;
    {
        const std::scoped_lock lock(mutex);
        const auto existing = results.find(device);
        if (existing != results.end()) {
            result = existing->second;
        } else {
            result = std::forward<Configure>(configure)();
            results.emplace(device, result);
        }
    }
    CUDA_CHECK(result);
}

// The most ranks one context holds. Layer pipelines and tensor groups are both bounded well below
// this; the limit exists so a mistyped device list fails at construction.
inline constexpr std::size_t kMaxRanks = 8;

// Makes a CUDA device the calling thread's current device for a scope, and restores whatever was
// current before. It carries no state of its own, so nothing else can be left pointing at the wrong
// device by an exception or an early return.
class DeviceBinding {
public:
    explicit DeviceBinding(int device);
    ~DeviceBinding() noexcept;

    DeviceBinding(const DeviceBinding&)            = delete;
    DeviceBinding& operator=(const DeviceBinding&) = delete;

private:
    int previous_ = 0;
    bool changed_ = false;
};

// One device's execution resources inside a DeviceContext: its streams, fence and properties.
// Everything a rank owns is fixed at construction. Ranks may name the same physical device (a test
// mode that exercises the multi-rank paths on one card), in which case they still own distinct
// streams.
struct RankContext {
    std::size_t index = 0;
    int device        = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Carries a vision encode that runs beside the decode of other lanes. Empty unless a window
    // borrows free KV memory, which is what makes the overlap safe.
    cudaStream_t vision_stream = nullptr;
    cudaEvent_t fence          = nullptr;
    cudaDeviceProp props{};

    [[nodiscard]] int compute_capability() const noexcept {
        return props.major * 10 + props.minor;
    }
    // Streaming-multiprocessor count. Distinct from compute_capability(): every sm_86 part shares
    // capability 86 but not this count (RTX 3090 has 82, A4000 has 48), so any device-wide
    // residency or launch budget must read this.
    [[nodiscard]] int multiprocessor_count() const noexcept { return props.multiProcessorCount; }
    [[nodiscard]] std::size_t total_vram() const noexcept { return props.totalGlobalMem; }
    [[nodiscard]] DeviceExecutionView execution_view() const noexcept {
        return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
    }
};

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Carries a vision encode that runs beside the decode of other lanes. Empty unless a window
    // borrows free KV memory, which is what makes the overlap safe.
    cudaStream_t vision_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    // One entry keeps the single-device route. Several entries open one rank per entry for
    // model-parallel execution, up to kMaxRanks: matching compute capability is required across
    // all of them, before any weight is uploaded. Peer access is only probed and recorded as a
    // capability -- it is not required, since transfers stage through pinned host memory when it is
    // unavailable (see StageLink).
    explicit DeviceContext(std::span<const int> device_ids);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    // The active rank's device synchronization schedule ("spin", "blocking", "yield" or "auto"),
    // as selected by NINFER_CUDA_SYNC at construction. Reads the live CUDA flags rather than a
    // cached value, so it always reflects the device this rank is actually bound to.
    [[nodiscard]] const char* sync_mode() const;
    int compute_capability() const noexcept;
    // Streaming-multiprocessor count of the attached device. Distinct from compute_capability():
    // every sm_86 part shares capability 86 but not this count (RTX 3090 has 82, RTX 3090 Ti has
    // 84), so any device-wide residency budget must read this, not compute_capability().
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool model_parallel() const noexcept;
    // Rank `index`'s resources. Throws std::out_of_range past the end.
    [[nodiscard]] const RankContext& rank(std::size_t index) const;
    // The rank that takes input (embedding, ingress) and the rank that produces output (head,
    // sampling, egress). Both are rank 0 on one device.
    [[nodiscard]] const RankContext& entry() const { return rank(0); }
    [[nodiscard]] const RankContext& head() const { return rank(size() - 1); }
    // Whether two ranks are the same physical device.
    [[nodiscard]] bool same_physical_device(std::size_t a, std::size_t b) const;
    // True when the devices can DMA directly to each other. False is not an error: cudaMemcpyPeer
    // still works, staging through host memory at roughly 13us per hop instead of a couple. Only
    // consult this to pick between schedules -- a design crossing once per token does not care,
    // one crossing twice per layer does. The no-argument form is true only when every pair of
    // ranks can reach each other.
    [[nodiscard]] bool peer_access() const noexcept;
    [[nodiscard]] bool peer_access(std::size_t from, std::size_t to) const;
    [[nodiscard]] std::size_t active_rank() const noexcept;
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept;
    [[nodiscard]] cudaStream_t stream_for_rank(std::size_t rank) const;
    // Weight upload goes to the arena of the rank that owns the object, and a host-to-device copy
    // has to be issued on a stream belonging to the destination device.
    [[nodiscard]] cudaStream_t transfer_stream_for_rank(std::size_t rank) const;
    [[nodiscard]] cudaEvent_t fence_for_rank(std::size_t rank) const;
    void activate_rank(std::size_t rank);
    void synchronize_rank(std::size_t rank) const;
    void synchronize() const;
    int sm() const noexcept;

private:
    void refresh_active_aliases() noexcept;
    void release() noexcept;

    std::vector<RankContext> endpoints_;
    std::vector<int> device_ids_;
    // peer_matrix_[from * size + to]: whether `from` can DMA into `to`. A rank always reaches a
    // rank on its own physical device.
    std::vector<char> peer_matrix_;
    std::size_t active_rank_ = 0;
    bool peer_access_        = false;
};

// Makes rank `rank`'s device current for a scope and restores the previous current device.
// Unlike ScopedDeviceRank it does not touch the context's active-rank aliases, so it is safe
// anywhere a context is only read.
class RankBinding {
public:
    RankBinding(const DeviceContext& context, std::size_t rank)
        : binding_(context.rank(rank).device) {}

private:
    DeviceBinding binding_;
};

// One CUDA stream per rank, indexed by rank. An object whose memory lives on several ranks takes one
// of these, so each piece of work goes on the stream of the device that owns the memory it touches.
//
// A single stream converts implicitly, which is what a single-rank object wants. Handing one to an
// object that spans several ranks fails on the first rank past 0 rather than silently issuing that
// rank's copies on the wrong device's stream.
class RankStreams {
public:
    RankStreams(cudaStream_t single = nullptr) noexcept : count_(1) { streams_[0] = single; }
    explicit RankStreams(std::span<const cudaStream_t> streams) : count_(streams.size()) {
        if (streams.empty() || streams.size() > kMaxRanks) {
            throw std::invalid_argument("RankStreams needs between one and kMaxRanks streams");
        }
        std::copy(streams.begin(), streams.end(), streams_.begin());
    }
    // Every rank's compute stream, or every rank's transfer stream.
    [[nodiscard]] static RankStreams compute(const DeviceContext& context);
    [[nodiscard]] static RankStreams transfer(const DeviceContext& context);

    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] cudaStream_t operator[](std::size_t rank) const {
        if (rank >= count_) {
            throw std::out_of_range("no stream was given for rank " + std::to_string(rank) +
                                    " (" + std::to_string(count_) + " given)");
        }
        return streams_[rank];
    }

private:
    std::array<cudaStream_t, kMaxRanks> streams_{};
    std::size_t count_ = 0;
};

inline RankStreams RankStreams::compute(const DeviceContext& context) {
    std::array<cudaStream_t, kMaxRanks> streams{};
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        streams[rank] = context.rank(rank).stream;
    }
    return RankStreams(std::span<const cudaStream_t>(streams.data(), context.size()));
}

inline RankStreams RankStreams::transfer(const DeviceContext& context) {
    std::array<cudaStream_t, kMaxRanks> streams{};
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        streams[rank] = context.rank(rank).transfer_stream;
    }
    return RankStreams(std::span<const cudaStream_t>(streams.data(), context.size()));
}

// Binds a rank for the duration of a scope and restores the previous one, so a caller that has to
// touch the secondary device cannot leave the thread bound to it.
class ScopedDeviceRank {
public:
    ScopedDeviceRank(DeviceContext& context, std::size_t rank);
    ~ScopedDeviceRank() noexcept;

    ScopedDeviceRank(const ScopedDeviceRank&)            = delete;
    ScopedDeviceRank& operator=(const ScopedDeviceRank&) = delete;

private:
    DeviceContext& context_;
    std::size_t previous_rank_ = 0;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    explicit CudaCompletionEvent(const RankContext& rank);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    // Capture a record node observable by the host while later graph nodes are still running.
    // Only call during stream capture; this event must outlive every graph referencing it.
    void record_external(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

// One completion event per rank, for work that fans out across ranks and has to be fenced on all of
// them: a context transaction copies state on every rank's transfer stream, and is complete only
// when the last of them is. A single stream converts implicitly, so a one-rank set behaves as one
// event; a set of several ranks given one stream fails on the first rank past 0.
class RankFenceSet {
public:
    explicit RankFenceSet(const DeviceContext& context);

    RankFenceSet(const RankFenceSet&)            = delete;
    RankFenceSet& operator=(const RankFenceSet&) = delete;

    // Records rank r's event on its stream.
    void record(RankStreams streams);
    // Makes rank r's stream wait for rank r's event.
    void wait(RankStreams streams) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    std::vector<CudaCompletionEvent> events_;
};

} // namespace ninfer
