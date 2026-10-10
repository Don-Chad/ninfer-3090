#pragma once
//
// ninfer_bench_common.h — shared bench harness for L1 op performance binaries.
//
// Adapted from ~/chunked_gdn/bench/bench_common.h. Timing uses CUDA events
// with inner-iter batching to amortize the per-sample host sync; throughput is
// reported from the MEDIAN per-launch time (robust to host scheduling spikes).
//
// IMPORTANT (docs/maintainer/op-development.md §9): the GB/s printed here is a convenience
// readout, not a universal acceptance gate. Interpret it with cache conditions,
// a same-topology payload control, and only the profiler evidence needed for the
// concrete kernel question.

#include "../common/fixture_data.cuh"
#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::bench {

// ---- Device-derived hardware ceilings ---------------------------------------------------------
//
// This harness was written against upstream's RTX 5090 target and carried its ceilings as
// literals: 1792 GB/s DRAM, 1674.5 GB/s sustained read, and the GB202 FP8 tensor peaks. On the
// sm_86 fork target those overstate DRAM bandwidth by 1.91x, so every DRAM_%, READ_%, mem_% and
// TC_% column printed by a bench binary is wrong by that factor on the hardware this fork exists
// to serve. Derive the DRAM spec from the device itself and keep measured per-architecture
// ceilings beside it.
//
// The sm_86 figures are measured on an RTX 3090 (82 SM, 1695 MHz boost, 384-bit GDDR6X):
// sustained read from tools/hbm_bandwidth_probe.cu, tensor peaks from a register-resident MMA
// issue-rate probe. Measured integer rates run about 18% above the GA102 whitepaper's dense
// figures, which are quoted at a lower clock; the float rates match it.
struct DeviceSpecs {
    double dram_spec_gbs      = 0.0;
    double sustained_read_gbs = 0.0;
    double bf16_f32acc_tflops = 0.0; // BF16 MMA, f32 accumulate — what the A16 routes issue.
    double int8_tops          = 0.0; // s8 MMA, s32 accumulate — what mma_s8 issues.
    double fp8_f16acc_tflops  = 0.0; // NaN where the architecture has no FP8 tensor path.
    double fp8_f32acc_tflops  = 0.0;
    const char* reference     = "unknown";
};

inline const DeviceSpecs& device_specs() {
    static const DeviceSpecs specs = [] {
        DeviceSpecs out;
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return out; }
        // cudaDeviceProp lost its memory-clock fields in CUDA 13; the attribute queries survive
        // both toolkits, so ask for them that way rather than through the struct.
        int clock_khz = 0;
        int bus_bits  = 0;
        int major     = 0;
        int minor     = 0;
        if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device) !=
                cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
                cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) !=
                cudaSuccess) {
            return out;
        }
        // The reported memory clock is the single-data-rate figure; GDDR transfers on both edges.
        out.dram_spec_gbs =
            2.0 * static_cast<double>(clock_khz) * 1.0e3 * (static_cast<double>(bus_bits) / 8.0) /
            1.0e9;
        const int arch = major * 10 + minor;
        if (arch == 86) {
            out.reference          = "RTX_3090";
            out.sustained_read_gbs = 894.5;
            out.bf16_f32acc_tflops = 70.7;
            out.int8_tops          = 335.3;
            out.fp8_f16acc_tflops  = std::numeric_limits<double>::quiet_NaN();
            out.fp8_f32acc_tflops  = std::numeric_limits<double>::quiet_NaN();
        } else if (arch >= 120) {
            out.reference          = "RTX_5090";
            out.sustained_read_gbs = 1674.5;
            out.bf16_f32acc_tflops = std::numeric_limits<double>::quiet_NaN();
            out.int8_tops          = std::numeric_limits<double>::quiet_NaN();
            out.fp8_f16acc_tflops  = 838.0;
            out.fp8_f32acc_tflops  = 419.0;
        } else {
            // Unmeasured architecture: the DRAM spec is still exact, the rest is not claimed.
            // 0.93 is the sustained-read fraction both measured architectures land on.
            out.reference          = "device-derived";
            out.sustained_read_gbs = out.dram_spec_gbs * 0.93;
            out.bf16_f32acc_tflops = std::numeric_limits<double>::quiet_NaN();
            out.int8_tops          = std::numeric_limits<double>::quiet_NaN();
            out.fp8_f16acc_tflops  = std::numeric_limits<double>::quiet_NaN();
            out.fp8_f32acc_tflops  = std::numeric_limits<double>::quiet_NaN();
        }
        return out;
    }();
    return specs;
}

__host__ __device__ inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return std::uint16_t(u >> 16);
}

// Every operand supplies a stable seed; allocation order does not select the data.
inline DeviceBuffer make_bf16(std::size_t n, std::uint64_t seed, float low = -0.5F,
                              float high = 0.5F) {
    DeviceBuffer result(n * sizeof(__nv_bfloat16));
    CUDA_CHECK(fixture::fill_values(static_cast<__nv_bfloat16*>(result.p), n, seed, low, high));
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

inline DeviceBuffer make_f32(std::size_t n, std::uint64_t seed, float low, float high) {
    DeviceBuffer result(n * sizeof(float));
    CUDA_CHECK(fixture::fill_values(static_cast<float*>(result.p), n, seed, low, high));
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

// Snapshot only mutable operands. Restore is setup, never part of a timed Op.
class SavedBuffer {
public:
    explicit SavedBuffer(DeviceBuffer& buffer) : destination_(buffer.p), initial_(buffer.bytes) {
        CUDA_CHECK(cudaMemcpy(initial_.p, buffer.p, buffer.bytes, cudaMemcpyDeviceToDevice));
    }

    void restore(cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(destination_, initial_.p, initial_.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
private:
    void* destination_;
    DeviceBuffer initial_;
};

inline DeviceBuffer make_zeros(std::size_t bytes) {
    DeviceBuffer d(bytes);
    d.fill();
    return d;
}

// The in-process GB/s stays informational (ncu is the acceptance gate), but it now reports
// against this device's own DRAM spec instead of a hardcoded RTX 5090 roofline.
inline double device_peak_bw_gbs(int /*dev*/ = 0) { return device_specs().dram_spec_gbs; }

struct ColdTiming {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

class TimedGraph {
public:
    TimedGraph() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~TimedGraph() {
        if (exec_ != nullptr) { cudaGraphExecDestroy(exec_); }
        if (graph_ != nullptr) { cudaGraphDestroy(graph_); }
        if (start_ != nullptr) { cudaEventDestroy(start_); }
        if (stop_ != nullptr) { cudaEventDestroy(stop_); }
    }

    TimedGraph(const TimedGraph&)            = delete;
    TimedGraph& operator=(const TimedGraph&) = delete;

    template <class Body>
    void capture(cudaStream_t stream, Body&& body) {
        if (graph_ != nullptr || exec_ != nullptr) {
            throw std::logic_error("benchmark graph is already captured");
        }
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            body(stream);
        } catch (...) {
            cudaGraph_t discard = nullptr;
            cudaStreamEndCapture(stream, &discard);
            if (discard != nullptr) { cudaGraphDestroy(discard); }
            throw;
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes_));
        if (nodes_ == 0) { throw std::runtime_error("captured benchmark graph is empty"); }
    }

    void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(exec_, stream)); }

    double launch_timed(cudaStream_t stream) const {
        CUDA_CHECK(cudaEventRecord(start_, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) * 1000.0;
    }

    [[nodiscard]] std::size_t nodes() const noexcept { return nodes_; }

private:
    cudaGraph_t graph_    = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaEvent_t start_    = nullptr;
    cudaEvent_t stop_     = nullptr;
    std::size_t nodes_    = 0;
};

inline ColdTiming summarize_timings(std::vector<double> samples) {
    if (samples.empty()) { throw std::invalid_argument("cannot summarize empty timings"); }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double fraction) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    return {percentile(0.50), samples.front(), percentile(0.95)};
}

template <class Launch>
ColdTiming measure_launch(Launch&& launch, cudaStream_t stream, int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument("benchmark requires nonnegative warmup and positive repeat");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int index = 0; index < warmup; ++index) { launch(stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return summarize_timings(std::move(samples));
}

inline ColdTiming measure_graph(const TimedGraph& graph, cudaStream_t stream, int warmup,
                                int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument("benchmark requires nonnegative warmup and positive repeat");
    }
    for (int index = 0; index < warmup; ++index) { graph.launch(stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) { samples.push_back(graph.launch_timed(stream)); }
    return summarize_timings(std::move(samples));
}

namespace detail {
inline constexpr int kEvictionBlocks  = 1024;
inline constexpr int kEvictionThreads = 256;

static __global__ void read_eviction_buffer(const uint4* input, std::size_t vectors,
                                            uint4* checksums) {
    uint4 value{};
    for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 loaded = input[i];
        value.x ^= loaded.x;
        value.y ^= loaded.y;
        value.z ^= loaded.z;
        value.w ^= loaded.w;
    }
    for (int offset = 16; offset; offset >>= 1) {
        value.x ^= __shfl_xor_sync(0xffffffffU, value.x, offset);
        value.y ^= __shfl_xor_sync(0xffffffffU, value.y, offset);
        value.z ^= __shfl_xor_sync(0xffffffffU, value.z, offset);
        value.w ^= __shfl_xor_sync(0xffffffffU, value.w, offset);
    }
    __shared__ uint4 warps[kEvictionThreads / 32];
    if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint4 total{};
        for (const auto& partial : warps) {
            total.x ^= partial.x;
            total.y ^= partial.y;
            total.z ^= partial.z;
            total.w ^= partial.w;
        }
        checksums[blockIdx.x] = total;
    }
}
} // namespace detail

// Reading a preinitialized buffer evicts cache lines without leaving an L2-sized dirty
// working set whose writeback would become part of the next measured Op's traffic.
class L2FlushBuffer : public DeviceBuffer {
public:
    explicit L2FlushBuffer(std::size_t bytes)
        : DeviceBuffer(bytes), checksums_(detail::kEvictionBlocks * sizeof(uint4)) {
        CUDA_CHECK(fixture::fill_bytes(p, bytes, 0x4c324576696374ULL));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void evict(cudaStream_t stream) {
        detail::
            read_eviction_buffer<<<detail::kEvictionBlocks, detail::kEvictionThreads, 0, stream>>>(
                static_cast<const uint4*>(p), bytes / sizeof(uint4),
                static_cast<uint4*>(checksums_.p));
        CUDA_CHECK(cudaGetLastError());
    }

private:
    DeviceBuffer checksums_;
};

inline void flush_l2(L2FlushBuffer& flush, cudaStream_t stream) { flush.evict(stream); }

template <class Launch>
ColdTiming measure_cold_launch(Launch&& launch, L2FlushBuffer& flush, cudaStream_t stream,
                               int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument(
            "cold benchmark requires nonnegative warmup and positive repeat");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int index = 0; index < warmup; ++index) {
        flush_l2(flush, stream);
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        flush_l2(flush, stream);
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return summarize_timings(std::move(samples));
}

inline ColdTiming measure_cold_graph(const TimedGraph& graph, L2FlushBuffer& flush,
                                     cudaStream_t stream, int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument(
            "cold benchmark requires nonnegative warmup and positive repeat");
    }
    for (int index = 0; index < warmup; ++index) {
        flush_l2(flush, stream);
        graph.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        flush_l2(flush, stream);
        samples.push_back(graph.launch_timed(stream));
    }
    return summarize_timings(std::move(samples));
}

template <class Prepare, class Launch>
ColdTiming measure_launch_prepared(Prepare&& prepare, Launch&& launch, cudaStream_t stream,
                                   int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) throw std::invalid_argument("invalid benchmark repetitions");
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) {
        prepare(stream);
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> samples;
    samples.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        prepare(stream);
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(double(ms) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return summarize_timings(std::move(samples));
}

template <class Prepare>
ColdTiming measure_graph_prepared(Prepare&& prepare, const TimedGraph& graph, cudaStream_t stream,
                                  int warmup, int repeat) {
    return measure_launch_prepared(
        std::forward<Prepare>(prepare), [&](cudaStream_t s) { graph.launch(s); }, stream, warmup,
        repeat);
}

// Cold measurement for an Op that updates its own input in place. `prepare` restores that input to
// the same initial value before every warmup launch and every sample, so each timed call sees the
// same operand; prepare and the L2 flush both run outside the timed interval.
template <class Prepare, class Launch>
ColdTiming measure_cold_launch_prepared(Prepare&& prepare, Launch&& launch, L2FlushBuffer& flush,
                                        cudaStream_t stream, int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument(
            "cold benchmark requires nonnegative warmup and positive repeat");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int index = 0; index < warmup; ++index) {
        prepare(stream);
        flush_l2(flush, stream);
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        prepare(stream);
        flush_l2(flush, stream);
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return summarize_timings(std::move(samples));
}

template <class Prepare>
ColdTiming measure_cold_graph_prepared(Prepare&& prepare, const TimedGraph& graph,
                                       L2FlushBuffer& flush, cudaStream_t stream, int warmup,
                                       int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument(
            "cold benchmark requires nonnegative warmup and positive repeat");
    }
    for (int index = 0; index < warmup; ++index) {
        prepare(stream);
        flush_l2(flush, stream);
        graph.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        prepare(stream);
        flush_l2(flush, stream);
        samples.push_back(graph.launch_timed(stream));
    }
    return summarize_timings(std::move(samples));
}

struct Result {
    int n_runs       = 0;
    int inner_iters  = 1;
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
    double mean_us   = 0.0;
    double gbs       = 0.0; // from median
};

using launch_fn = std::function<void(cudaStream_t)>;

inline Result bench_loop(const launch_fn& launch, double bytes_moved, int warmup = 20,
                         int repeat = 100, int min_time_ms = 500) {
    cudaStream_t stream = nullptr;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    for (int i = 0; i < warmup; ++i) launch(stream);
    cudaStreamSynchronize(stream);

    // Auto-size inner_iters so each timed batch is ~500us (amortize sync wait).
    int inner = 0;
    {
        constexpr int probe = 4;
        cudaEventRecord(a, stream);
        for (int i = 0; i < probe; ++i) launch(stream);
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        const double per_us = double(ms) * 1000.0 / probe;
        inner = std::max(1, std::min(1024, int(std::ceil(500.0 / std::max(per_us, 1.0)))));
    }

    std::vector<double> samples;
    long long total_us = 0;
    while (int(samples.size()) < repeat || total_us < (long long)min_time_ms * 1000) {
        cudaEventRecord(a, stream);
        for (int i = 0; i < inner; ++i) launch(stream);
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        const double batch_us = double(ms) * 1000.0;
        samples.push_back(batch_us / inner);
        total_us += (long long)batch_us;
        if (samples.size() > 100000) break;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) {
        const std::size_t idx = std::min(sorted.size() - 1, std::size_t(q * sorted.size()));
        return sorted[idx];
    };
    double sum = 0.0;
    for (double v : samples) sum += v;

    Result r;
    r.n_runs         = int(samples.size());
    r.inner_iters    = inner;
    r.median_us      = pct(0.50);
    r.min_us         = sorted.front();
    r.p95_us         = pct(0.95);
    r.mean_us        = sum / samples.size();
    const double sec = r.median_us * 1e-6;
    r.gbs            = (sec > 0.0) ? bytes_moved / sec / 1e9 : 0.0;
    return r;
}

// Mutating Ops use one captured call per sample, restored outside the graph and events.
// Repeatedly applying the Op inside an inner loop changes the measured workload.
inline Result bench_loop_prepared(const launch_fn& prepare, const launch_fn& launch,
                                  double bytes_moved, int warmup = 20, int repeat = 100) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TimedGraph graph;
    graph.capture(stream, launch);
    const auto timing = measure_graph_prepared(prepare, graph, stream, warmup, repeat);
    CUDA_CHECK(cudaStreamDestroy(stream));
    Result result;
    result.n_runs    = repeat;
    result.median_us = timing.median_us;
    result.min_us    = timing.min_us;
    result.p95_us    = timing.p95_us;
    result.gbs       = bytes_moved / timing.median_us / 1.0e3;
    return result;
}

inline void print_result(const char* tag, const Result& r) {
    std::printf(
        "%-32s median=%8.2f us  min=%8.2f us  p95=%8.2f us  %8.1f GB/s  (%.1f%% of %.0f GB/s "
        "roofline)\n",
        tag, r.median_us, r.min_us, r.p95_us, r.gbs,
        r.gbs / device_specs().dram_spec_gbs * 100.0, device_specs().dram_spec_gbs);
}

} // namespace ninfer::bench
