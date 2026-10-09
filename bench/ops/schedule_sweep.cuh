#pragma once

// Shared driver for route-boundary sweeps.
//
// The public Op benches answer "how fast is the route the plan picked". Choosing *where* a boundary
// belongs needs the opposite view: every schedule timed at the same column count, including the
// ones the current table would never select. That is what this drives.
//
// It exists as a shared header because the alternative -- a bespoke ~150-line bench per Op -- is
// how the boundaries drifted in the first place. After a catch-up merge, retuning an Op should
// cost a 40-line file, not an afternoon, or it will not happen and upstream's numbers will stay.
//
// Two traps are handled here rather than left to each caller, because both produced confidently
// wrong answers before being caught:
//
//   * Kernels do not police their column domains. q4_q5 parent_split_fixed is defined for
//     cols <= 12 and corrupts memory past it (a naive sweep took the whole process down at T=16);
//     q4 swiglu gemv_pair is registered for a single column and, handed more, silently does one
//     column's work at a constant cost and appears to win at every width. Every entry therefore
//     declares max_cols and is skipped outside it.
//   * Warm timings pick different winners than cold ones. These projections stream tens of MB of
//     weights per call against 6 MB of L2, so production is always cold. measure_cold_launch
//     flushes L2 between reps; a bare event loop does not.
//
// Passing public_op alongside the schedules is worth the extra column: for every route except a
// broken one, the public Op's time equals the chosen kernel's time. A gap means dispatch is doing
// something the kernel is not -- that is how the Q4/Q5 switch fallthrough was found, where
// selecting one schedule silently ran a second kernel over the top of it.

#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::bench {

// invoke(tokens, stream) launches the schedule for that column count. max_cols is the largest
// column count the kernel is defined for; 0 means unbounded.
struct SweepEntry {
    const char* name;
    std::function<void(std::int32_t, cudaStream_t)> invoke;
    std::int32_t max_cols = 0;
};

struct SweepOptions {
    std::vector<std::int32_t> tokens;
    int repeat                = 9;
    int warmup                = 3;
    std::size_t flush_bytes   = 256ULL << 20;
    const char* title         = "schedules";
    // Print "median min..p95" per cell instead of the median alone. A route decision is only
    // meaningful if the margin between two schedules clears their own spread, and the median by
    // itself cannot tell you that -- TODO section 5 has a band that stayed unresolved for exactly
    // this reason, where the losing schedule's own samples ranged 19.6%.
    bool spread               = false;
    // Optional: what the Op's own resolver picks at this width, and the public Op itself. Supply
    // both or neither.
    std::function<const char*(std::int32_t)> routed_name;
    std::function<void(std::int32_t, cudaStream_t)> public_op;
};

// Parses "--tokens A,B,C", "--repeat N", "--warmup N", "--spread". Returns false on an
// unrecognised argument so the caller can print its own usage.
inline bool parse_sweep_args(int argc, char** argv, SweepOptions& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--tokens" && i + 1 < argc) {
            options.tokens.clear();
            const std::string value(argv[++i]);
            std::size_t start = 0;
            while (start <= value.size()) {
                const std::size_t comma = value.find(',', start);
                const std::string item  = value.substr(start, comma - start);
                if (!item.empty()) { options.tokens.push_back(std::atoi(item.c_str())); }
                if (comma == std::string::npos) { break; }
                start = comma + 1;
            }
        } else if (arg == "--repeat" && i + 1 < argc) {
            options.repeat = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = std::atoi(argv[++i]);
        } else if (arg == "--spread") {
            options.spread = true;
        } else {
            return false;
        }
    }
    return !options.tokens.empty();
}

inline void run_sweep(const SweepOptions& options, const std::vector<SweepEntry>& schedules) {
    L2FlushBuffer flush(options.flush_bytes);
    cudaStream_t stream = nullptr;

    const int width = options.spread ? 30 : 22;
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("# gpu=%s sm=%d%d  %s, cold, %s of %d\n", properties.name, properties.major,
                properties.minor, options.title,
                options.spread ? "median min..p95" : "median", options.repeat);
    std::printf("%6s", "T");
    for (const SweepEntry& entry : schedules) { std::printf(" %*s", width, entry.name); }
    if (options.public_op) { std::printf(" %12s %-34s", "public_op", "routed_to"); }
    std::printf("   %-22s\n", "winner");

    for (const std::int32_t tokens : options.tokens) {
        double best_us        = 0.0;
        const char* best_name = "-";
        std::printf("%6d", tokens);
        for (const SweepEntry& entry : schedules) {
            if (entry.max_cols != 0 && tokens > entry.max_cols) {
                std::printf(" %*s", width, "out-of-domain");
                continue;
            }
            const auto launch = [&](cudaStream_t launch_stream) {
                entry.invoke(tokens, launch_stream);
            };
            ColdTiming timing{};
            try {
                timing = measure_cold_launch(launch, flush, stream, options.warmup,
                                             options.repeat);
            } catch (const std::exception&) {
                cudaGetLastError();
                std::printf(" %*s", width, "n/a");
                continue;
            }
            const double us = timing.median_us;
            if (options.spread) {
                char cell[64];
                std::snprintf(cell, sizeof(cell), "%.1f %.1f..%.1f", us, timing.min_us,
                              timing.p95_us);
                std::printf(" %*s", width, cell);
            } else {
                std::printf(" %*.3f", width, us);
            }
            if (best_us == 0.0 || us < best_us) {
                best_us   = us;
                best_name = entry.name;
            }
        }
        if (options.public_op) {
            double public_us   = 0.0;
            const char* routed = "?";
            try {
                if (options.routed_name) { routed = options.routed_name(tokens); }
                const auto launch = [&](cudaStream_t launch_stream) {
                    options.public_op(tokens, launch_stream);
                };
                public_us =
                    measure_cold_launch(launch, flush, stream, options.warmup, options.repeat)
                        .median_us;
            } catch (const std::exception&) { cudaGetLastError(); }
            std::printf(" %12.3f %-34s", public_us, routed);
        }
        std::printf("   %-22s\n", best_name);
        std::fflush(stdout);
    }
}

} // namespace ninfer::bench
