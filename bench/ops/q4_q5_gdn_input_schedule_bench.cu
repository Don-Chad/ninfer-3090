// Maintainer benchmark for the Q4/Q5 GDN input-projection route boundaries.
//
// This Op was the last route table here with no schedule bench, and it is the one that needed one
// most. Its {{1, 6}} / {{7, 32}} boundary is where DFlash2 loses 15% between five and six draft
// tokens (verification width is k+1, so k=6 is the first count to cross it) and where a C8 decode
// cohort starts paying a 32-wide MMA tile for eight live columns -- 13.8 ms of a 56.3 ms round.
// Both were diagnosed by profiling whole runs, because nothing could time the two schedules
// against each other at the same width. This is that.
//
// Like its siblings it calls q4_q5_gdn_input_execute_schedule, the real dispatch minus its
// plan-matches-problem check, so every schedule runs at every in-domain width regardless of which
// one the table would pick. Nothing in the engine consumes this.
//
// Two things to know before reading a column:
//
//   * IndependentDirectFixed is bounded at 15, not 6. launch_q4 and launch_q5 each carry a
//     dedicated R8C8 route for 5..15, which makes widening the route band look free. It is not:
//     measured end to end it costs 3.6% at width 7 and 23.8% at width 9. That the direct path is
//     *defined* to 15 and *loses* from 7 is exactly the thing this bench makes visible.
//
//   * The grouped tiles cost what their padded width costs, not what the live token count costs,
//     so their columns should read nearly flat across this sweep. A grouped tile winning at a
//     width far below its own is not a paradox -- it means the MMA path beats the SIMT direct path
//     by more than the padding wastes.
//
//   build-ninja\bench\ninfer_q4_q5_gdn_input_schedule_bench.exe --spread
//
// Use --spread when a margin looks close: it prints min..p95 beside the median, and a boundary is
// only decidable if the margin clears the spread. See TODO section 3 on cold-flush margins -- they
// overstate wins, so confirm anything narrow with a profile of the real workload.

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;

using Id = ninfer::ops::detail::Q4Q5GdnInputScheduleId;

struct Schedule {
    const char* name;
    Id id;
    // Largest column count the kernel is defined for; 0 means unbounded. The independent route
    // throws above 15 rather than misbehaving, so the domain is advisory here -- but printing
    // "out-of-domain" is more honest than printing a caught exception as "n/a".
    std::int32_t max_cols;
};

const Schedule kSchedules[] = {
    {"independent_direct", Id::IndependentDirectFixed, 15},
    {"small_t_mma", Id::SmallTMma, 32},
    {"grouped_r64_c8", Id::GroupedMixedMmaR64C8, 0},
    {"grouped_r64_c16", Id::GroupedMixedMmaR64C16, 0},
    {"grouped_r64_c32", Id::GroupedMixedMmaR64C32, 0},
    {"grouped_r64_c64", Id::GroupedMixedMmaR64C64, 0},
    {"grouped_r64_c128", Id::GroupedMixedMmaR64C128, 0},
};
constexpr int kScheduleCount = static_cast<int>(sizeof(kSchedules) / sizeof(kSchedules[0]));

// The one shape q4_q5_gdn_input_admits: hidden 5120, qk 4096, value+z 12288, qkv 10240, z 6144.
constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueZRows = 12288;
constexpr std::int32_t kQkvRows    = 10240;
constexpr std::int32_t kZRows      = 6144;
constexpr std::size_t kFlushBytes  = 256ULL << 20;

} // namespace

int main(int argc, char** argv) {
    std::vector<std::int32_t> tokens;
    int repeat  = 9;
    int warmup  = 3;
    bool spread = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--tokens" && i + 1 < argc) {
            std::string value(argv[++i]);
            std::size_t start = 0;
            while (start <= value.size()) {
                const std::size_t comma = value.find(',', start);
                const std::string item  = value.substr(start, comma - start);
                if (!item.empty()) { tokens.push_back(std::atoi(item.c_str())); }
                if (comma == std::string::npos) { break; }
                start = comma + 1;
            }
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--spread") {
            spread = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--tokens T,...] [--repeat N] [--warmup N] [--spread]\n",
                         argv[0]);
            return 2;
        }
    }
    // Dense through the contested band, then the widths the other route bands anchor. 1..16 one at
    // a time because the whole question is where a boundary belongs, and 6/7 and 8/9 are both live.
    if (tokens.empty()) {
        tokens = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                  14, 15, 16, 20, 24, 32, 33, 48, 64, 65, 96, 128};
    }
    const std::int32_t max_tokens = *std::max_element(tokens.begin(), tokens.end());

    ninfer::bench::PackedQuantizedWeight qk = ninfer::bench::make_row_split_weight(
        QType::Q4_G64_FP16, kQkRows, kHidden, kHidden, 0x51U);
    ninfer::bench::PackedQuantizedWeight vz = ninfer::bench::make_row_split_weight(
        QType::Q5_G64_FP16, kValueZRows, kHidden, kHidden, 0x52U);

    ninfer::DeviceBuffer input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    ninfer::DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * max_tokens * 2);
    ninfer::DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    ninfer::bench::L2FlushBuffer flush(kFlushBytes);
    cudaStream_t stream = nullptr;

    const int width = spread ? 30 : 22;
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("# gpu=%s sm=%d%d  q4_q5 GDN input schedules, cold, %s of %d\n", properties.name,
                properties.major, properties.minor, spread ? "median min..p95" : "median", repeat);
    std::printf("%6s", "T");
    for (const Schedule& schedule : kSchedules) { std::printf(" %*s", width, schedule.name); }
    std::printf(" %-46s   %-20s\n", "routed_to", "winner");

    for (const std::int32_t token_count : tokens) {
        Tensor x(input.p, DType::BF16, {kHidden, token_count});
        Tensor tqkv(qkv.p, DType::BF16, {kQkvRows, token_count});
        Tensor tz(z.p, DType::BF16, {kZRows, token_count});

        double best_us        = 0.0;
        const char* best_name = "-";
        std::printf("%6d", token_count);
        for (int s = 0; s < kScheduleCount; ++s) {
            const Schedule& schedule = kSchedules[s];
            if (schedule.max_cols != 0 && token_count > schedule.max_cols) {
                std::printf(" %*s", width, "out-of-domain");
                continue;
            }
            const auto invoke = [&](cudaStream_t launch_stream) {
                ninfer::ops::detail::q4_q5_gdn_input_execute_schedule(
                    schedule.id, x, qk.weight, vz.weight, tqkv, tz, launch_stream);
            };
            ninfer::bench::ColdTiming timing{};
            try {
                timing = ninfer::bench::measure_cold_launch(invoke, flush, stream, warmup, repeat);
            } catch (const std::exception&) {
                cudaGetLastError();
                std::printf(" %*s", width, "n/a");
                continue;
            }
            const double us = timing.median_us;
            if (spread) {
                char cell[64];
                std::snprintf(cell, sizeof(cell), "%.1f %.1f..%.1f", us, timing.min_us,
                              timing.p95_us);
                std::printf(" %*s", width, cell);
            } else {
                std::printf(" %*.3f", width, us);
            }
            if (best_us == 0.0 || us < best_us) {
                best_us   = us;
                best_name = schedule.name;
            }
        }
        const char* routed = "?";
        try {
            const ninfer::ops::detail::Q4Q5GdnInputProblem problem{
                kHidden, kQkRows, kValueZRows, kQkvRows, kZRows, kHidden, token_count};
            routed = ninfer::ops::detail::q4_q5_gdn_input_schedule_name(
                ninfer::ops::detail::q4_q5_gdn_input_resolve_plan(problem).schedule);
        } catch (const std::exception&) { cudaGetLastError(); }
        std::printf(" %-46s   %-20s\n", routed, best_name);
        std::fflush(stdout);
    }
    return 0;
}
