// Maintainer benchmark for the Q4 SwiGLU route boundaries, companion to
// q4_q5_attn_input_schedule_bench.cu and written for the same reason: the public bench times the
// route the plan picks, which cannot tell you whether the boundary between two routes is in the
// right place. This one times each schedule at the same column count.
//
// The immediate question it answers: this fork's route table sends 2..32 to SmallTTiled on
// upstream's authority. The boundary it replaced (24) was measured against
// q4_linear_swiglu_small_t_exact, a kernel upstream deleted, so it had nothing behind it -- but
// neither did 32 on sm_86.
//
// Materialized is included as of the sm_86 sweep of the {49,128}/{257,384}/{513,640} bands. It is
// not a kernel -- it is linear() into a workspace followed by silu_mul() over the two halves -- so
// it has no launch signature to put in a function-pointer table, which is why it was left out
// before. Every schedule now goes through q4_linear_swiglu_execute_schedule instead, the real
// dispatch minus its plan-matches-problem check, so the bench times the Op rather than a replica.

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
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
using ninfer::Weight;

using Id = ninfer::ops::detail::Q4LinearSwiGluScheduleId;

struct Schedule {
    const char* name;
    Id id;
    // Largest column count the kernel actually processes; 0 means unbounded. gemv_pair is a
    // decode kernel registered for {1,1}: hand it more columns and it silently does one column's
    // work in a constant 122.9 us, which makes it look like it wins everywhere.
    std::int32_t max_cols;
};

const Schedule kSchedules[] = {
    {"gemv_pair", Id::GemvPair, 1},
    {"small_t_tiled", Id::SmallTTiled, 32},
    {"split_half_pair_c40", Id::MmaSplitHalfPairR32C40, 0},
    {"split_half_pair_c48", Id::MmaSplitHalfPairR32C48, 0},
    {"split_half_pair_c128", Id::MmaSplitHalfPairR32C128, 0},
    {"materialized", Id::Materialized, 0},
    // Exploratory: int8 tensor-core small-T, T=32 only (see q4_small_t_mma_i8.cuh). Not in
    // resolve_plan's route table; this row exists purely to compare it against small_t_tiled at the
    // one width it currently supports.
    {"small_t_tiled_i8", Id::SmallTTiledI8, 32},
    // Prices the runtime column count against small_t_tiled, which now drops it at exact widths.
    {"small_t_masked", Id::SmallTTiledMasked, 32},
};
constexpr int kScheduleCount = static_cast<int>(sizeof(kSchedules) / sizeof(kSchedules[0]));

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = 17408;
constexpr std::int32_t kHidden     = 5120;
constexpr std::size_t kFlushBytes  = 256ULL << 20;

} // namespace

int main(int argc, char** argv) {
    std::vector<std::int32_t> tokens;
    int repeat  = 9;
    int warmup  = 3;
    // A route boundary is only decidable if the margin between two schedules clears their own
    // spread. This Op is the noisiest of the route tables and the {513,640} band went unresolved
    // because of it, so print min..p95 beside the median when asked.
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
            std::fprintf(stderr, "usage: %s [--tokens T,...] [--repeat N] [--warmup N] [--spread]\n", argv[0]);
            return 2;
        }
    }
    if (tokens.empty()) { tokens = {1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64}; }

    const std::int32_t max_tokens = *std::max_element(tokens.begin(), tokens.end());
    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
        QType::Q4_G64_FP16, kGateUpRows, kHidden, kHidden, 0x52U);
    ninfer::DeviceBuffer input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    ninfer::DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_tokens * 2);
    ninfer::bench::L2FlushBuffer flush(kFlushBytes);
    cudaStream_t stream = nullptr;

    // Materialized stages the whole gate+up product, so its workspace scales with the widest
    // column count swept. Size it once for the whole range rather than per point, and directly
    // from max_tokens rather than through resolve_plan's route table: every schedule here runs at
    // every in-domain token count regardless of which one the table would actually pick, and the
    // table often does not route to Materialized anywhere in [1, max_tokens] -- which is the whole
    // point of sweeping it outside its routed interval.
    const std::size_t workspace_bytes =
        ninfer::ops::detail::q4_linear_swiglu_materialized_workspace_bytes(kGateUpRows,
                                                                           max_tokens);
    ninfer::WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 1));

    const int width = spread ? 30 : 22;
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("# gpu=%s sm=%d%d  q4 swiglu schedules, cold, %s of %d\n", properties.name,
                properties.major, properties.minor, spread ? "median min..p95" : "median",
                repeat);
    std::printf("%6s", "T");
    for (const Schedule& schedule : kSchedules) { std::printf(" %*s", width, schedule.name); }
    std::printf("   %-22s\n", "winner");

    for (const std::int32_t token_count : tokens) {
        Tensor x(input.p, DType::BF16, {kHidden, token_count});
        Tensor out(output.p, DType::BF16, {kOutputRows, token_count});
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
                ninfer::ops::detail::q4_linear_swiglu_execute_schedule(
                    schedule.id, x, packed.weight, out, workspace, launch_stream);
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
        std::printf("   %-22s\n", best_name);
        std::fflush(stdout);
    }
    return 0;
}
