// Maintainer benchmark for choosing the Q4/Q5 attention-input route boundaries.
//
// This is deliberately NOT attn_input_proj_bench.cu: that one benchmarks the public Op and
// documents that production dispatch is owned exclusively by attn_input_proj(). Picking the
// boundaries needs the opposite view -- every schedule timed at the same column count, including
// the ones the current table would never select -- so it calls the internal launches directly.
// Nothing in the engine consumes this; it exists to produce the numbers behind kRoutes.

#include "ninfer/ops/attn_input_proj.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;

using Launch = void (*)(const Tensor&, const Weight&, const Weight&, Tensor&, Tensor&, Tensor&,
                        Tensor&, cudaStream_t);

struct Schedule {
    const char* name;
    Launch launch;
    // Largest column count the kernel is defined for; 0 means unbounded. parent_split_fixed
    // handles cols <= 12 only (q4_q5_attn_input_small_t.cu) and corrupts memory past that, so a
    // sweep that ignores the domain takes the whole process down with it.
    std::int32_t max_cols;
};

// The six distinct kernels behind the seven schedule ids. GroupedHomogeneousPairMmaR32C64S4 and
// PairR32C64S4 both dispatch to grouped_mma_r32_c64_s4, so it appears once.
const Schedule kSchedules[] = {
    {"parent_split_fixed", &ninfer::ops::detail::q4_q5_attn_input_small_t_launch, 12},
    {"small_t_mma", &ninfer::ops::detail::q4_q5_attn_input_small_t_mma_launch, 32},
    {"grouped_r32_c32_s4", &ninfer::ops::detail::q4_q5_attn_input_grouped_mma_r32_c32_s4_launch, 0},
    {"grouped_r32_c64_s4", &ninfer::ops::detail::q4_q5_attn_input_grouped_mma_r32_c64_s4_launch, 0},
    {"mixed_r32_c64_s3", &ninfer::ops::detail::q4_q5_attn_input_mixed_r32_c64_s3_launch, 0},
    {"pair_r32_c64_s3", &ninfer::ops::detail::q4_q5_attn_input_pair_r32_c64_s3_launch, 0},
    {"mixed_r64_c128_s2", &ninfer::ops::detail::q4_q5_attn_input_mixed_r64_c128_s2_launch, 0},
};
constexpr int kScheduleCount = static_cast<int>(sizeof(kSchedules) / sizeof(kSchedules[0]));

constexpr std::size_t kFlushBytes = 128u << 20;

} // namespace

int main(int argc, char** argv) {
    std::vector<std::int32_t> tokens;
    int repeat  = 9;
    int warmup  = 3;
    // A route boundary is only decidable if the margin clears the two schedules' own spread, and
    // ColdTiming carries min and p95 already. See TODO section 3 on cold-flush margins.
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
    if (tokens.empty()) {
        tokens = {1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 104, 112, 128, 144, 160, 192, 256};
    }

    constexpr std::int32_t hidden      = 5120;
    constexpr std::int32_t q_rows      = 6144;
    constexpr std::int32_t kv_rows     = 1024;
    constexpr std::int32_t parent_rows = q_rows + kv_rows;
    const std::int32_t max_tokens      = *std::max_element(tokens.begin(), tokens.end());

    ninfer::bench::PackedQuantizedWeight qk = ninfer::bench::make_row_split_weight(
        QType::Q4_G64_FP16, parent_rows, hidden, hidden, 0x51U);
    ninfer::bench::PackedQuantizedWeight gv = ninfer::bench::make_row_split_weight(
        QType::Q5_G64_FP16, parent_rows, hidden, hidden, 0x52U);

    ninfer::DeviceBuffer input(static_cast<std::size_t>(hidden) * max_tokens * 2);
    ninfer::DeviceBuffer q(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    ninfer::DeviceBuffer gate(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    ninfer::DeviceBuffer k(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    ninfer::DeviceBuffer v(static_cast<std::size_t>(kv_rows) * max_tokens * 2);

    ninfer::bench::L2FlushBuffer flush(kFlushBytes);
    cudaStream_t stream = nullptr;

    const int width = spread ? 30 : 20;
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("# gpu=%s sm=%d%d  q4_q5 attention-input schedules, %s of %d\n", properties.name,
                properties.major, properties.minor, spread ? "median min..p95" : "median",
                repeat);
    std::printf("%6s", "T");
    for (const Schedule& schedule : kSchedules) { std::printf(" %*s", width, schedule.name); }
    // public_op is the same measurement through attn_input_proj(); a gap between it and
    // the schedule the router picked is dispatch overhead, not kernel cost.
    std::printf(" %12s %-22s   %-20s\n", "public_op", "routed_to", "winner");

    for (const std::int32_t token_count : tokens) {
        Tensor x(input.p, DType::BF16, {hidden, token_count});
        Tensor tq(q.p, DType::BF16, {q_rows, token_count});
        Tensor tg(gate.p, DType::BF16, {q_rows, token_count});
        Tensor tk(k.p, DType::BF16, {kv_rows, token_count});
        Tensor tv(v.p, DType::BF16, {kv_rows, token_count});

        double best_us       = 0.0;
        const char* best_name = "-";
        std::printf("%6d", token_count);
        for (int s = 0; s < kScheduleCount; ++s) {
            const Schedule& schedule = kSchedules[s];
            if (schedule.max_cols != 0 && token_count > schedule.max_cols) {
                std::printf(" %*s", width, "out-of-domain");
                continue;
            }
            const auto invoke = [&](cudaStream_t launch_stream) {
                schedule.launch(x, qk.weight, gv.weight, tq, tg, tk, tv, launch_stream);
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
        double public_us   = 0.0;
        const char* routed = "?";
        try {
            const ninfer::ops::detail::Q4Q5AttnInputProblem problem{hidden, q_rows, kv_rows, hidden,
                                                                    token_count};
            routed = ninfer::ops::detail::q4_q5_attn_input_schedule_name(
                ninfer::ops::detail::q4_q5_attn_input_resolve_plan(problem).schedule);
            const auto public_invoke = [&](cudaStream_t launch_stream) {
                ninfer::ops::attn_input_proj(x, qk.weight, gv.weight, tq, tg, tk, tv, launch_stream);
            };
            public_us =
                ninfer::bench::measure_cold_launch(public_invoke, flush, stream, warmup, repeat)
                    .median_us;
        } catch (const std::exception&) { cudaGetLastError(); }
        std::printf(" %12.3f %-22s   %-20s\n", public_us, routed, best_name);
        std::fflush(stdout);
    }

    return 0;
}
