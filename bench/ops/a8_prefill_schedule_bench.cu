// Maintainer benchmark for the integer-activation prefill routes on sm_86.
//
// Nothing timed these kernels directly before, which is how they stayed at ~30% of the card's INT8
// tensor-core rate while an end-to-end prefill number moved for other reasons. This times the
// public Op under A16Only and under AllowA8Int at the same column count, so the A8 route's win over
// the A16 route it replaces is the difference between two cells of one row.
//
// It reports TOP/s beside microseconds because that is the number with a known ceiling: 314.8 TOP/s
// measured by tools/tensor_core_rate_probe.cu. Work is 2*N*K*T, the convention the probes and
// TODO.md already use, so these cells are comparable with both.
//
// Cold timing is not optional here. These GEMMs stream tens of MB of weights per call against 6 MB
// of L2, so a warm loop measures a residency production never has.

#include "core/device.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer_bench_common.h"
#include "artifact/permute_row_split.h"
#include "core/weight_view.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <span>
#include <string_view>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::ops::LinearPolicy;

// The measured INT8 m16n8k32 ceiling on this card (tools/tensor_core_rate_probe.cu).
constexpr double kInt8CeilingTops = 314.8;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Profile {
    const char* name;
    QType qtype;
    std::int32_t rows;  // weight n
    std::int32_t cols;  // weight k, which is also the activation row count
    std::int32_t out_rows;
    bool swiglu; // false selects linear_add, whose residual is its own output
};

// gate_up and down are the MLP's registered A8 prefill profiles. The 5120x6144 LinearAdd is the
// attention o_proj / GDN out_proj shape, now also registered for A8: its two cells are the real
// A16-versus-A8 comparison for that route, same as the MLP pair above.
const Profile kProfiles[] = {
    {"mlp/gate_up", QType::Q4_G64_FP16, 34816, 5120, 17408, true},
    {"mlp/down", QType::Q5_G64_FP16, 5120, 17408, 5120, false},
    {"out_proj", QType::Q5_G64_FP16, 5120, 6144, 5120, false},
};

double tops(const Profile& profile, std::int32_t tokens, double microseconds) {
    const double work = 2.0 * profile.rows * profile.cols * tokens;
    return work / (microseconds * 1.0e-6) / 1.0e12;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::int32_t> tokens;
    int repeat = 9;
    int warmup = 3;
    // Permute the weight into the panel layout and let the routes read it there, so the A8 column
    // measures the production kernel -- epilogue included -- on the layout the probe measured.
    bool panel = false;
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
        } else if (arg == "--panel") {
            panel = true;
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T,...] [--repeat N] [--warmup N]\n", argv[0]);
            return 2;
        }
    }
    if (tokens.empty()) { tokens = {128, 512, 1024}; }

    const std::int32_t max_tokens = *std::max_element(tokens.begin(), tokens.end());
    ninfer::bench::L2FlushBuffer flush(kFlushBytes);
    cudaStream_t stream = nullptr;

    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("# gpu=%s sm=%d%d  integer-activation prefill routes, cold, median of %d\n",
                properties.name, properties.major, properties.minor, repeat);
    std::printf("# work=2*N*K*T, ceiling %.1f TOP/s\n", kInt8CeilingTops);
    std::printf("%-12s %6s %12s %12s %10s %10s %8s\n", "profile", "T", "A16 us", "A8 us",
                "A16 TOP/s", "A8 TOP/s", "% ceil");

    for (const Profile& profile : kProfiles) {
        ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
            profile.qtype, profile.rows, profile.cols, profile.cols, 0x52U);
        if (panel) {
            const std::uint64_t shape[2] = {static_cast<std::uint64_t>(profile.rows),
                                            static_cast<std::uint64_t>(profile.cols)};
            const auto geometry = ninfer::weight_geometry(
                profile.qtype, ninfer::QuantLayout::RowSplitPanel,
                std::span<const std::uint64_t>(shape, 2));
            ninfer::artifact::permute_row_split_to_panel(
                geometry, static_cast<std::byte*>(packed.storage.p), nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            packed.weight.layout = ninfer::QuantLayout::RowSplitPanel;
        }
        ninfer::DeviceBuffer input(static_cast<std::size_t>(profile.cols) * max_tokens * 2);
        ninfer::DeviceBuffer output(static_cast<std::size_t>(profile.out_rows) * max_tokens * 2);

        for (const std::int32_t token_count : tokens) {
            Tensor x(input.p, DType::BF16, {profile.cols, token_count});
            Tensor out(output.p, DType::BF16, {profile.out_rows, token_count});

            double cell_us[2] = {0.0, 0.0};
            const LinearPolicy policies[2] = {LinearPolicy::A16Only, LinearPolicy::AllowA8Int};
            for (int p = 0; p < 2; ++p) {
                const LinearPolicy policy = policies[p];
                std::size_t capacity      = 0;
                try {
                    capacity = profile.swiglu
                                   ? ninfer::ops::linear_swiglu_workspace_capacity_bytes(
                                         profile.qtype, profile.rows, profile.cols, policy,
                                         token_count, token_count)
                                   : ninfer::ops::linear_add_workspace_capacity_bytes(
                                         profile.qtype, profile.out_rows, profile.cols, policy,
                                         token_count, token_count);
                } catch (const std::exception&) {
                    continue; // the policy is not admitted for this profile
                }
                ninfer::WorkspaceArena workspace(std::max<std::size_t>(capacity, 1));
                const auto invoke = [&](cudaStream_t launch_stream) {
                    if (profile.swiglu) {
                        ninfer::ops::linear_swiglu(x, packed.weight, out, policy, workspace,
                                                   launch_stream);
                    } else {
                        ninfer::ops::linear_add(x, packed.weight, out, policy, workspace,
                                                launch_stream);
                    }
                };
                try {
                    cell_us[p] =
                        ninfer::bench::measure_cold_launch(invoke, flush, stream, warmup, repeat)
                            .median_us;
                } catch (const std::exception&) {
                    cudaGetLastError();
                }
            }

            const double a8_tops = cell_us[1] > 0.0 ? tops(profile, token_count, cell_us[1]) : 0.0;
            std::printf("%-12s %6d %12.1f %12.1f %10.1f %10.1f %7.0f%%\n", profile.name,
                        token_count, cell_us[0], cell_us[1],
                        cell_us[0] > 0.0 ? tops(profile, token_count, cell_us[0]) : 0.0, a8_tops,
                        100.0 * a8_tops / kInt8CeilingTops);
            std::fflush(stdout);
        }
    }
    return 0;
}
