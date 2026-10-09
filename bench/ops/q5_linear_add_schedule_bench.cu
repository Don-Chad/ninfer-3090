// Route-boundary sweep for Q5 linear-add (the 35B-A3B residual path).
//
// The catch-up merge replaced an existing boundary here rather than adding a new table:
// {49,128} MmaResidualR64C64 became {49,192} MmaResidualR64C32S4 for k=6144 (and ...S3 for
// k=17408), on upstream's sm_120 authority. R64C64 is still a registered schedule that the table
// no longer selects at any width, which is the shape of every routing bug found in this merge.
//
// Both supported k are swept because they carry different tables.

#include "ninfer/ops/linear_add.h"

#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
namespace detail = ninfer::ops::detail;

constexpr std::int32_t kRows = 5120;

void sweep_for_k(std::int32_t hidden, const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens =
        *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
        QType::Q5_G64_FP16, kRows, hidden, hidden, 0x52U);
    ninfer::DeviceBuffer input(static_cast<std::size_t>(hidden) * max_tokens * 2);
    ninfer::DeviceBuffer residual(static_cast<std::size_t>(kRows) * max_tokens * 2);

    const auto make = [&](void (*launch)(const Tensor&, const Weight&, Tensor&, cudaStream_t)) {
        return [&, launch](std::int32_t tokens, cudaStream_t stream) {
            Tensor x(input.p, DType::BF16, {hidden, tokens});
            Tensor out(residual.p, DType::BF16, {kRows, tokens});
            launch(x, packed.weight, out, stream);
        };
    };

    std::vector<ninfer::bench::SweepEntry> schedules{
        // split2_exact is a decode-shaped route registered for narrow extents; past its domain it
        // does not fail, it just stops being meaningful. (The gemv residual kernel was removed
        // upstream; split2 took T=1 from it on sm_86 before that.)
        {"split2_exact", make(&detail::q5_linear_add_split2_exact_launch), 10},
        {"small_t_mma", make(&detail::q5_linear_add_small_t_mma_launch), 32},
        {"mma_r64_c16", make(&detail::q5_linear_add_mma_r64_c16_launch), 0},
        {"mma_r64_c24", make(&detail::q5_linear_add_mma_r64_c24_launch), 0},
        {"mma_r64_c32", make(&detail::q5_linear_add_mma_r64_c32_launch), 0},
        {"mma_r64_c64", make(&detail::q5_linear_add_mma_r64_c64_launch), 0},
        {"mma_r64_c32_s3", make(&detail::q5_linear_add_mma_r64_c32_s3_launch), 0},
        {"mma_r64_c32_s4", make(&detail::q5_linear_add_mma_r64_c32_s4_launch), 0},
        {"mma_r64_c128", make(&detail::q5_linear_add_mma_r64_c128_launch), 0},
    };

    const std::string title = "q5 linear_add k=" + std::to_string(hidden);

    ninfer::bench::SweepOptions options = base;
    options.title                       = title.c_str();
    std::string routed;
    options.routed_name = [hidden, &routed](std::int32_t tokens) -> const char* {
        const detail::Q5LinearAddProblem problem{kRows, hidden, hidden, tokens};
        routed = detail::q5_linear_add_schedule_name(
            detail::q5_linear_add_resolve_plan(problem).schedule);
        return routed.c_str();
    };
    const std::size_t workspace_capacity = ninfer::ops::linear_add_workspace_capacity_bytes(
        QType::Q5_G64_FP16, kRows, hidden, 1, max_tokens);
    ninfer::WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor out(residual.p, DType::BF16, {kRows, tokens});
        ninfer::ops::linear_add(x, packed.weight, out, workspace, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::SweepOptions options;
    options.tokens = {1, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 160, 176, 192, 208, 256};
    if (!ninfer::bench::parse_sweep_args(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s [--tokens T,...] [--repeat N] [--warmup N] [--spread]\n", argv[0]);
        return 2;
    }
    sweep_for_k(6144, options);
    sweep_for_k(17408, options);
    return 0;
}
