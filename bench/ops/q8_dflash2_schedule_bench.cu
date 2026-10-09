// Route-boundary sweep for the two Q8 DFlash2 tables, on the 27B hot path.
//
// These tables arrived whole in the catch-up merge -- new, live, and tuned on sm_120. Nothing in
// this fork has ever measured them, and DFlash2 on the 27B is a commonly used configuration here,
// so every boundary below is currently someone else's guess about a different GPU.
//
// Both Ops are swept in one binary because they are the same decision on the same path: the
// drafter runs attention-input and SwiGLU back to back at the same column count.
//
// Column domains, which the kernels do not police silently but *do* enforce by throwing:
// the attention small-T kernel covers 1..48 and the SwiGLU one 1..40, both dispatching through
// a table of per-T specialisations. Outside that they throw std::invalid_argument, which the
// sweep driver catches and prints as n/a -- but declaring max_cols keeps the table readable.

#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
namespace detail = ninfer::ops::detail;

// The exact shape is_dflash2_shape() admits in q8_attn_input_plan.cpp. Miss it by one and the
// resolver silently routes through the companion table instead, which would make every number
// here a measurement of the wrong thing.
constexpr std::int32_t kHidden      = 5120;
constexpr std::int32_t kQueryRows   = 4096;
constexpr std::int32_t kKVRows      = 1024;
constexpr std::int32_t kParentRows  = 6144;
constexpr std::int32_t kGateUpRows  = 34816;
constexpr std::int32_t kOutputRows  = 17408;

void sweep_attn_input(const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
        QType::Q8_G32_FP16, kParentRows, kHidden, kHidden, 0x51U);
    ninfer::DeviceBuffer input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    ninfer::DeviceBuffer query(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    ninfer::DeviceBuffer key(static_cast<std::size_t>(kKVRows) * max_tokens * 2);
    ninfer::DeviceBuffer value(static_cast<std::size_t>(kKVRows) * max_tokens * 2);

    using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, cudaStream_t);
    const auto make = [&](Launch launch) {
        return [&, launch](std::int32_t tokens, cudaStream_t stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor q(query.p, DType::BF16, {kQueryRows, tokens});
            Tensor k(key.p, DType::BF16, {kKVRows, tokens});
            Tensor v(value.p, DType::BF16, {kKVRows, tokens});
            launch(x, packed.weight, q, k, v, stream);
        };
    };

    std::vector<ninfer::bench::SweepEntry> schedules{
        {"small_t", make(&detail::q8_dflash2_attn_input_small_t_launch), 48},
        {"mma_r16_c64_k128", make(&detail::q8_dflash2_attn_input_mma_r16_c64_k128_launch), 0},
        {"mma_r32_c32_k128", make(&detail::q8_dflash2_attn_input_mma_r32_c32_k128_launch), 0},
        {"mma_r32_c64_k128", make(&detail::q8_dflash2_attn_input_mma_r32_c64_k128_launch), 0},
        {"mma_r32_c64", make(&detail::q8_dflash2_attn_input_mma_r32_c64_launch), 0},
        {"mma_r64_c128", make(&detail::q8_dflash2_attn_input_mma_r64_c128_launch), 0},
    };

    ninfer::bench::SweepOptions options = base;
    options.title                       = "q8 dflash2 attn_input (27B)";
    std::string routed;
    options.routed_name = [&routed](std::int32_t tokens) -> const char* {
        const detail::Q8AttnInputProblem problem{kHidden,     kQueryRows, kKVRows,
                                                 kParentRows, kHidden,    tokens};
        routed = detail::q8_attn_input_schedule_name(
            detail::q8_attn_input_resolve_plan(problem).schedule);
        return routed.c_str();
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {kHidden, tokens});
        Tensor q(query.p, DType::BF16, {kQueryRows, tokens});
        Tensor k(key.p, DType::BF16, {kKVRows, tokens});
        Tensor v(value.p, DType::BF16, {kKVRows, tokens});
        detail::q8_attn_input_dispatch(x, packed.weight, q, k, v, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

void sweep_linear_swiglu(const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
        QType::Q8_G32_FP16, kGateUpRows, kHidden, kHidden, 0x51U);
    ninfer::DeviceBuffer input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    ninfer::DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_tokens * 2);

    using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
    const auto make = [&](Launch launch) {
        return [&, launch](std::int32_t tokens, cudaStream_t stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
            launch(x, packed.weight, out, stream);
        };
    };

    std::vector<ninfer::bench::SweepEntry> schedules{
        {"small_t", make(&detail::q8_dflash2_linear_swiglu_small_t_launch), 40},
        {"mma_r32_c64_k128", make(&detail::q8_dflash2_linear_swiglu_mma_r32_c64_k128_launch), 0},
        {"mma_r64_c64_k128", make(&detail::q8_dflash2_linear_swiglu_mma_r64_c64_k128_launch), 0},
        {"mma_r64_c80_k128", make(&detail::q8_dflash2_linear_swiglu_mma_r64_c80_k128_launch), 0},
        {"mma_r64_c96_k128", make(&detail::q8_dflash2_linear_swiglu_mma_r64_c96_k128_launch), 0},
        // What DFlash2MmaR64C128 resolves to: the shared non-DFlash2 kernel, not a k128 variant.
        {"mma_r64_c128", make(&detail::q8_linear_swiglu_mma_r64_c128_launch), 0},
    };

    ninfer::bench::SweepOptions options = base;
    options.title                       = "q8 dflash2 linear_swiglu (27B)";
    std::string routed;
    options.routed_name = [&routed](std::int32_t tokens) -> const char* {
        const detail::Q8LinearSwiGluProblem problem{kGateUpRows, kOutputRows, kHidden, kHidden,
                                                    tokens};
        routed = detail::q8_linear_swiglu_schedule_name(
            detail::q8_linear_swiglu_resolve_plan(problem).schedule);
        return routed.c_str();
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {kHidden, tokens});
        Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
        detail::q8_linear_swiglu_dispatch(x, packed.weight, out, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::SweepOptions options;
    options.tokens = {1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 256, 384};
    if (!ninfer::bench::parse_sweep_args(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s [--tokens T,...] [--repeat N] [--warmup N] [--spread]\n", argv[0]);
        return 2;
    }
    sweep_attn_input(options);
    sweep_linear_swiglu(options);
    return 0;
}
