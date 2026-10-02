// Correctness cover for the integer-activation prefill routes: the Q4G64 gate_up LinearSwiGLU and
// the Q5G64 down LinearAdd.
//
// These routes are reached from the target variant rather than through ops::linear_swiglu, so the
// shared run_profile harness cannot see them. They are verified here the same way: a deterministic
// row-split payload from the quantized-weight fixture, an FP64 oracle over exactly decoded weights
// and represented BF16 activations, and the activation-compute criterion for A8.
//
// The pass bound is upstream's own kA8QuantizationAllowance (0.04), the allowance the A8 activation
// compute path is held to in tests/ops/linear/linear_test_common.cpp. The observed relative L2 is
// printed on every case so drift inside that allowance is visible rather than silent -- it measures
// about 0.009 on the real 27B weight, so a four-fold regression would still pass the gate but be
// obvious in the output.

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::test::ReductionCriterion;
using ninfer::test::ReductionStats;
using ninfer::test::compute_reduction_stats;
using ninfer::test::verify_reduction;
using ninfer::test::quantized_weight::PackedWeight;
namespace qw = ninfer::test::quantized_weight;

// One BF16 unit roundoff, and the A8 activation-quantisation allowance, both mirroring
// tests/ops/linear/linear_test_common.cpp.
constexpr double kBf16UnitRoundoff        = 1.0 / 256.0;
constexpr double kA8QuantizationAllowance = 0.04;
constexpr ReductionCriterion kA8Criterion{kA8QuantizationAllowance, kBf16UnitRoundoff,
                                          1.5 * kA8QuantizationAllowance};

// Hidden states are roughly Gaussian with a few fixed channels an order of magnitude larger. The
// outlier channels are what per-token activation scaling gets wrong, so they belong in the fixture.
std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t r = 0; r < rows; ++r) {
            const std::uint64_t h = qw::detail::mix64(
                (static_cast<std::uint64_t>(t) << 32) ^ static_cast<std::uint64_t>(r) ^ seed);
            float v = static_cast<float>(static_cast<std::int32_t>(h & 0xffffu) - 32768) / 32768.0F;
            if ((r % 431) == 7) { v *= 24.0F; } // fixed outlier channels
            // BF16, round-to-nearest-even, so the oracle sees exactly what the kernel reads.
            std::uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            const std::uint32_t lsb = (bits >> 16) & 1u;
            bits += 0x7fffu + lsb;
            out[static_cast<std::size_t>(t) * rows + r] = static_cast<std::uint16_t>(bits >> 16);
        }
    }
    return out;
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::vector<double> read_bf16(const test::GuardedDeviceBuffer& buffer, std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    buffer.copy_to_host(host.data(), elements * sizeof(std::uint16_t));
    std::vector<double> out(elements);
    for (std::size_t i = 0; i < elements; ++i) { out[i] = bf16_value(host[i]); }
    return out;
}

// Sampled (row, token) pairs, so the FP64 oracle stays affordable at registered shapes.
struct Samples {
    std::vector<double> actual;
    std::vector<double> reference;
};

int run_gate_up(std::int32_t tokens) {
    constexpr std::int32_t kRows = 34816;
    constexpr std::int32_t kCols = 5120;
    constexpr std::int32_t kOut  = kRows / 2;

    const PackedWeight host_weight =
        qw::make_patterned_weight(QType::Q4_G64_FP16, kRows, kCols, 4801U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 91U);

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));

    const std::size_t out_elements = static_cast<std::size_t>(kOut) * tokens;
    test::GuardedDeviceBuffer output(out_elements * sizeof(std::uint16_t));
    output.fill(0xff);

    WorkspaceArena workspace(
        std::max<std::size_t>(ops::detail::q4a8_swiglu_workspace_capacity_bytes(tokens, tokens), 256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor destination(output.data(), DType::BF16, {kOut, tokens});
    ops::detail::q4a8_swiglu_launch(x, weight, destination, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize q4a8 swiglu");

    const std::string label = "LinearSwiGLU Q4_A8INT T=" + std::to_string(tokens);
    int failures            = 0;
    failures += output.verify_guards(label);

    const std::vector<double> got = read_bf16(output, out_elements);
    Samples s;
    std::vector<float> input(static_cast<std::size_t>(kCols));
    for (int pick = 0; pick < 24; ++pick) {
        const std::int32_t token = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 7919U + 3U) % static_cast<std::uint64_t>(tokens));
        const std::int32_t row = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 104729U + 11U) % static_cast<std::uint64_t>(kOut));
        for (std::int32_t k = 0; k < kCols; ++k) {
            input[static_cast<std::size_t>(k)] =
                bf16_value(activation[static_cast<std::size_t>(token) * kCols + k]);
        }
        const double gate = qw::dot_fp64(host_weight, row, input.data(), kCols);
        const double up   = qw::dot_fp64(host_weight, kOut + row, input.data(), kCols);
        s.reference.push_back(gate / (1.0 + std::exp(-gate)) * up);
        s.actual.push_back(got[static_cast<std::size_t>(token) * kOut + row]);
    }

    const ReductionStats stats = compute_reduction_stats(
        s.actual.data(), s.reference.data(), static_cast<std::int64_t>(s.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kA8QuantizationAllowance << ")\n";
    failures += verify_reduction(label, s.actual, s.reference, kA8Criterion);
    return failures;
}

// SmallTTiledI8 (q4_small_t_mma_i8.cuh): exploratory int8 tensor-core small-T route for the same
// gate_up weight, T=32 only so far. Not routed by resolve_plan -- reached directly through
// execute_schedule, the same way the schedule bench times it. Same oracle and allowance as the
// prefill route above: same weight profile, same per-(token,64-group) s8 activation quantisation.
int run_gate_up_small_t_i8(std::int32_t kTokens) {
    constexpr std::int32_t kRows = 34816;
    constexpr std::int32_t kCols = 5120;
    constexpr std::int32_t kOut  = kRows / 2;

    const PackedWeight host_weight =
        qw::make_patterned_weight(QType::Q4_G64_FP16, kRows, kCols, 4802U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, kTokens, 92U);

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));

    const std::size_t out_elements = static_cast<std::size_t>(kOut) * kTokens;
    test::GuardedDeviceBuffer output(out_elements * sizeof(std::uint16_t));
    output.fill(0xff);

    WorkspaceArena workspace(std::max<std::size_t>(
        ops::detail::q4_linear_swiglu_small_t_tiled_i8_workspace_bytes(kTokens), 256));
    Tensor x(device_x.data(), DType::BF16, {kCols, kTokens});
    Tensor destination(output.data(), DType::BF16, {kOut, kTokens});
    ops::detail::q4_linear_swiglu_execute_schedule(
        ops::detail::Q4LinearSwiGluScheduleId::SmallTTiledI8, x, weight, destination, workspace,
        nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize q4 small-T i8 swiglu");

    const std::string label = "LinearSwiGLU Q4_SMALL_T_I8 T=" + std::to_string(kTokens);
    int failures            = 0;
    failures += output.verify_guards(label);

    const std::vector<double> got = read_bf16(output, out_elements);
    Samples s;
    std::vector<float> input(static_cast<std::size_t>(kCols));
    for (int pick = 0; pick < 24; ++pick) {
        const std::int32_t token = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 7927U + 3U) % static_cast<std::uint64_t>(kTokens));
        const std::int32_t row = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 104743U + 11U) % static_cast<std::uint64_t>(kOut));
        for (std::int32_t k = 0; k < kCols; ++k) {
            input[static_cast<std::size_t>(k)] =
                bf16_value(activation[static_cast<std::size_t>(token) * kCols + k]);
        }
        const double gate = qw::dot_fp64(host_weight, row, input.data(), kCols);
        const double up   = qw::dot_fp64(host_weight, kOut + row, input.data(), kCols);
        s.reference.push_back(gate / (1.0 + std::exp(-gate)) * up);
        s.actual.push_back(got[static_cast<std::size_t>(token) * kOut + row]);
    }

    const ReductionStats stats = compute_reduction_stats(
        s.actual.data(), s.reference.data(), static_cast<std::int64_t>(s.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kA8QuantizationAllowance << ")\n";
    failures += verify_reduction(label, s.actual, s.reference, kA8Criterion);
    return failures;
}

// kCols selects the registered profile: 17408 is mlp/down, 6144 the attention o_proj and GDN
// out_proj. They share one kernel per format, so both widths and both formats are checked against
// the oracle.
int run_down(std::int32_t tokens, std::int32_t kCols, QType format) {
    constexpr std::int32_t kRows = 5120;

    const PackedWeight host_weight = qw::make_patterned_weight(format, kRows, kCols, 5501U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 77U);
    const std::vector<std::uint16_t> residual0  = make_activation(kRows, tokens, 13U);

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));

    const std::size_t res_elements = static_cast<std::size_t>(kRows) * tokens;
    test::GuardedDeviceBuffer device_residual(res_elements * sizeof(std::uint16_t));
    device_residual.copy_from_host(residual0.data(), res_elements * sizeof(std::uint16_t));

    WorkspaceArena workspace(std::max<std::size_t>(
        ops::detail::a8_add_workspace_capacity_bytes(kCols, tokens, tokens), 256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor residual(device_residual.data(), DType::BF16, {kRows, tokens});
    ops::detail::a8_add_launch(x, weight, residual, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize a8 add");

    const std::string label = std::string("LinearAdd ") +
                              (format == QType::Q5_G64_FP16 ? "Q5" : "Q4") + "_A8INT K=" +
                              std::to_string(kCols) + " T=" + std::to_string(tokens);
    int failures            = 0;
    failures += device_residual.verify_guards(label);

    const std::vector<double> got = read_bf16(device_residual, res_elements);
    Samples s;
    std::vector<float> input(static_cast<std::size_t>(kCols));
    for (int pick = 0; pick < 24; ++pick) {
        const std::int32_t token = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 6151U + 5U) % static_cast<std::uint64_t>(tokens));
        const std::int32_t row = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 199933U + 17U) % static_cast<std::uint64_t>(kRows));
        for (std::int32_t k = 0; k < kCols; ++k) {
            input[static_cast<std::size_t>(k)] =
                bf16_value(activation[static_cast<std::size_t>(token) * kCols + k]);
        }
        const double acc = qw::dot_fp64(host_weight, row, input.data(), kCols);
        const double base =
            bf16_value(residual0[static_cast<std::size_t>(token) * kRows + row]);
        s.reference.push_back(base + acc);
        s.actual.push_back(got[static_cast<std::size_t>(token) * kRows + row]);
    }

    const ReductionStats stats = compute_reduction_stats(
        s.actual.data(), s.reference.data(), static_cast<std::int64_t>(s.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kA8QuantizationAllowance << ")\n";
    failures += verify_reduction(label, s.actual, s.reference, kA8Criterion);
    return failures;
}


// The GDN input projection is two parents over one activation, scattered into three destinations:
// qk -> qkv[0,4096), value_z rows [0,6144) -> qkv[4096,10240), rows [6144,12288) -> z. A row
// mapped to the wrong destination still produces plausible numbers in the other two, so every
// range is sampled.
int run_gdn_input(std::int32_t tokens) {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kQkRows    = 4096;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kQkvRows   = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;

    const PackedWeight host_qk =
        qw::make_patterned_weight(QType::Q4_G64_FP16, kQkRows, kHidden, 909U);
    const PackedWeight host_vz =
        qw::make_patterned_weight(QType::Q5_G64_FP16, kParentRows, kHidden, 313U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 41U);

    test::GuardedDeviceBuffer device_qk(host_qk.payload.size());
    device_qk.copy_from_host(host_qk.payload.data(), host_qk.payload.size());
    test::GuardedDeviceBuffer device_vz(host_vz.payload.size());
    device_vz.copy_from_host(host_vz.payload.data(), host_vz.payload.size());
    const Weight qk = host_qk.device_weight(device_qk.data());
    const Weight vz = host_vz.device_weight(device_vz.data());

    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_qkv(static_cast<std::size_t>(kQkvRows) * tokens *
                                         sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_z(static_cast<std::size_t>(kZRows) * tokens *
                                       sizeof(std::uint16_t));

    WorkspaceArena workspace(std::max<std::size_t>(
        ops::detail::q4_q5_gdn_input_a8_workspace_capacity_bytes(tokens, tokens), 256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor qkv(device_qkv.data(), DType::BF16, {kQkvRows, tokens});
    Tensor z(device_z.data(), DType::BF16, {kZRows, tokens});
    ops::detail::q4_q5_gdn_input_a8_launch(x, qk, vz, qkv, z, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize q4_q5 gdn input a8");

    const std::string label = "GdnInputProj A8INT T=" + std::to_string(tokens);
    int failures            = 0;
    failures += device_qkv.verify_guards(label);
    failures += device_z.verify_guards(label);

    const std::vector<double> got_qkv =
        read_bf16(device_qkv, static_cast<std::size_t>(kQkvRows) * tokens);
    const std::vector<double> got_z =
        read_bf16(device_z, static_cast<std::size_t>(kZRows) * tokens);

    Samples s;
    std::vector<float> input(static_cast<std::size_t>(kHidden));
    for (int pick = 0; pick < 24; ++pick) {
        const std::int32_t token = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 7907U + 3U) % static_cast<std::uint64_t>(tokens));
        for (std::int32_t k = 0; k < kHidden; ++k) {
            input[static_cast<std::size_t>(k)] =
                bf16_value(activation[static_cast<std::size_t>(token) * kHidden + k]);
        }
        // Rotate through the three destination ranges so each is sampled evenly.
        const int range = pick % 3;
        if (range == 0) {
            const std::int32_t row = static_cast<std::int32_t>(
                qw::detail::mix64(pick * 131071U + 11U) % static_cast<std::uint64_t>(kQkRows));
            s.reference.push_back(qw::dot_fp64(host_qk, row, input.data(), kHidden));
            s.actual.push_back(got_qkv[static_cast<std::size_t>(token) * kQkvRows + row]);
        } else if (range == 1) {
            const std::int32_t row = static_cast<std::int32_t>(
                qw::detail::mix64(pick * 65521U + 13U) % static_cast<std::uint64_t>(kValueRows));
            s.reference.push_back(qw::dot_fp64(host_vz, row, input.data(), kHidden));
            s.actual.push_back(
                got_qkv[static_cast<std::size_t>(token) * kQkvRows + kQkRows + row]);
        } else {
            const std::int32_t row = static_cast<std::int32_t>(
                qw::detail::mix64(pick * 32749U + 17U) % static_cast<std::uint64_t>(kZRows));
            s.reference.push_back(qw::dot_fp64(host_vz, kValueRows + row, input.data(), kHidden));
            s.actual.push_back(got_z[static_cast<std::size_t>(token) * kZRows + row]);
        }
    }

    const ReductionStats stats = compute_reduction_stats(
        s.actual.data(), s.reference.data(), static_cast<std::int64_t>(s.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kA8QuantizationAllowance << ")\n";
    failures += verify_reduction(label, s.actual, s.reference, kA8Criterion);
    return failures;
}


// The attention input projection is two parents over one activation, split into four destinations:
// query_key rows [0,6144) -> q and [6144,7168) -> k, gate_value the same into gate and v. The k/v
// ranges are 1024 rows, the narrowest tile grid this kernel is launched with.
int run_attn_input(std::int32_t tokens) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQRows      = 6144;
    constexpr std::int32_t kKvRows     = 1024;
    constexpr std::int32_t kParentRows = kQRows + kKvRows;

    const PackedWeight host_qk =
        qw::make_patterned_weight(QType::Q4_G64_FP16, kParentRows, kHidden, 4441U);
    const PackedWeight host_gv =
        qw::make_patterned_weight(QType::Q5_G64_FP16, kParentRows, kHidden, 977U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 59U);

    test::GuardedDeviceBuffer device_qk(host_qk.payload.size());
    device_qk.copy_from_host(host_qk.payload.data(), host_qk.payload.size());
    test::GuardedDeviceBuffer device_gv(host_gv.payload.size());
    device_gv.copy_from_host(host_gv.payload.data(), host_gv.payload.size());
    const Weight qk = host_qk.device_weight(device_qk.data());
    const Weight gv = host_gv.device_weight(device_gv.data());

    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t q_elements  = static_cast<std::size_t>(kQRows) * tokens;
    const std::size_t kv_elements = static_cast<std::size_t>(kKvRows) * tokens;
    test::GuardedDeviceBuffer device_q(q_elements * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_gate(q_elements * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_k(kv_elements * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_v(kv_elements * sizeof(std::uint16_t));

    WorkspaceArena workspace(std::max<std::size_t>(
        ops::detail::q4_q5_attn_input_a8_workspace_capacity_bytes(tokens, tokens), 256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor q(device_q.data(), DType::BF16, {kQRows, tokens});
    Tensor gate(device_gate.data(), DType::BF16, {kQRows, tokens});
    Tensor k(device_k.data(), DType::BF16, {kKvRows, tokens});
    Tensor v(device_v.data(), DType::BF16, {kKvRows, tokens});
    ops::detail::q4_q5_attn_input_a8_launch(x, qk, gv, q, gate, k, v, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize q4_q5 attn input a8");

    const std::string label = "AttnInputProj A8INT T=" + std::to_string(tokens);
    int failures            = 0;
    failures += device_q.verify_guards(label);
    failures += device_gate.verify_guards(label);
    failures += device_k.verify_guards(label);
    failures += device_v.verify_guards(label);

    const std::vector<double> got_q    = read_bf16(device_q, q_elements);
    const std::vector<double> got_gate = read_bf16(device_gate, q_elements);
    const std::vector<double> got_k    = read_bf16(device_k, kv_elements);
    const std::vector<double> got_v    = read_bf16(device_v, kv_elements);

    Samples s;
    std::vector<float> input(static_cast<std::size_t>(kHidden));
    for (int pick = 0; pick < 24; ++pick) {
        const std::int32_t token = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 5153U + 7U) % static_cast<std::uint64_t>(tokens));
        for (std::int32_t c = 0; c < kHidden; ++c) {
            input[static_cast<std::size_t>(c)] =
                bf16_value(activation[static_cast<std::size_t>(token) * kHidden + c]);
        }
        const int range = pick % 4;
        const std::int32_t wide = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 99991U + 23U) % static_cast<std::uint64_t>(kQRows));
        const std::int32_t narrow = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 49999U + 29U) % static_cast<std::uint64_t>(kKvRows));
        if (range == 0) {
            s.reference.push_back(qw::dot_fp64(host_qk, wide, input.data(), kHidden));
            s.actual.push_back(got_q[static_cast<std::size_t>(token) * kQRows + wide]);
        } else if (range == 1) {
            s.reference.push_back(qw::dot_fp64(host_qk, kQRows + narrow, input.data(), kHidden));
            s.actual.push_back(got_k[static_cast<std::size_t>(token) * kKvRows + narrow]);
        } else if (range == 2) {
            s.reference.push_back(qw::dot_fp64(host_gv, wide, input.data(), kHidden));
            s.actual.push_back(got_gate[static_cast<std::size_t>(token) * kQRows + wide]);
        } else {
            s.reference.push_back(qw::dot_fp64(host_gv, kQRows + narrow, input.data(), kHidden));
            s.actual.push_back(got_v[static_cast<std::size_t>(token) * kKvRows + narrow]);
        }
    }

    const ReductionStats stats = compute_reduction_stats(
        s.actual.data(), s.reference.data(), static_cast<std::int64_t>(s.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kA8QuantizationAllowance << ")\n";
    failures += verify_reduction(label, s.actual, s.reference, kA8Criterion);
    return failures;
}

// The routes must decline anything they do not cover, so the caller falls back to A16 rather than
// producing a wrong answer. Decode and partial prefill chunks depend on this.
int run_admission() {
    int failures = 0;
    const PackedWeight q4 =
        qw::make_patterned_weight(QType::Q4_G64_FP16, 34816, 5120, 1U);
    const PackedWeight q5 =
        qw::make_patterned_weight(QType::Q5_G64_FP16, 5120, 17408, 2U);
    const PackedWeight wrong_shape =
        qw::make_patterned_weight(QType::Q4_G64_FP16, 4096, 5120, 3U);
    // The mixer output projections: same kernel, narrower K.
    const PackedWeight q5_mixer =
        qw::make_patterned_weight(QType::Q5_G64_FP16, 5120, 6144, 4U);
    const PackedWeight q5_unregistered =
        qw::make_patterned_weight(QType::Q5_G64_FP16, 5120, 8192, 5U);
    const PackedWeight q4_down = qw::make_patterned_weight(QType::Q4_G64_FP16, 5120, 17408, 6U);
    const PackedWeight q4_mixer = qw::make_patterned_weight(QType::Q4_G64_FP16, 5120, 6144, 7U);

    void* fake = reinterpret_cast<void*>(static_cast<std::uintptr_t>(4096));
    struct Case {
        const char* what;
        bool got;
        bool want;
    };
    const Case cases[] = {
        {"q4 accepts 128 tokens", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 128), true},
        {"q4 accepts 1024 tokens", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 1024), true},
        {"q4 declines decode", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 1), false},
        {"q4 declines partial tile", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 200), false},
        {"q4 declines zero tokens", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 0), false},
        {"q4 declines a Q5 weight", ops::detail::q4a8_swiglu_supported(q5.device_weight(fake), 128), false},
        {"q4 declines another shape", ops::detail::q4a8_swiglu_supported(wrong_shape.device_weight(fake), 128), false},
        {"q4 declines a null payload", ops::detail::q4a8_swiglu_supported(q4.device_weight(nullptr), 128), false},
        {"add accepts 128 tokens", ops::detail::a8_add_supported(q5.device_weight(fake), 128), true},
        {"add declines decode", ops::detail::a8_add_supported(q5.device_weight(fake), 1), false},
        {"add declines partial tile", ops::detail::a8_add_supported(q5.device_weight(fake), 129), false},
        {"add declines the gate_up shape", ops::detail::a8_add_supported(q4.device_weight(fake), 128), false},
        {"add declines a null payload", ops::detail::a8_add_supported(q5.device_weight(nullptr), 128), false},
        {"add accepts the Q5 6144 mixer output", ops::detail::a8_add_supported(q5_mixer.device_weight(fake), 128), true},
        {"add declines an unregistered K", ops::detail::a8_add_supported(q5_unregistered.device_weight(fake), 128), false},
        {"add accepts a Q4 down", ops::detail::a8_add_supported(q4_down.device_weight(fake), 128), true},
        {"add accepts a Q4 mixer output", ops::detail::a8_add_supported(q4_mixer.device_weight(fake), 128), true},
    };
    for (const Case& c : cases) {
        if (c.got != c.want) {
            std::cerr << "admission: " << c.what << " returned " << (c.got ? "true" : "false")
                      << ", expected " << (c.want ? "true" : "false") << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        int failures = run_admission();
        for (const std::int32_t tokens : {128, 256, 512}) {
            failures += run_gate_up(tokens);
            for (const QType format : {QType::Q4_G64_FP16, QType::Q5_G64_FP16}) {
                failures += run_down(tokens, 17408, format);
                failures += run_down(tokens, 6144, format);
            }
            failures += run_gdn_input(tokens);
            failures += run_attn_input(tokens);
        }
        // Every band of the T=2..32 dispatch, including widths that exercise column masking.
        for (const std::int32_t t : {2, 5, 8, 9, 16, 17, 24, 25, 31, 32}) {
            failures += run_gate_up_small_t_i8(t);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " integer-activation prefill routes correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "integer-activation prefill route test failed: " << error.what() << '\n';
        return 1;
    }
}
