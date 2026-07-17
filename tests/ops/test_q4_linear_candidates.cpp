#include "ops/linear/q4/q4_rowsplit_launch.h"
#include "ops/op_tester.h"
#include "ops/row_split_pack.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<double> cpu_q4_linear(const std::vector<float>& x,
                                  const row_split::PackedWeight& weight, std::int32_t n,
                                  std::int32_t k, std::int32_t t) {
    std::vector<double> out(static_cast<std::size_t>(n) * t, 0.0);
    for (std::int32_t col = 0; col < t; ++col) {
        for (std::int32_t row = 0; row < n; ++row) {
            double acc        = 0.0;
            const float* wrow = weight.dequant.data() + static_cast<std::size_t>(row) * k;
            const float* xcol = x.data() + static_cast<std::size_t>(col) * k;
            for (std::int32_t kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(wrow[kk]) * static_cast<double>(xcol[kk]);
            }
            out[static_cast<std::size_t>(col) * n + row] = acc;
        }
    }
    return out;
}

int one_candidate(ops::detail::Q4ScheduleId schedule, ops::detail::Q4KernelVariant variant,
                  std::int32_t n, std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    fill_uniform(source_weight, seed + 1000u, -0.125f, 0.125f);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(source_weight);
    round_to_bf16(x);

    row_split::PackedWeight packed =
        row_split::pack_row_split_lowbit(source_weight, n, k, QType::Q4G64_F16S);
    std::vector<double> ref = cpu_q4_linear(x, packed, n, k, t);

    DBuf dx = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);

    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor ty(dout.p, DType::BF16, {n, t});
    ops::detail::q4_rowsplit_launch_candidate(schedule, variant, tx, packed.device_weight(dw.p), ty,
                                              nullptr);
    cudaDeviceSynchronize();

    const std::string label = std::string(ops::detail::q4_schedule_name(schedule)) + "." +
                              ops::detail::q4_kernel_variant_name(variant) + " [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] C=" + std::to_string(t);
    const Tolerance tolerance = ops::detail::q4_schedule_uses_mma(schedule)
                                    ? Tolerance::linear_tc()
                                    : Tolerance::linear_bf16();
    return verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t), ref,
                  tolerance);
}

int exact_parent_candidate_sampled(ops::detail::Q4ScheduleId schedule,
                                   ops::detail::Q4KernelVariant variant, std::int32_t t,
                                   std::uint32_t seed) {
    constexpr std::int32_t n = 7168;
    constexpr std::int32_t k = 5120;
    auto packed = row_split::make_patterned_weight(QType::Q4G64_F16S, n, k, seed + 1000u);
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    fill_uniform(x, seed, -0.01f, 0.01f);
    round_to_bf16(x);

    DBuf dx = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor ty(dout.p, DType::BF16, {n, t});
    ops::detail::q4_rowsplit_launch_candidate(schedule, variant, tx, packed.device_weight(dw.p), ty,
                                              nullptr);
    cudaDeviceSynchronize();

    const std::vector<double> full = from_device_bf16(dout, static_cast<std::size_t>(n) * t);
    std::vector<double> actual;
    std::vector<double> reference;
    for (const std::int32_t col : {0, t / 2, t - 1}) {
        for (const std::int32_t row : {0, 1, 6143, 6144, n - 2, n - 1}) {
            actual.push_back(full[static_cast<std::size_t>(col) * n + row]);
            reference.push_back(row_split::dot_row_split_lowbit_fp64(
                packed, row, x.data() + static_cast<std::size_t>(col) * k, k));
        }
    }
    const std::string label = std::string(ops::detail::q4_schedule_name(schedule)) + "." +
                              ops::detail::q4_kernel_variant_name(variant) +
                              " [7168,5120] sampled C=" + std::to_string(t);
    return verify(label.c_str(), actual, reference, Tolerance::linear_bf16());
}

int illegal_candidate_rejection() {
    constexpr std::int32_t n = 128;
    constexpr std::int32_t k = 128;
    constexpr std::int32_t t = 5;
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k, 0.0f);
    std::vector<float> x(static_cast<std::size_t>(k) * t, 0.0f);
    row_split::PackedWeight packed =
        row_split::pack_row_split_lowbit(source_weight, n, k, QType::Q4G64_F16S);
    DBuf dx = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor ty(dout.p, DType::BF16, {n, t});

    try {
        ops::detail::q4_rowsplit_launch_candidate(ops::detail::Q4ScheduleId::SimtR8C4,
                                                  ops::detail::Q4KernelVariant::Full, tx,
                                                  packed.device_weight(dw.p), ty, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "q4 fixed candidate accepted an illegal Full variant\n";
    return 1;
}

int legality_contract_rejections() {
    using S = ops::detail::Q4ScheduleId;
    using V = ops::detail::Q4KernelVariant;

    int failures      = 0;
    const auto reject = [&](const char* label, S schedule, V variant,
                            const ops::detail::Q4Problem& problem) {
        if (ops::detail::q4_candidate_is_legal(schedule, variant, problem)) {
            std::cerr << "q4 candidate legality accepted " << label << '\n';
            ++failures;
        }
    };

    reject("GEMV Full lifecycle mismatch", S::GemvR4W1Direct, V::Full, {128, 512, 512, 1});
    reject("static GEMV K mismatch", S::GemvR1W8Direct, V::None, {128, 512, 512, 1});
    reject("SIMT None lifecycle mismatch", S::SimtR8C4, V::None, {128, 1024, 1024, 4});
    reject("MMA None lifecycle mismatch", S::MmaR64C64, V::None, {128, 128, 128, 64});
    reject("Kpad mismatch", S::SimtR8C4, V::Full, {128, 1024, 1152, 4});
    reject("unknown schedule", static_cast<S>(999), V::Predicated, {128, 1024, 1024, 4});
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    using S      = ops::detail::Q4ScheduleId;
    using V      = ops::detail::Q4KernelVariant;
    int failures = 0;

    failures += one_candidate(S::GemvR4W1Direct, V::None, 128, 512, 1, 13u);
    failures += one_candidate(S::GemvR4W1Direct, V::None, 128, 5120, 1, 15u);
    failures += one_candidate(S::GemvR1W8Direct, V::None, 128, 5120, 1, 18u);

    failures += one_candidate(S::SimtR8C4, V::Full, 128, 1024, 4, 19u);
    failures += one_candidate(S::SimtR8C4, V::Predicated, 144, 1152, 5, 23u);
    failures += one_candidate(S::SimtR8C8, V::Full, 128, 1024, 8, 29u);
    failures += one_candidate(S::SimtR8C8, V::Predicated, 144, 1152, 9, 31u);
    failures += one_candidate(S::SimtR8C4, V::Full, 128, 3072, 4, 33u);
    failures += one_candidate(S::SimtR8C8, V::Predicated, 144, 3072, 9, 35u);

    failures += one_candidate(S::MmaR64C64, V::Full, 128, 128, 64, 37u);
    failures += one_candidate(S::MmaR64C64, V::Predicated, 144, 128, 65, 41u);
    failures += one_candidate(S::MmaR64C128, V::Full, 128, 128, 128, 43u);
    failures += one_candidate(S::MmaR64C128, V::Predicated, 144, 128, 129, 47u);
    failures += one_candidate(S::MmaR64C64, V::Full, 128, 384, 64, 49u);
    failures += one_candidate(S::MmaR64C128, V::Predicated, 144, 384, 129, 53u);

    // Exact Attention parent geometry: qualify every Small-T mainloop candidate and both tiled
    // lifecycle variants against the independent payload decoder, including the 6143/6144 seam.
    failures += exact_parent_candidate_sampled(S::GemvR1W8Direct, V::None, 1, 61u);
    failures += exact_parent_candidate_sampled(S::GemvR4W1Direct, V::None, 1, 63u);
    failures += exact_parent_candidate_sampled(S::SimtR8C4, V::Predicated, 3, 65u);
    failures += exact_parent_candidate_sampled(S::SimtR8C4, V::Full, 4, 67u);
    failures += exact_parent_candidate_sampled(S::SimtR8C8, V::Predicated, 7, 69u);
    failures += exact_parent_candidate_sampled(S::SimtR8C8, V::Full, 8, 71u);

    failures += illegal_candidate_rejection();
    failures += legality_contract_rejections();

    std::cout << (failures ? "FAIL" : "OK") << " Q4 Linear fixed candidates\n";
    return failures ? 1 : 0;
}
