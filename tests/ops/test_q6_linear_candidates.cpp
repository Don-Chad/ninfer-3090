#include "ops/linear/q6/q6_rowsplit_launch.h"
#include "ops/op_tester.h"
#include "ops/row_split_pack.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<double> cpu_q6_linear(const std::vector<float>& x,
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

int one_candidate(ops::detail::Q6ScheduleId schedule, ops::detail::Q6KernelVariant variant,
                  std::int32_t n, std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    fill_uniform(source_weight, seed + 1000u, -0.125f, 0.125f);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(source_weight);
    round_to_bf16(x);

    row_split::PackedWeight packed =
        row_split::pack_row_split_lowbit(source_weight, n, k, QType::Q6G64_F16S);
    std::vector<double> ref = cpu_q6_linear(x, packed, n, k, t);

    DBuf dx = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);

    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor ty(dout.p, DType::BF16, {n, t});
    ops::detail::q6_rowsplit_launch_candidate(schedule, variant, tx, packed.device_weight(dw.p), ty,
                                              nullptr);
    cudaDeviceSynchronize();

    const std::string label = std::string(ops::detail::q6_schedule_name(schedule)) + "." +
                              ops::detail::q6_kernel_variant_name(variant) + " [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] C=" + std::to_string(t);
    const Tolerance tolerance = ops::detail::q6_schedule_uses_mma(schedule)
                                    ? Tolerance::linear_tc()
                                    : Tolerance::linear_bf16();
    return verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t), ref,
                  tolerance);
}

int legality_contract_rejections() {
    using S = ops::detail::Q6ScheduleId;
    using V = ops::detail::Q6KernelVariant;

    int failures      = 0;
    const auto reject = [&](const char* label, S schedule, V variant,
                            const ops::detail::Q6Problem& problem) {
        if (ops::detail::q6_candidate_is_legal(schedule, variant, problem)) {
            std::cerr << "q6 candidate legality accepted " << label << '\n';
            ++failures;
        }
    };

    reject("SIMT Full lifecycle mismatch", S::SimtR8C4, V::Full, {128, 1536, 1536, 4});
    reject("MMA None lifecycle mismatch", S::MmaR64C64, V::None, {128, 1536, 1536, 64});
    reject("MMA Full column mismatch", S::MmaR64C64, V::Full, {128, 1536, 1536, 32});
    reject("MMA Full Kpad mismatch", S::MmaR64C128, V::Full, {128, 1408, 1536, 128});
    reject("invalid padded K", S::SimtR8C4, V::None, {128, 1536, 1472, 4});
    reject("unknown schedule", static_cast<S>(999), V::Predicated, {128, 1536, 1536, 64});
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    using S      = ops::detail::Q6ScheduleId;
    using V      = ops::detail::Q6KernelVariant;
    int failures = 0;

    failures += one_candidate(S::SimtR8C4, V::None, 128, 5120, 1, 11u);
    failures += one_candidate(S::SimtR8C4, V::None, 128, 1536, 4, 13u);
    failures += one_candidate(S::SimtR8C4, V::None, 128, 5120, 6, 17u);
    failures += one_candidate(S::SimtR8C4, V::None, 128, 1536, 36, 19u);

    failures += one_candidate(S::MmaR64C64, V::Full, 128, 1536, 64, 23u);
    failures += one_candidate(S::MmaR64C64, V::Predicated, 144, 1536, 65, 29u);
    failures += one_candidate(S::MmaR64C128, V::Full, 128, 1536, 128, 31u);
    failures += one_candidate(S::MmaR64C128, V::Predicated, 144, 1536, 129, 37u);

    failures += legality_contract_rejections();

    std::cout << (failures ? "FAIL" : "OK") << " Q6 Linear fixed candidates\n";
    return failures ? 1 : 0;
}
