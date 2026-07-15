#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/scatter.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int test_add_bias(std::int32_t d, std::int32_t tokens, std::uint32_t seed) {
    const std::size_t n = static_cast<std::size_t>(d) * tokens;
    std::vector<float> x(n), bias(d);
    fill_uniform(x, seed, -8.0f, 8.0f);
    fill_uniform(bias, seed + 1, -2.0f, 2.0f);
    round_to_bf16(x);
    round_to_bf16(bias);
    std::vector<double> reference(n);
    for (std::size_t i = 0; i < n; ++i) {
        reference[i] = static_cast<double>(x[i]) + bias[i % static_cast<std::size_t>(d)];
    }
    DBuf dx = to_device_bf16(x);
    DBuf db = to_device_bf16(bias);
    Tensor tx(dx.p, DType::BF16, {d, tokens});
    Tensor tb(db.p, DType::BF16, {d});
    ops::add_bias(tb, tx, nullptr);
    cudaDeviceSynchronize();
    return verify("vision add_bias", from_device_bf16(dx, n), reference,
                  Tolerance::bf16_elementwise());
}

int test_add_bias_unaligned() {
    constexpr std::int32_t d      = 8;
    constexpr std::int32_t tokens = 3;
    constexpr std::size_t n       = static_cast<std::size_t>(d) * tokens;
    std::vector<float> x(n), bias(d);
    fill_uniform(x, 2026u, -8.0f, 8.0f);
    fill_uniform(bias, 2027u, -2.0f, 2.0f);
    round_to_bf16(x);
    round_to_bf16(bias);
    std::vector<double> reference(n);
    for (std::size_t i = 0; i < n; ++i) {
        reference[i] = static_cast<double>(x[i]) + bias[i % static_cast<std::size_t>(d)];
    }

    std::vector<std::uint16_t> packed_x(n + 1), packed_bias(d + 1);
    for (std::size_t i = 0; i < n; ++i) packed_x[i + 1] = f32_to_bf16(x[i]);
    for (std::size_t i = 0; i < bias.size(); ++i) packed_bias[i + 1] = f32_to_bf16(bias[i]);
    DBuf dx(packed_x.size() * sizeof(std::uint16_t));
    DBuf db(packed_bias.size() * sizeof(std::uint16_t));
    cudaMemcpy(dx.p, packed_x.data(), dx.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db.p, packed_bias.data(), db.bytes, cudaMemcpyHostToDevice);
    auto* xptr = static_cast<unsigned char*>(dx.p) + sizeof(std::uint16_t);
    auto* bptr = static_cast<unsigned char*>(db.p) + sizeof(std::uint16_t);
    Tensor tx(xptr, DType::BF16, {d, tokens});
    Tensor tb(bptr, DType::BF16, {d});
    ops::add_bias(tb, tx, nullptr);
    cudaDeviceSynchronize();

    DBuf out(n * sizeof(std::uint16_t));
    cudaMemcpy(out.p, xptr, out.bytes, cudaMemcpyDeviceToDevice);
    return verify("vision add_bias unaligned", from_device_bf16(out, n), reference,
                  Tolerance::bf16_elementwise());
}

double gelu_reference(double x, ops::GeluMode mode) {
    if (mode == ops::GeluMode::Tanh) {
        constexpr double root = 0.79788456080286535588;
        return 0.5 * x * (1.0 + std::tanh(root * (x + 0.044715 * x * x * x)));
    }
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

int test_gelu(ops::GeluMode mode, std::int32_t d, std::int32_t tokens, std::uint32_t seed) {
    const std::size_t n = static_cast<std::size_t>(d) * tokens;
    std::vector<float> x(n);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(x);
    std::vector<double> reference(n);
    for (std::size_t i = 0; i < n; ++i) reference[i] = gelu_reference(x[i], mode);
    DBuf dx = to_device_bf16(x);
    Tensor tx(dx.p, DType::BF16, {d, tokens});
    ops::gelu(tx, mode, nullptr);
    cudaDeviceSynchronize();
    return verify(mode == ops::GeluMode::Tanh ? "vision gelu tanh" : "vision gelu exact",
                  from_device_bf16(dx, n), reference, Tolerance::bf16_elementwise());
}

int test_scatter(std::uint32_t seed) {
    constexpr std::int32_t d = 5120;
    constexpr std::int32_t v = 64;
    constexpr std::int32_t t = 129;
    std::vector<float> src(static_cast<std::size_t>(d) * v);
    std::vector<float> dst(static_cast<std::size_t>(d) * t);
    std::vector<int> indices(v);
    for (int i = 0; i < v; ++i) indices[i] = 1 + i * 2;
    fill_uniform(src, seed, -8.0f, 8.0f);
    fill_uniform(dst, seed + 1, -8.0f, 8.0f);
    round_to_bf16(src);
    round_to_bf16(dst);
    std::vector<double> reference(dst.begin(), dst.end());
    for (int col = 0; col < v; ++col) {
        for (int row = 0; row < d; ++row) {
            reference[static_cast<std::size_t>(indices[col]) * d + row] =
                src[static_cast<std::size_t>(col) * d + row];
        }
    }
    DBuf dsrc = to_device_bf16(src);
    DBuf ddst = to_device_bf16(dst);
    DBuf didx = to_device_i32(indices);
    Tensor tsrc(dsrc.p, DType::BF16, {d, v});
    Tensor tdst(ddst.p, DType::BF16, {d, t});
    Tensor tidx(didx.p, DType::I32, {v});
    ops::scatter(tsrc, tidx, tdst, nullptr);
    cudaDeviceSynchronize();
    return verify("vision scatter", from_device_bf16(ddst, reference.size()), reference,
                  Tolerance::bf16_elementwise());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int failures = 0;
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        failures += test_add_bias(1152, 256, seed);
        failures += test_add_bias(3456, 256, seed + 10);
        failures += test_add_bias(4304, 256, seed + 11);
        failures += test_add_bias(4608, 64, seed + 12);
        failures += test_add_bias(2048, 64, seed + 13);
        failures += test_add_bias(10, 3, seed + 14);
        failures += test_add_bias(1153, 3, seed + 15);
        failures += test_add_bias(5120, 1, seed + 20);
        failures += test_gelu(ops::GeluMode::Tanh, 4304, 8, seed + 30);
        failures += test_gelu(ops::GeluMode::Tanh, 4304, 256, seed + 31);
        failures += test_gelu(ops::GeluMode::Exact, 4608, 2, seed + 40);
        failures += test_gelu(ops::GeluMode::Exact, 4608, 64, seed + 41);
        failures += test_scatter(seed + 50);
    }
    failures += test_add_bias(3456, 4096, 2025u);
    failures += test_add_bias_unaligned();
    failures += test_gelu(ops::GeluMode::Tanh, 4304, 4096, 2026u);
    failures += test_gelu(ops::GeluMode::Exact, 4608, 1024, 2027u);
    std::cout << (failures ? "FAIL" : "OK") << " vision elementwise correctness\n";
    return failures ? 1 : 0;
}
