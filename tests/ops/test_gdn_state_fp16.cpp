// FP16 recurrent-state storage against FP32 over a long decode, through the public
// gated_delta_net Op. Both chains see identical inputs; the FP16 chain rounds its stored state
// once per call. Reports the output error per step (relative L2 over all heads) and fails if it
// exceeds 2e-2 -- five bf16 rounding steps -- anywhere, or if it trends upward (the last quarter's
// mean above twice the second quarter's).
//
// Inputs mimic the 27B's layer statistics loosely: 16 query/key heads and 48 value heads,
// normalized q/k, per-head decays from short (alpha 0.9) to near-lossless memory (alpha 0.9999),
// and beta in (0, 1).

#include "ninfer/ops/gated_delta_net.h"
#include "core/arena.h"
#include "ops/op_tester.h"
#include "ops/small_t_oracle.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::Tensor;
namespace oracle = ninfer::test::small_t_oracle;

constexpr int kQk    = 16;
constexpr int kHv    = 48;
constexpr int kDim   = 128;
constexpr int kSteps = 4096;

} // namespace

int main(int argc, char** argv) {
    const int width = argc > 1 ? std::atoi(argv[1]) : 1; // tokens per call: 1 = decode, 4 = MTP3
    try {
        std::mt19937 rng(0x5eed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::vector<std::uint16_t> q(static_cast<std::size_t>(kDim) * kQk * kSteps);
        std::vector<std::uint16_t> k(q.size());
        std::vector<std::uint16_t> v(static_cast<std::size_t>(kDim) * kHv * kSteps);
        std::vector<float> g(static_cast<std::size_t>(kHv) * kSteps);
        std::vector<float> beta(g.size());
        for (auto& x : q) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (auto& x : k) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (auto& x : v) { x = oracle::f32_to_bf16_rne(normal(rng)); }
        for (int t = 0; t < kSteps; ++t) {
            for (int h = 0; h < kHv; ++h) {
                const float alpha = 1.0F - std::pow(10.0F, -1.0F - 3.0F * h / (kHv - 1));
                g[static_cast<std::size_t>(t) * kHv + h] =
                    std::log(alpha) * (1.0F + 0.1F * normal(rng));
                beta[static_cast<std::size_t>(t) * kHv + h] = 1.0F / (1.0F + std::exp(-normal(rng)));
            }
        }
        ninfer::test::GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2);
        ninfer::test::GuardedDeviceBuffer dg(g.size() * 4), db(beta.size() * 4);
        dq.copy_from_host(q.data(), q.size() * 2);
        dk.copy_from_host(k.data(), k.size() * 2);
        dv.copy_from_host(v.data(), v.size() * 2);
        dg.copy_from_host(g.data(), g.size() * 4);
        db.copy_from_host(beta.data(), beta.size() * 4);
        const std::size_t state_elems = static_cast<std::size_t>(kDim) * kDim * kHv;
        ninfer::test::GuardedDeviceBuffer s32(state_elems * 4), s16(state_elems * 2);
        s32.fill(0);
        s16.fill(0);
        const std::size_t out_bytes = static_cast<std::size_t>(kDim) * kHv * width * 2;
        ninfer::test::GuardedDeviceBuffer o32(out_bytes), o16(out_bytes);
        ninfer::WorkspaceArena ws(std::max<std::size_t>(
            256, ninfer::ops::gated_delta_net_workspace_capacity_bytes(kQk, kHv, width, width)));
        int device = 0;
        ninfer::test::cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp props{};
        ninfer::test::cuda_check(cudaGetDeviceProperties(&props, device),
                                 "cudaGetDeviceProperties");
        const ninfer::DeviceExecutionView execution{nullptr, props.multiProcessorCount};
        Tensor state32(s32.data(), DType::FP32, {kDim, kDim, kHv});
        Tensor state16(s16.data(), DType::FP16, {kDim, kDim, kHv});
        const float scale = 1.0F / std::sqrt(static_cast<float>(kDim));

        std::vector<std::uint16_t> h32(static_cast<std::size_t>(kDim) * kHv * width);
        std::vector<std::uint16_t> h16(h32.size());
        std::vector<double> err(kSteps / width);
        double worst = 0.0;
        for (int call = 0; call < kSteps / width; ++call) {
            const std::size_t t0 = static_cast<std::size_t>(call) * width;
            const auto slice     = [&](void* base, std::size_t elem, std::size_t per_token,
                                   DType dtype, std::initializer_list<std::int32_t> shape) {
                return Tensor(static_cast<std::uint8_t*>(base) + t0 * per_token * elem, dtype,
                              shape);
            };
            const Tensor tq = slice(dq.data(), 2, kDim * kQk, DType::BF16, {kDim, kQk, width});
            const Tensor tk = slice(dk.data(), 2, kDim * kQk, DType::BF16, {kDim, kQk, width});
            const Tensor tv = slice(dv.data(), 2, kDim * kHv, DType::BF16, {kDim, kHv, width});
            const Tensor tg = slice(dg.data(), 4, kHv, DType::FP32, {kHv, width});
            const Tensor tb = slice(db.data(), 4, kHv, DType::FP32, {kHv, width});
            Tensor out32(o32.data(), DType::BF16, {kDim, kHv, width});
            Tensor out16(o16.data(), DType::BF16, {kDim, kHv, width});
            ninfer::ops::gated_delta_net(tq, tk, tv, tg, tb, scale, true, ws, state32, out32,
                                         execution);
            ninfer::ops::gated_delta_net(tq, tk, tv, tg, tb, scale, true, ws, state16, out16,
                                         execution);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "gated_delta_net");
            o32.copy_to_host(h32.data(), h32.size() * 2);
            o16.copy_to_host(h16.data(), h16.size() * 2);
            double num = 0.0, den = 0.0;
            for (std::size_t i = 0; i < h32.size(); ++i) {
                const double a = oracle::bf16_to_f32(h32[i]);
                const double b = oracle::bf16_to_f32(h16[i]);
                num += (a - b) * (a - b);
                den += a * a;
            }
            err[call] = std::sqrt(num / std::max(den, 1e-30));
            worst     = std::max(worst, err[call]);
        }
        const auto mean = [&](int begin, int end) {
            double sum = 0.0;
            for (int i = begin; i < end; ++i) { sum += err[i]; }
            return sum / (end - begin);
        };
        const int calls = kSteps / width;
        const double q1 = mean(0, calls / 4), q2 = mean(calls / 4, calls / 2),
                     q4 = mean(3 * calls / 4, calls);
        std::cout << "width " << width << ": relative output error, mean by quarter " << q1 << ' '
                  << q2 << ' ' << mean(calls / 2, 3 * calls / 4) << ' ' << q4 << ", worst "
                  << worst << '\n';
        const bool ok = worst <= 2e-2 && q4 <= 2.0 * q2;
        std::cout << (ok ? "OK" : "FAIL") << " FP16 GDN state tracks FP32 over " << kSteps
                  << " tokens\n";
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP16 state test failed: " << error.what() << '\n';
        return 1;
    }
}
