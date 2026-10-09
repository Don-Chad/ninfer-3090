// The cuBLAS prefill route against the integer-activation route it would replace, timed and
// compared on the same inputs. A fast wrong GEMM is the main hazard of a route whose whole point is
// that it computes something slightly different, so `relL2` is printed beside every speedup.
//
// That column is a *difference between two approximations*, not an error against exact arithmetic:
// the A8Int route carries per-group activation scales, this route carries per-row weight scales and
// per-token activation ones. It is the size of the trade, not a correctness bound -- a few times
// 1e-2 is the expected disagreement, and only a wildly larger number means a bug. The quality
// question this cannot answer is perplexity. See ops/linear_swiglu/q4cublas/w4_cublas_prefill.h.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer_bench_common.h"
#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;

constexpr double kInt8CeilingTops   = 314.8;
constexpr std::size_t kFlushBytes   = 256ULL << 20;

struct Profile {
    const char* name;
    QType qtype;
    std::int32_t rows;
    std::int32_t cols;
    std::int32_t out_rows;
    bool swiglu;
};

constexpr Profile kProfiles[] = {
    {"mlp/gate_up", QType::Q4_G64_FP16, 34816, 5120, 17408, true},
    {"mlp/down", QType::Q5_G64_FP16, 5120, 17408, 5120, false},
    {"out_proj", QType::Q5_G64_FP16, 5120, 6144, 5120, false},
};

double tops(const Profile& p, std::int32_t tokens, double us) {
    return 2.0 * p.rows * p.cols * tokens / (us * 1.0e-6) / 1.0e12;
}

// The activation buffer has to carry real values: with a zeroed input both routes emit zeros, the
// norm is zero, and the comparison below reports a perfect match while checking nothing.
__global__ void fill_activations(__nv_bfloat16* x, std::size_t count, unsigned seed) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    unsigned h = seed ^ static_cast<unsigned>(i * 2654435761u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    // Roughly N(0, 1)-ish in [-2, 2); the absolute distribution does not matter, only that it is
    // not degenerate and that both arms see the same thing.
    x[i] = __float2bfloat16((static_cast<float>(h & 0xffffu) / 16384.0F) - 2.0F);
}

double relative_l2(const void* a, const void* b, std::size_t count) {
    std::vector<__nv_bfloat16> ha(count);
    std::vector<__nv_bfloat16> hb(count);
    cudaMemcpy(ha.data(), a, count * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, count * 2, cudaMemcpyDeviceToHost);
    double error = 0.0;
    double norm  = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double x = __bfloat162float(ha[i]);
        const double y = __bfloat162float(hb[i]);
        error += (x - y) * (x - y);
        norm += x * x;
    }
    return norm > 0.0 ? std::sqrt(error / norm) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::int32_t> tokens{1024, 4096};
    int repeat = 5;
    int warmup = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--tokens" && i + 1 < argc) {
            tokens.clear();
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
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T,...] [--repeat N] [--warmup N]\n", argv[0]);
            return 2;
        }
    }

    const std::int32_t max_tokens = *std::max_element(tokens.begin(), tokens.end());
    ninfer::bench::L2FlushBuffer flush(kFlushBytes);
    cudaStream_t stream = nullptr;
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("# gpu=%s  cuBLAS prefill route vs the integer-activation route, cold, median of %d\n",
                properties.name, repeat);
    std::printf("# ceiling %.1f TOP/s\n", kInt8CeilingTops);
    std::printf("%-12s %6s %12s %12s %10s %10s %8s %10s\n", "profile", "T", "a8int us", "cublas us",
                "a8 TOP/s", "cbl TOP/s", "speedup", "relL2");

    for (const Profile& profile : kProfiles) {
        ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
            profile.qtype, profile.rows, profile.cols, profile.cols, 0x52U);
        ninfer::DeviceBuffer input(static_cast<std::size_t>(profile.cols) * max_tokens * 2);
        ninfer::DeviceBuffer output(static_cast<std::size_t>(profile.out_rows) * max_tokens * 2);
        ninfer::DeviceBuffer reference(static_cast<std::size_t>(profile.out_rows) * max_tokens * 2);
        {
            const std::size_t count =
                static_cast<std::size_t>(profile.cols) * max_tokens;
            fill_activations<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
                static_cast<__nv_bfloat16*>(input.p), count, 0x9e3779b9u);
            cudaDeviceSynchronize();
        }

        for (const std::int32_t t : tokens) {
            Tensor x(input.p, DType::BF16, {profile.cols, t});
            Tensor out(output.p, DType::BF16, {profile.out_rows, t});

            double a8_us = 0.0;
            try {
                const auto capacity =
                    profile.swiglu ? ninfer::ops::linear_swiglu_workspace_capacity_bytes(
                                         profile.qtype, profile.rows, profile.cols,
                                         ninfer::ops::LinearPolicy::AllowA8Int, t, t)
                                   : ninfer::ops::linear_add_workspace_capacity_bytes(
                                         profile.qtype, profile.out_rows, profile.cols,
                                         ninfer::ops::LinearPolicy::AllowA8Int, t, t);
                ninfer::WorkspaceArena ws(std::max<std::size_t>(capacity, 1));
                const auto invoke = [&](cudaStream_t s) {
                    if (profile.swiglu) {
                        ninfer::ops::linear_swiglu(x, packed.weight, out, ninfer::ops::LinearPolicy::AllowA8Int,
                                                   ws, s);
                    } else {
                        ninfer::ops::linear_add(x, packed.weight, out, ninfer::ops::LinearPolicy::AllowA8Int, ws,
                                                s);
                    }
                };
                a8_us = ninfer::bench::measure_cold_launch(invoke, flush, stream, warmup, repeat)
                            .median_us;
            } catch (const std::exception&) {
                cudaGetLastError();
            }

            double cublas_us = 0.0;
            try {
                const auto capacity =
                    ninfer::ops::detail::w4_cublas_prefill_workspace_capacity_bytes(
                        profile.rows, profile.cols, t, t);
                ninfer::WorkspaceArena ws(capacity);
                const auto invoke = [&](cudaStream_t s) {
                    if (profile.swiglu) {
                        ninfer::ops::detail::w4_cublas_swiglu_launch(x, packed.weight, out, ws, s);
                    } else {
                        ninfer::ops::detail::w4_cublas_add_launch(x, packed.weight, out, ws, s);
                    }
                };
                cublas_us = ninfer::bench::measure_cold_launch(invoke, flush, stream, warmup, repeat)
                                .median_us;
            } catch (const std::exception& error) {
                cudaGetLastError();
                std::printf("%-12s %6d  cublas route unavailable: %s\n", profile.name, t,
                            error.what());
                continue;
            }

            // Re-run both arms once into distinct buffers, so the comparison sees exactly the
            // inputs the timings did. linear_add accumulates, so both start from a zeroed residual.
            double rel = -1.0;
            try {
                Tensor ref(reference.p, DType::BF16, {profile.out_rows, t});
                const auto a8_capacity =
                    profile.swiglu ? ninfer::ops::linear_swiglu_workspace_capacity_bytes(
                                         profile.qtype, profile.rows, profile.cols,
                                         ninfer::ops::LinearPolicy::AllowA8Int, t, t)
                                   : ninfer::ops::linear_add_workspace_capacity_bytes(
                                         profile.qtype, profile.out_rows, profile.cols,
                                         ninfer::ops::LinearPolicy::AllowA8Int, t, t);
                ninfer::WorkspaceArena a8_ws(std::max<std::size_t>(a8_capacity, 1));
                ninfer::WorkspaceArena cb_ws(
                    ninfer::ops::detail::w4_cublas_prefill_workspace_capacity_bytes(
                        profile.rows, profile.cols, t, t));
                const std::size_t elements = static_cast<std::size_t>(profile.out_rows) * t;
                cudaMemset(ref.data, 0, elements * 2);
                cudaMemset(out.data, 0, elements * 2);
                if (profile.swiglu) {
                    ninfer::ops::linear_swiglu(x, packed.weight, ref,
                                               ninfer::ops::LinearPolicy::AllowA8Int, a8_ws,
                                               nullptr);
                    ninfer::ops::detail::w4_cublas_swiglu_launch(x, packed.weight, out, cb_ws,
                                                                 nullptr);
                } else {
                    ninfer::ops::linear_add(x, packed.weight, ref,
                                            ninfer::ops::LinearPolicy::AllowA8Int, a8_ws, nullptr);
                    ninfer::ops::detail::w4_cublas_add_launch(x, packed.weight, out, cb_ws, nullptr);
                }
                cudaDeviceSynchronize();
                rel = relative_l2(out.data, ref.data, elements);
            } catch (const std::exception&) {
                cudaGetLastError();
            }

            std::printf("%-12s %6d %12.1f %12.1f %10.1f %10.1f %7.2fx %10.3e\n", profile.name, t,
                        a8_us, cublas_us, a8_us > 0 ? tops(profile, t, a8_us) : 0.0,
                        tops(profile, t, cublas_us),
                        a8_us > 0 && cublas_us > 0 ? a8_us / cublas_us : 0.0, rel);
        }
    }
    return 0;
}
