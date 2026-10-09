#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <vector>

namespace {
using namespace ninfer::test::linear;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

constexpr std::array kGeometries{
    Geometry{1024, 2048, 257U},  Geometry{1024, 5120, 223U},  Geometry{2048, 4096, 251U},
    Geometry{2048, 4608, 271U},  Geometry{2048, 16384, 283U}, Geometry{4608, 4608, 277U},
    Geometry{5120, 4608, 281U},  Geometry{5120, 6144, 239U},  Geometry{5120, 10240, 211U},
    Geometry{5120, 17408, 241U}, Geometry{5120, 25600, 293U}, Geometry{6144, 5120, 227U},
    Geometry{9216, 2048, 263U},  Geometry{12288, 2048, 269U}, Geometry{14336, 5120, 229U},
    Geometry{34816, 5120, 233U}, Geometry{248320, 5120, 197U}};

int q8_a16_conformance() {
    int failures = 0;
    for (const auto& shape : kGeometries) {
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 159, 160,
                      161, 191, 192, 193, 256, 1024}) {
            calls.push_back({t});
        }
        if (shape.n == 2048 && shape.k == 4096) {
            for (int t : {895, 896, 897}) calls.push_back({t});
        }
        if (shape.n == 6144 && shape.k == 5120) {
            for (int t : {191, 192, 193}) calls.push_back({t});
        }
        if (shape.k == 4608) {
            for (int t : {6, 11, 12, 13, 14, 19, 20, 21, 27, 28, 29}) calls.push_back({t});
            if (shape.n == 2048) {
                for (int t : {870, 871, 872}) calls.push_back({t});
            }
            if (shape.n == 4608) {
                for (int t : {255, 257}) calls.push_back({t});
            }
        }
        if (shape.n == 9216 && shape.k == 2048) {
            for (int t : {12, 13, 14}) calls.push_back({t});
        }
        if (shape.n == 248320) calls.push_back({34});
        if (shape.n == 2048 && shape.k == 16384) {
            for (int t : {383,  384,  385,  479,  480,  481,  639,  640,  641,  703,
                          704,  705,  959,  960,  961,  1343, 1344, 1345, 1679, 1680,
                          1681, 2015, 2016, 2017, 2111, 2112, 2113, 4096}) {
                calls.push_back({t});
            }
        }
        for (int t : {1, 8, 16, 32, 48, 64, 65, 128}) {
            calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        for (int t : {1, 16, 64, 128}) calls.push_back({t, CallForm::A16Convenience});
        calls.push_back({17, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA8});
        calls.push_back({64, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA4});
        failures += run_shape("Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                              {shape.n, shape.k, shape.seed, Comparison::SampledRows, true, calls});
    }
    return failures;
}

// Every shape the shipped Qwen3.8-27B v3 artifact actually stores as `q8_g32_fp16`, read off its
// object records (thirty tensors, ten distinct shapes). Nine of them are here; `dflash2/layers/*/
// attention/output` at [5120, 4096] has no shape table of its own and is left to the generic
// dispatch the geometries above already cover.
//
//   [248320, 5120]  text/output_head, text/token_embedding -- the only ones on the target model's
//                   own output path, so the only ones whose error is not absorbed by verification
//   [5120, 25600]   dflash2/feature_projection
//   [5120, 17408]   dflash2/layers/{0..4}/mlp/down, mtp/layer/mlp/down
//   [34816, 5120]   dflash2/layers/{0..4}/mlp/gate_up, mtp/layer/mlp/gate_up
//   [6144, 5120]    dflash2/layers/{0..4}/attention/query_key_value
//   [14336, 5120]   mtp/layer/attention/query_key_gate_value
//   [5120, 10240]   mtp/input_projection
//   [5120, 6144]    mtp/layer/attention/output
//   [4608, 4608]    vision/merger/fc1        [5120, 4608] vision/merger/fc2
//
// Every other Q8 geometry above is a shape the dispatcher admits rather than one an artifact
// carries. These nine are run a second time against an all-one-sign activation, because a centered
// one cannot tell a weight decode that rounds from one that does not: see ActivationSigns. The
// widths are each table's tiled bands plus the K-split rung below the first of them, so a step at a
// route boundary shows up as a step rather than as a single number.
constexpr std::array kBiasedGeometries{
    Geometry{4608, 4608, 307U},   Geometry{5120, 4608, 311U},  Geometry{5120, 6144, 313U},
    Geometry{5120, 10240, 317U},  Geometry{5120, 17408, 331U}, Geometry{5120, 25600, 337U},
    Geometry{6144, 5120, 347U},   Geometry{14336, 5120, 349U}, Geometry{34816, 5120, 353U},
    Geometry{248320, 5120, 359U}};

int q8_a16_biased_activation() {
    int failures = 0;
    for (const auto& shape : kBiasedGeometries) {
        std::vector<Invocation> calls;
        for (int t : {1,  8,   16,  24,  32,  40,  48,  49,  56,  64,
                      65, 80,  96,  112, 128, 129, 160, 192, 193, 256,
                      512, 1024}) {
            calls.push_back({t});
        }
        failures += run_shape("Q8_A16_biased", ActivationCompute::A16, make_q8_g32_fp16_weight,
                              {shape.n, shape.k, shape.seed, Comparison::SampledRows, false, calls,
                               ActivationSigns::Biased});
    }
    return failures;
}
} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = q8_a16_conformance() + q8_a16_biased_activation();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
