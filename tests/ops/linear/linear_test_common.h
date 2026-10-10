#pragma once

#include "ninfer/ops/linear.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::test::linear {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
};

enum class CallForm : std::uint8_t {
    Policy,
    A16Convenience,
};

// Which output rows the FP64 oracle covers. Every column is always compared, in both cases. An
// error that is a property of the contraction -- a dequantization that rounds, a dropped K tile, a
// wrapped scale stride -- reaches one column as readily as the next, so sampling columns bought
// nothing but a noisier reading of the same ratio and a cap of 1,024 elements however wide the
// call. It is also not the more expensive choice: the oracle depends on the token index alone, so
// it is evaluated once per shape at the widest invocation and sliced, where the thirty-two-column
// version was recomputed per invocation.
enum class Comparison : std::uint8_t {
    Full,
    SampledRows,
};

struct Invocation {
    std::int32_t t;
    CallForm call_form       = CallForm::Policy;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    bool graph_replay        = false;
};

// The sign distribution of the activation. `Centered` draws from [-0.5, 0.5); `Biased` draws from
// [0.125, 0.496], every element the same sign, as a SwiGLU or a softmax output is. A centered
// activation lets a per-weight error cancel along K at the same rate as the dot product it
// perturbs; a one-sign activation does not, so it is the harsher of the two on a weight decode that
// rounds. Measured on the Q8 MMA tiles' `round_bf16(code * scale)`, the two agree within 0.02 of
// the criterion on nine of the ten Q8 shapes the shipped artifact carries, and on the tenth
// ([14336, 5120]) `Biased` reads 0.75 against `Centered`'s 0.60 -- the difference between a quarter
// of the budget in hand and two fifths.
//
// The `linear_add` suite, whose fixture is one-sign, measured 1.15 for the same rounding at
// [5120, 17408], where both of these read 0.60. So how much a decode that rounds costs is a
// property of the fixture's code-and-activation pairing and not of K alone; run the shapes that
// matter both ways rather than reasoning from one number.
enum class ActivationSigns : std::uint8_t {
    Centered,
    Biased,
};

// A structured activation that replaces the random one. `KTail` is zero except the final sixteen
// columns, so only a K-tail tile contributes; `Cancellation` alternates +-0.25 so a dropped or
// duplicated K element shows up against a near-zero exact result.
enum class ActivationPattern : std::uint8_t { Dense, KTail, Cancellation };

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    Comparison comparison;
    bool verify_input_preservation;
    std::span<const Invocation> invocations;
    ActivationPattern activation_pattern = ActivationPattern::Dense;
    // Applies to the Dense pattern only; a structured pattern fixes every element.
    ActivationSigns activation_signs = ActivationSigns::Centered;
};

using WeightGenerator = quantized_weight::PackedWeight (*)(std::int32_t, std::int32_t,
                                                           std::uint32_t);

quantized_weight::PackedWeight make_bf16_weight(std::int32_t n, std::int32_t k, std::uint32_t seed);

quantized_weight::PackedWeight make_q4_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q5_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q6_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q8_g32_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed);
quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed);

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t);

bool cuda_available();

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape);

// A declared token interval must reserve enough space for every public point it contains.
int verify_workspace_envelopes(QType qtype, std::int32_t n, std::int32_t k);

} // namespace ninfer::test::linear
