#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5AttnInputScheduleId schedule;
};

// Measured on an RTX 3090 (sm_86) with bench/ops/q4_q5_attn_input_schedule_bench.cu, which times
// every schedule at the same column count. Medians of 9-11, cold, us:
//
//     T      parent  grp_c32  grp_c64  mix_c64  pair_c64  mix_c128
//     8       191.5    230.4    392.1    286.6     286.7     458.8
//     9       333.8    231.4    392.2    287.7     288.8     459.8
//    32           -    223.2    401.3    293.9     295.9     458.8
//    34           -    384.0    352.3    257.0     258.0     398.1
//    64           -    417.8    372.7    301.1     305.2     464.9
//    68           -    577.5    586.8    458.8     507.9     416.8
//   124           -    742.4    615.4    478.2     525.3     427.0
//   128           -    678.9    567.3    475.1     505.8     526.3
//   192           -   1027.1    867.1    716.8     742.4     740.4
//   208           -   1340.4   1137.7    924.5     973.8     748.5
//
// parent_split_fixed is only defined for cols <= 12 (q4_q5_attn_input_small_t.cu) but already
// loses to the 32-wide grouped tile at 9.
//
// grouped_r32_c64_s4 and pair_r32_c64_s3 win at no width, and that is now measured rather than
// assumed. The table above covers 8..208; extending the sweep to 4096 settles the rest, us:
//
//     T           256    512   1024   2048   4096
//     mixed_c128  854   1605   2900   6000  11296     <- wins at every width
//     grouped_c64 1159  2149   4239   8449  16918     1.4-1.5x the winner
//     pair_c64    1027  1908   3750   7400  14952     always just behind mixed_r32_c64_s3
//
// pair_r32_c64_s3 is the interesting one: it tracks mixed_r32_c64_s3 to within a couple of percent
// at every width and is never ahead of it, so it is not a different trade-off, it is a slower
// twin. PairR32C64S4 is not a third kernel at all -- it dispatches to the same launch as
// grouped_r32_c64_s4 (see the execute switch), so those two share a measurement.
//
// All three stay in the enum. Deleting an upstream schedule costs merge effort at every future
// catch-up for no measured gain here, and the reason to worry about dead schedules -- they are
// where the two switch fallthroughs hid -- is now covered by tests/ops/test_route_coverage.cpp,
// which enumerates the unrouted set and fails if its membership changes.
//
// 2026-09-11, RTX 3090 under Linux: SmallTMma -- the Q4 and Q5 small-T MMA kernels over query_key
// and gate_value -- replaces ParentSplitFixed across 1..8, the T=1 GEMVs included. Cold, median
// of 31 (us):
//
//   T                  1      2      3      4      5      6      7      8
//   parent_split   81.9   93.2   94.2   90.1  164.9  174.2  215.0  161.8
//   small_t        69.6   69.6   68.6   68.6   66.6   66.7   68.6   69.6
//
// At T=4 that is 71% of the 48.8 us both weights cost to stream once, against 54%.
//
// Its 16- and 32-column tiles against the r32/c32 tile (us):
//
//   T              9     12     16     20     24     28     31     32
//   small_t     94.2   93.2  109.6  145.4  162.8  183.3  194.6  200.7
//   r32/c32    209.9  194.6  195.6  199.7  199.7  200.7  201.7  190.5
//
// small_t won through 31; at exactly 32 columns the grouped tile was full and won by 5%. Once its
// 16- and 32-column tiles shared each staged activation slab between two to four row tiles
// (ops/common/small_t_layout.cuh), same bench, median min..p95 of 31 (us):
//
//   T              9           16           17           24           32
//   small_t     87.0  84..88   90.1  85..92 132.1 131..133 133.1 131..134 135.2 134..136
//   r32/c32    209.9 209..211 197.6 197..199 201.7 200..203 201.7 201..203 191.5 189..193
//
// so it takes all of 1..32 and the r32/c32 tile serves nothing.
constexpr std::array<RouteSpec, 5> kRoutes{{
    {{1, 32}, Q4Q5AttnInputScheduleId::SmallTMma},
    {{33, 64}, Q4Q5AttnInputScheduleId::MixedR32C64S3},
    {{65, 127}, Q4Q5AttnInputScheduleId::MixedR64C128S2},
    {{128, 192}, Q4Q5AttnInputScheduleId::MixedR32C64S3},
    {{193, kAnyCols}, Q4Q5AttnInputScheduleId::MixedR64C128S2},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes[kRoutes.size() - 1].cols.last == kAnyCols;
}

static_assert(catalog_is_closed(), "attention input routes must be exact and closed");

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C32S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c32.s4";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        return "attn_input_proj.q4_q5.mixed.r32.c32.s2";
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        return "attn_input_proj.q4_q5.mixed.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        return "attn_input_proj.q4_q5.pair.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        return "attn_input_proj.q4_q5.mixed.r64.c128.s2";
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        return "attn_input_proj.q4_q5.pair.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::SmallTMma:
        return "attn_input_proj.q4_q5.small_t.mma";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    // One source of truth: the table above. It used to sit here as dead code next to a hardcoded
    // if-chain carrying upstream's sm_120 boundaries, so the tuned table was never consulted.
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.contains(problem.cols)) { return {route.schedule}; }
    }
    throw std::invalid_argument("Q4/Q5 attention input: column count is not covered by any route");
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C32S4:
        q4_q5_attn_input_grouped_mma_r32_c32_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        q4_q5_attn_input_mixed_r32_c32_s2_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        q4_q5_attn_input_mixed_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        q4_q5_attn_input_pair_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        q4_q5_attn_input_mixed_r64_c128_s2_launch(x, query_key_weight, gate_value_weight, q, gate,
                                                  k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::SmallTMma:
        q4_q5_attn_input_small_t_mma_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                            stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  stream);
}

} // namespace ninfer::ops::detail
