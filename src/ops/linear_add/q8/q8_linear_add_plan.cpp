#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"

#include "ops/linear_add/q8/q8_linear_add_kernels.h"
#include "ops/common/token_slices.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8LinearAddScheduleId schedule;
};

constexpr std::array<RouteSpec, 5> kK4096Routes{{
    {1, 1, Q8LinearAddScheduleId::SimtR8C4},
    {2, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, Q8LinearAddScheduleId::MediumSplitK},
    {129, 640, Q8LinearAddScheduleId::MmaR32C128},
    {641, kAnyCols, Q8LinearAddScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 33> kK6144Routes{{
    {1, 1, Q8LinearAddScheduleId::DecodeR16},
    {2, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, Q8LinearAddScheduleId::MediumSplitK},
    {129, 191, Q8LinearAddScheduleId::MmaR32C128},
    {192, 192, Q8LinearAddScheduleId::MmaR32C96},
    {193, 256, Q8LinearAddScheduleId::MmaR32C128},
    {257, 384, Q8LinearAddScheduleId::MmaR32C64},
    {385, 399, Q8LinearAddScheduleId::MmaR32C96},
    {400, 400, Q8LinearAddScheduleId::MmaR32C80},
    {401, 447, Q8LinearAddScheduleId::MmaR32C96},
    {448, 448, Q8LinearAddScheduleId::MmaR32C64},
    {449, 480, Q8LinearAddScheduleId::MmaR32C96},
    {481, 640, Q8LinearAddScheduleId::MmaR32C128},
    {641, 672, Q8LinearAddScheduleId::MmaR48C96},
    {673, 704, Q8LinearAddScheduleId::MmaR48C64},
    {705, 784, Q8LinearAddScheduleId::MmaR48C112},
    {785, 896, Q8LinearAddScheduleId::MmaR48C128},
    {897, 960, Q8LinearAddScheduleId::MmaR64C96},
    {961, 1023, Q8LinearAddScheduleId::MmaR64C112},
    {1024, 1024, Q8LinearAddScheduleId::MmaR64C128},
    {1025, 1120, Q8LinearAddScheduleId::MmaR64C112},
    {1121, 1280, Q8LinearAddScheduleId::MmaR64C128},
    {1281, 1344, Q8LinearAddScheduleId::MmaR128C64},
    {1345, 1408, Q8LinearAddScheduleId::MmaR48C128},
    {1409, 1680, Q8LinearAddScheduleId::MmaR128C80},
    {1681, 1791, Q8LinearAddScheduleId::MmaR48C128},
    {1792, 1792, Q8LinearAddScheduleId::MmaR64C128},
    {1793, 1919, Q8LinearAddScheduleId::MmaR48C128},
    {1920, 1920, Q8LinearAddScheduleId::MmaR64C128},
    {1921, 2016, Q8LinearAddScheduleId::MmaR64C96},
    {2017, 2047, Q8LinearAddScheduleId::MmaR64C112},
    {2048, 2048, Q8LinearAddScheduleId::MmaR64C128},
    {2049, kAnyCols, Q8LinearAddScheduleId::MmaR64C128},
}};

// The dense (5120-row) tables. Upstream's was two entries -- K-split capacity to 64 columns, then
// the grouped split-K for everything above -- where the 2048-row tables above are thirty-three.
// Re-measured on sm_86 2026-09-17 with bench/ops/dense_linear_add_schedule_bench.cu, cold, median
// of 11, and the grouped route is roughly **2x slower than a plain MMA tile at every width it
// covers** on this card (us, chosen route vs what shipped):
//
//   k=6144   T=80  r32_c96 128.0 vs 222.2 (+74%)   T=96  139.3 vs 254.0 (+82%)
//            T=112 r32_c128 152.6 vs 287.7 (+89%)  T=128 ~166 vs 321.5 (+94%)
//            T=160 r64_c96 215.0 vs 449.5 (+109%)  T=192 206.8 vs 485.4 (+135%)
//            T=224 r64_c112 262.1 vs 512.0 (+95%)  T=256 r64_c128 263.2 vs 551.9 (+110%)
//            T=512 516.1 vs 1054.7 (+104%)         T=1024 1013.8 vs 2262.0 (+123%)
//   k=17408  the same tiles measure 1.6-2.5x and did not pass the Op's oracle at that K until the
//            scale moved off the weight; the note on that table has the reason and the retune.
//
// The K-split capacity route keeps the narrow end, where it is genuinely the best thing available,
// but its ceiling is not 64 on this card: at k=6144 a 32x64 tile is already 19-62% faster from 33
// columns up. The two k therefore get different tables, as the 2048-row side already does.
//
// Read the bench's own header before extending this: the decode, exact-T and medium split-K
// launches are 2048-row kernels and produce a confident 2-8x "win" here by computing 2048 of the
// 5120 rows. They are not candidates at this shape and are not offered by the bench.
//
// T=128 is its own entry because the 128-wide tile loses there and only there: re-measured
// 2026-09-18 with the row-tile predicate fixed, `mma_r32_c128` 170-173 us against `mma_r64_c64`
// 150 us, twice, while at 112..124 the order is the other way round (154 vs 160, 162 vs 163) and
// the two are inside each other's spread from 116 up. Single-width entries are how the 2048-row
// table above already spells this shape of boundary.
constexpr std::array<RouteSpec, 8> kN5120K6144Routes{{
    {1, 32, Q8LinearAddScheduleId::SplitKMmaCapacity},
    {33, 64, Q8LinearAddScheduleId::MmaR32C64},
    {65, 96, Q8LinearAddScheduleId::MmaR32C96},
    {97, 127, Q8LinearAddScheduleId::MmaR32C128},
    {128, 128, Q8LinearAddScheduleId::MmaR64C64},
    {129, 192, Q8LinearAddScheduleId::MmaR64C96},
    {193, 224, Q8LinearAddScheduleId::MmaR64C112},
    {225, kAnyCols, Q8LinearAddScheduleId::MmaR64C128},
}};

// k=17408 shipped as two entries -- K-split capacity to 64 columns, the grouped split-K above --
// and the 2026-09-17 sweep found every MMA tile 1.6-2.5x faster here and every one of them failing
// the Op's oracle. The failure was real and it was the default tile's dequantization, not a bug in
// the tiling: the old Q8 MMA tile (now q8_a16_mma.cuh) folded the Q8G32 scale into the BF16 weight, and
// `round_bf16(code * scale)` throws away up to eleven bits of every weight. That error does not
// average out over K, so at K = 17408 it spends 1.15 of the relative-L2 criterion where K = 6144
// spends 0.57. The K-split and grouped routes never had it: they hold the bare code, which BF16
// represents exactly, and scale the FP32 group partial. That is why this table was right as it
// shipped, and it is the whole reason.
//
// The tiles now offer the same arrangement (`with_exact_group_scale`), which puts them at 0.42 of
// the criterion -- the K-split routes' own accuracy -- so this table can finally take them.
//
// Swept 2026-09-18 on sm_86 with bench/ops/dense_linear_add_schedule_bench.cu, cold, median of 15,
// against the grouped route this replaces (us):
//
//   T=65    398 vs  681 (1.71x)   T=128   445 vs  829 (1.86x)   T=192   578 vs 1317 (2.28x)
//   T=224   801 vs 1419 (1.77x)   T=256   887 vs 1551 (1.75x)   T=512  1755 vs 3227 (1.84x)
//   T=1024 3648 vs 6913 (1.89x)
//
// The capacity route keeps 1..40, not 1..64: it is 286 us at T=40 against the tile's 324, and 335
// against 326 at T=41, crossing between them. Above 256 the pick is `r64_c64` because it is the
// one candidate that never loses badly across 320/448/512/1024 (best at three of the four, +12 %
// at T=768 where `r64_c96` lands on a whole number of column tiles); `r64_c96` is +41 % at T=256
// and +17 % at T=320, so it is not the safer default it looks like at T=384.
constexpr std::array<RouteSpec, 8> kN5120K17408Routes{{
    {1, 40, Q8LinearAddScheduleId::SplitKMmaCapacity},
    {41, 64, Q8LinearAddScheduleId::MmaExactR32C64},
    {65, 96, Q8LinearAddScheduleId::MmaExactR32C96},
    {97, 128, Q8LinearAddScheduleId::MmaExactR64C64},
    {129, 192, Q8LinearAddScheduleId::MmaExactR64C96},
    {193, 224, Q8LinearAddScheduleId::MmaExactR64C112},
    {225, 256, Q8LinearAddScheduleId::MmaExactR64C128},
    {257, kAnyCols, Q8LinearAddScheduleId::MmaExactR64C64},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kK4096Routes) && routes_are_closed(kK6144Routes) &&
                  routes_are_closed(kN5120K6144Routes) &&
                  routes_are_closed(kN5120K17408Routes),
              "Q8 LinearAdd routes must be exact, contiguous, and closed");

std::int32_t schedule_rows(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
    case Q8LinearAddScheduleId::MediumSplitK:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 8;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR32C128:
    case Q8LinearAddScheduleId::MmaExactR32C64:
    case Q8LinearAddScheduleId::MmaExactR32C96:
    case Q8LinearAddScheduleId::MmaExactR32C128:
        return 32;
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR48C128:
    case Q8LinearAddScheduleId::MmaExactR48C64:
    case Q8LinearAddScheduleId::MmaExactR48C96:
        return 48;
    case Q8LinearAddScheduleId::MmaR64C64:
    case Q8LinearAddScheduleId::MmaR64C96:
    case Q8LinearAddScheduleId::MmaR64C112:
    case Q8LinearAddScheduleId::MmaR64C128:
    case Q8LinearAddScheduleId::MmaExactR64C64:
    case Q8LinearAddScheduleId::MmaExactR64C96:
    case Q8LinearAddScheduleId::MmaExactR64C112:
    case Q8LinearAddScheduleId::MmaExactR64C128:
        return 64;
    case Q8LinearAddScheduleId::MmaR128C64:
    case Q8LinearAddScheduleId::MmaR128C80:
    case Q8LinearAddScheduleId::MmaExactR128C64:
    case Q8LinearAddScheduleId::MmaExactR128C80:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule has no row tile");
}

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule);

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
    case Q8LinearAddScheduleId::MediumSplitK:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 4;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR64C64:
    case Q8LinearAddScheduleId::MmaR128C64:
    case Q8LinearAddScheduleId::MmaExactR32C64:
    case Q8LinearAddScheduleId::MmaExactR48C64:
    case Q8LinearAddScheduleId::MmaExactR64C64:
    case Q8LinearAddScheduleId::MmaExactR128C64:
        return 64;
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR128C80:
    case Q8LinearAddScheduleId::MmaExactR128C80:
        return 80;
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR64C96:
    case Q8LinearAddScheduleId::MmaExactR32C96:
    case Q8LinearAddScheduleId::MmaExactR48C96:
    case Q8LinearAddScheduleId::MmaExactR64C96:
        return 96;
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR64C112:
    case Q8LinearAddScheduleId::MmaExactR64C112:
        return 112;
    case Q8LinearAddScheduleId::MmaR32C128:
    case Q8LinearAddScheduleId::MmaR48C128:
    case Q8LinearAddScheduleId::MmaR64C128:
    case Q8LinearAddScheduleId::MmaExactR32C128:
    case Q8LinearAddScheduleId::MmaExactR64C128:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule is not token-sliced");
}

} // namespace

const char* q8_linear_add_schedule_name(Q8LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
        return "linear_add.q8.decode.r16.residual";
    case Q8LinearAddScheduleId::GroupedSplitK:
        return "linear_add.q8.grouped_splitk.residual";
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
        return "linear_add.q8.splitk.mma.capacity.residual";
    case Q8LinearAddScheduleId::SplitKMmaExactT:
        return "linear_add.q8.splitk8.mma.r16.exact_t.residual";
    case Q8LinearAddScheduleId::MediumSplitK:
        return "linear_add.q8.medium_splitk.residual";
    case Q8LinearAddScheduleId::SimtR8C4:
        return "linear_add.q8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case Q8LinearAddScheduleId::MmaR32C64:
        return "linear_add.q8.mma.r32.c64.residual";
    case Q8LinearAddScheduleId::MmaR32C80:
        return "linear_add.q8.mma.r32.c80.residual";
    case Q8LinearAddScheduleId::MmaR32C96:
        return "linear_add.q8.mma.r32.c96.residual";
    case Q8LinearAddScheduleId::MmaR32C128:
        return "linear_add.q8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR48C64:
        return "linear_add.q8.mma.r48.c64.residual";
    case Q8LinearAddScheduleId::MmaR48C96:
        return "linear_add.q8.mma.r48.c96.residual";
    case Q8LinearAddScheduleId::MmaR48C112:
        return "linear_add.q8.mma.r48.c112.residual";
    case Q8LinearAddScheduleId::MmaR48C128:
        return "linear_add.q8.mma.r48.c128.residual";
    case Q8LinearAddScheduleId::MmaR64C64:
        return "linear_add.q8.mma.r64.c64.residual";
    case Q8LinearAddScheduleId::MmaR64C96:
        return "linear_add.q8.mma.r64.c96.residual";
    case Q8LinearAddScheduleId::MmaR64C112:
        return "linear_add.q8.mma.r64.c112.residual";
    case Q8LinearAddScheduleId::MmaR64C128:
        return "linear_add.q8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR128C64:
        return "linear_add.q8.mma.r128.c64.residual";
    case Q8LinearAddScheduleId::MmaR128C80:
        return "linear_add.q8.mma.r128.c80.residual";
    case Q8LinearAddScheduleId::MmaExactR32C64:
        return "linear_add.q8.mma.exact.r32.c64.residual";
    case Q8LinearAddScheduleId::MmaExactR32C96:
        return "linear_add.q8.mma.exact.r32.c96.residual";
    case Q8LinearAddScheduleId::MmaExactR32C128:
        return "linear_add.q8.mma.exact.r32.c128.residual";
    case Q8LinearAddScheduleId::MmaExactR48C64:
        return "linear_add.q8.mma.exact.r48.c64.residual";
    case Q8LinearAddScheduleId::MmaExactR48C96:
        return "linear_add.q8.mma.exact.r48.c96.residual";
    case Q8LinearAddScheduleId::MmaExactR64C64:
        return "linear_add.q8.mma.exact.r64.c64.residual";
    case Q8LinearAddScheduleId::MmaExactR64C96:
        return "linear_add.q8.mma.exact.r64.c96.residual";
    case Q8LinearAddScheduleId::MmaExactR64C112:
        return "linear_add.q8.mma.exact.r64.c112.residual";
    case Q8LinearAddScheduleId::MmaExactR64C128:
        return "linear_add.q8.mma.exact.r64.c128.residual";
    case Q8LinearAddScheduleId::MmaExactR128C64:
        return "linear_add.q8.mma.exact.r128.c64.residual";
    case Q8LinearAddScheduleId::MmaExactR128C80:
        return "linear_add.q8.mma.exact.r128.c80.residual";
    }
    return "linear_add.q8.unknown";
}

bool q8_linear_add_schedule_uses_mma(Q8LinearAddScheduleId schedule) noexcept {
    return schedule != Q8LinearAddScheduleId::DecodeR16 &&
           schedule != Q8LinearAddScheduleId::SimtR8C4;
}

bool q8_linear_add_admits(const Q8LinearAddProblem& problem) noexcept {
    const bool shape = (problem.rows == 2048 && (problem.k == 4096 || problem.k == 6144)) ||
                       (problem.rows == 5120 && (problem.k == 6144 || problem.k == 17408));
    return shape && problem.padded_k == problem.k && problem.cols >= 1;
}

Q8LinearAddPlan q8_linear_add_resolve_plan(const Q8LinearAddProblem& problem) {
    if (!q8_linear_add_admits(problem)) {
        throw std::invalid_argument("q8 linear_add: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("q8 linear_add: admitted problem has no covering route");
    };
    if (problem.rows == 5120) {
        if (problem.k == 6144) { return resolve_from(kN5120K6144Routes); }
        return resolve_from(kN5120K17408Routes);
    }
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK4096Routes);
}

void q8_linear_add_execute_plan(const Q8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q8LinearAddPlan resolved = q8_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("q8 linear_add: plan does not match the exact problem");
    }
    if (plan.schedule == Q8LinearAddScheduleId::GroupedSplitK) {
        q8_linear_add_grouped_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaCapacity) {
        q8_linear_add_splitk_capacity_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::DecodeR16) {
        q8_linear_add_decode_r16_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaExactT) {
        q8_linear_add_splitk_mma_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::MediumSplitK) {
        q8_linear_add_medium_splitk_launch(x, w, residual_out, stream);
        return;
    }
    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice  = x.slice(1, offset, count);
            Tensor residual_slice = residual_out.slice(1, offset, count);
            switch (plan.schedule) {
            case Q8LinearAddScheduleId::DecodeR16:
            case Q8LinearAddScheduleId::MediumSplitK:
                break;
            case Q8LinearAddScheduleId::SimtR8C4:
                q8_linear_add_simt_r8_c4_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C64:
                q8_linear_add_mma_r32_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C80:
                q8_linear_add_mma_r32_c80_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C96:
                q8_linear_add_mma_r32_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C128:
                q8_linear_add_mma_r32_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C64:
                q8_linear_add_mma_r48_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C96:
                q8_linear_add_mma_r48_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C112:
                q8_linear_add_mma_r48_c112_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C128:
                q8_linear_add_mma_r48_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C64:
                q8_linear_add_mma_r64_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C96:
                q8_linear_add_mma_r64_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C112:
                q8_linear_add_mma_r64_c112_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C128:
                q8_linear_add_mma_r64_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C64:
                q8_linear_add_mma_r128_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C80:
                q8_linear_add_mma_r128_c80_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR32C64:
                q8_linear_add_mma_exact_r32_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR32C96:
                q8_linear_add_mma_exact_r32_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR32C128:
                q8_linear_add_mma_exact_r32_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR48C64:
                q8_linear_add_mma_exact_r48_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR48C96:
                q8_linear_add_mma_exact_r48_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR64C64:
                q8_linear_add_mma_exact_r64_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR64C96:
                q8_linear_add_mma_exact_r64_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR64C112:
                q8_linear_add_mma_exact_r64_c112_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR64C128:
                q8_linear_add_mma_exact_r64_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR128C64:
                q8_linear_add_mma_exact_r128_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaExactR128C80:
                q8_linear_add_mma_exact_r128_c80_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::GroupedSplitK:
            case Q8LinearAddScheduleId::SplitKMmaCapacity:
            case Q8LinearAddScheduleId::SplitKMmaExactT:
                break;
            }
            throw std::logic_error("q8 linear_add: unknown tiled schedule");
        });
}

void q8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q8_linear_add_execute_plan(q8_linear_add_resolve_plan(problem), x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
