#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include "ops/linear_add/q5/q5_linear_add_kernels.h"

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

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

struct RouteSpec {
    ColsSet cols;
    Q5LinearAddScheduleId schedule;
};

constexpr std::array<SupportSpec, 2> kSupports{{
    {5120, 6144, 6144},
    {5120, 17408, 17408},
}};

// The 2..10 band is measured too (2026-09-09, same bench, --tokens 1,2,4,6,8,10,12,16, medians of
// 15): SIMT split2 wins it outright, and the boundary at 10/11 is where the two curves actually
// cross. k=6144, T=8: split2 74.8 vs c16 101.4. T=10: 93.2 vs 101.4, still split2 but only just,
// and split2's +8.7 us per row puts T=11 at ~102 -- level with c16, which is flat. Same story at
// k=17408 (T=8: 200.7 vs 282.6). split2 is registered only to 10, so 11 upward is not a measured
// flip; it is the domain edge landing exactly where the extrapolation says it should.
//
// Be clear about what that does and does not say, because a serving cohort of 8 lands here. It is
// the right choice between the kernels that exist, not a good one. split2 grows ~8.7 us per row
// (k=6144: 41.0/56.3/74.8/93.2 at T=4/6/8/10) so it barely amortises the weight read, while c16 is
// flat from T=4 to T=16 but flat at 4x the 25.3 us this weight costs to stream once at 854 GB/s.
// The T=1 GEMV reaches 41.0 us, 62% of that floor. A narrow-extent kernel with c16's flatness at
// the GEMV's efficiency is worth most of the C1-to-C8 scaling loss; see TODO section 2c.
//
// **A narrower tile than C16 does not help here, and that is not obvious.** Tried an R64C8 --
// BlockCols 8 with WarpCols 8, so one warp column, 4 warps and 128 threads against C16's 8 and
// 256 -- on the reasoning that a serving cohort of 8 and a k+1 verification round both land at
// extents this narrow and should not pay for 16 padded columns. Measured 2026-09-09, cold,
// medians of 21-31 (us):
//
//   T                   2      4      6      8     10     12     16
//   k=6144   c8      131.1  116.7  115.7  108.5  133.1  133.1  126.0
//            c16     114.7  103.4  102.4  102.4  103.4  102.4   95.2
//            split2   36.9   44.0   56.3   76.8   96.3     --     --
//   k=17408  c8         --  315.4     --  301.1     --     --  352.3
//            c16        --  289.8     --  291.8     --     --  285.7
//            split2     --  120.8     --  192.5     --     --     --
//
// c8 loses to c16 at every width and both lose to split2 below 11, so the route table is unchanged
// and the schedule is not registered.
//
// The reason is worth keeping, because the same change *works* on the GDN input projection (see
// q4_q5_gdn_input_plan.cpp, where C8 and C16 beat C32 by 9-11%). There, cost is dominated by
// padded MMA work, so cutting BN cuts the work. Here the kernel is already weight-read-bound --
// c16 sits at 24% of the 25.3 us this weight costs to stream once -- so cutting BN removes no
// work and halves the warps available to hide the read. Narrowing a tile helps when padding is the
// cost and hurts when bandwidth is.
//
// So TODO section 2c's stated prize -- "a narrow-extent MMA kernel that keeps mma_r64_c16's
// flatness at the GEMV's fraction of bandwidth" -- is not reachable by narrowing BN. The 4x
// between c16 and the weight-streaming floor is still there and still worth having; it just is not
// a tile-width problem.
// A tile narrower than the live extent repeats the whole weight pass per column slice, so
// above 16 columns the 32-wide tile covers a decode round in one pass instead of two.
// Measured on sm_86 with bench/ops/q5_linear_add_schedule_bench.cu, cold, medians of 9-15.
// Upstream's table sent 49..192 to R64C32S4 here; on this architecture that loses 17-23% from
// ~88 upward and 31% at 128 (us): T=88 s4 249.9 vs s3 201.7; T=104 s4 306.2 vs c64 236.5;
// T=128 s4 285.7 vs c128 236.5; T=192 s4 436.2 vs c128 292.9.
//
// The 104..127 band goes to R64C64 and everything from 128 to R64C128 rather than splitting the
// band around C64's spike at exactly 128 (c64 309.2 there against 239.6 at 126 and 272.4 at 130).
// 128 is not an arbitrary point: --prefill-chunk is required to be a multiple of 128, so every
// full prefill chunk lands exactly on it, and C128 also wins outright at 192/256/384. C64 keeps
// the 104..127 band, which only a prompt's ragged tail chunk reaches.
// The narrow band, re-measured 2026-09-11 on an RTX 3090 under Linux (892.8 GB/s achievable read)
// with q5_linear_add_small_t.cu registered: SmallTMmaResidual is the Q5 counterpart of the Q4
// small-T MMA (q5_small_t_mma.cuh) and is flat across T=1..8 where split2 grows ~5-12 us per row.
// Cold, median of 31 (us), p95 of the winner below the loser's min except where noted:
//
//   T                  1      2      3      4      5      6      8      9     10
//   k=6144  split2   33.8   33.8   35.8   38.9   43.0   48.1   63.5   72.7   77.8
//           small_t  34.8   34.8   34.8   34.8   34.8   35.8   35.8     --     --
//           gemv     38.9     --
//   k=17408 split2   75.8   78.8   86.0   99.3  116.7  130.0  176.1  196.6  204.8
//           small_t  81.9   82.9   83.0   82.9   85.0   85.0   88.1     --     --
//           gemv     88.1     --
//
// So split2 takes T=1 from the GEMV as well (13-14%: the GEMV's one warp per row in 16-row blocks
// is 320 blocks for 246 slots, 1.3 waves, where split2's 64-thread blocks are 3.9), split2 keeps
// T=2 (k=6144's T=2 is inside the spread; k=17408's is not), and small_t takes 3..8. At T=4, an
// MTP3 verify, that is 22% faster at k=6144 and 17% at k=17408.
//
// The same kernel's 16- and 32-column tiles, measured the same way (us):
//
//   T                  9     12     16     20     24     32
//   k=6144  small_t  39.9   38.9   45.1   62.7   69.6   93.2
//           best MMA 75.8*  100.4  94.2  109.6  103.4  103.4     (* split2)
//   k=17408 small_t  94.2   99.3  123.9  165.9  185.3  238.6
//           best MMA 192.5* 281.6 276.5  300.0  289.8  285.7
//
// so it takes everything from 3 to its 32-column limit, including the C4 (16) and C8 (32) MTP3
// verify widths the c16/c24/c32 tiles used to serve at 2-3x the cost.
//
// Those 16- and 32-column tiles then shared their staged activation slab between two row tiles
// (KWarps 4; q5_linear_add_small_t.cu has the layout sweep). Same bench (us):
//
//   T                  9     16     17     24     32
//   k=6144  small_t  36.9   37.9   49.2   50.2   55.3
//           best MMA 106.6  99.3  109.6  105.5  105.5
//   k=17408 small_t 106.5  107.6  128.0  132.1  144.4
//           best MMA 285.7 277.5  294.9  294.9  289.8
constexpr std::array<RouteSpec, 6> kK6144Routes{{
    {{1, 2}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{3, 32}, Q5LinearAddScheduleId::SmallTMmaResidual},
    {{33, 103}, Q5LinearAddScheduleId::MmaResidualR64C32S3},
    {{104, 127}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{128, 512}, Q5LinearAddScheduleId::MmaResidualR64C128},
    // Upstream's wave-tail composite (measured on RTX 5090 only; sm_86 was swept to 384 columns).
    {{513, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128Tail},
}};

// Same sweep, k=17408. This table had no 16-wide route at all, so 11..16 paid C32's cost for a
// C16-shaped extent (T=16: c16 270.3 vs c32 300.0). Above that the story matches k=6144, with the
// S4/S3 crossover later: T=48 s4 437.2 vs c24 474.1; T=96 s3 629.8 vs s4 683.0; T=128 c128 665.6
// vs s3 817.2; T=192 c128 798.7 vs s3 1359.9.
constexpr std::array<RouteSpec, 7> kK17408Routes{{
    {{1, 2}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{3, 32}, Q5LinearAddScheduleId::SmallTMmaResidual},
    {{33, 64}, Q5LinearAddScheduleId::MmaResidualR64C32S4},
    {{65, 103}, Q5LinearAddScheduleId::MmaResidualR64C32S3},
    {{104, 127}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{128, 512}, Q5LinearAddScheduleId::MmaResidualR64C128},
    // Upstream's wave-tail composite (measured on RTX 5090 only; sm_86 was swept to 384 columns).
    {{513, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128Tail},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kK6144Routes) && catalog_is_closed(kK17408Routes),
              "Q5 LinearAdd routes must be exact, contiguous, and closed");

bool supported_shape(const Q5LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

// The 128-wide MMA tile loads one row-block of weights per column tile, so a launch costs whole
// waves of 4 column tiles (80 row-blocks x 4 = 320 blocks at 5120 rows): measured on this host a
// 512-column launch costs ~456 us at k=17408 and a 513-column launch ~934 us, i.e. the trailing
// mostly-empty tile is billed as a full wave. Send up to 192 columns of remainder - the whole
// narrow band - to the narrow routes instead, which stay under that wave for every T in it. A
// wider remainder keeps the single wide launch: its tail needs a 128-wide tile of its own, which
// costs the wave the composite is trying to avoid.
constexpr std::int32_t kWaveCols       = 512;
constexpr std::int32_t kNarrowTailCols = 192;

void launch_wide_with_narrow_tail(const Tensor& x, const Weight& w, Tensor& residual_out,
                                  WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const std::int32_t wide = (cols / kWaveCols) * kWaveCols;
    const std::int32_t tail = cols - wide;
    if (wide == 0 || tail == 0 || tail > kNarrowTailCols) {
        q5_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    }

    const Tensor x_wide = x.slice(1, 0, wide);
    Tensor out_wide     = residual_out.slice(1, 0, wide);
    q5_linear_add_mma_r64_c128_launch(x_wide, w, out_wide, stream);

    const Tensor x_tail = x.slice(1, wide, tail);
    Tensor out_tail     = residual_out.slice(1, wide, tail);
    q5_linear_add_execute_plan(
        q5_linear_add_resolve_plan({residual_out.ne[0], x.ne[0], w.padded_shape[1], x_tail.ne[1]}),
        x_tail, w, out_tail, ws, stream);
}

} // namespace

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::Split2ExactResidual:
        return "linear_add.q5.simt.split2.exact.residual";
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        return "linear_add.q5.mma.r64.c16.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        return "linear_add.q5.mma.r64.c24.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C32:
        return "linear_add.q5.mma.r64.c32.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        return "linear_add.q5.mma.r64.c64.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C32S3:
        return "linear_add.q5.mma.r64.c32.s3.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C32S4:
        return "linear_add.q5.mma.r64.c32.s4.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        return "linear_add.q5.mma.r64.c128.cta_collective_residual";
    case Q5LinearAddScheduleId::SmallTMmaResidual:
        return "linear_add.q5.mma.small_t.residual";
    case Q5LinearAddScheduleId::MmaResidualR64C128Tail:
        return "linear_add.q5.mma.r64.c128.cta_collective_residual.narrow_tail";
    case Q5LinearAddScheduleId::SlicedR16T8W4S2:
        return "linear_add.q5.sliced.r16.t8.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR16T16W4S2:
        return "linear_add.q5.sliced.r16.t16.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR16T24W4S2:
        return "linear_add.q5.sliced.r16.t24.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR32T32W4S2:
        return "linear_add.q5.sliced.r32.t32.w4.s2.residual";
    case Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise:
        return "linear_add.q5.sliced.r32.t24.w4.s2.pairwise.residual";
    case Q5LinearAddScheduleId::SlicedR32T32W4S1:
        return "linear_add.q5.sliced.r32.t32.w4.s1.residual";
    case Q5LinearAddScheduleId::SlicedR32T32W2S2:
        return "linear_add.q5.sliced.r32.t32.w2.s2.residual";
    case Q5LinearAddScheduleId::SlicedR32T64W2S1:
        return "linear_add.q5.sliced.r32.t64.w2.s1.residual";
    case Q5LinearAddScheduleId::MmaResidualR32T32K128:
        return "linear_add.q5.mma.r32.t32.k128.residual";
    case Q5LinearAddScheduleId::MmaResidualR32T128:
        return "linear_add.q5.mma.r32.t128.residual";
    }
    return "linear_add.q5.unknown";
}

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem) {
    if (!q5_linear_add_admits(problem)) {
        throw std::invalid_argument("q5 linear_add: exact problem or column count is not admitted");
    }

    const auto resolve_from = [&](const auto& routes) -> Q5LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (route.cols.contains(problem.cols)) { return {route.schedule, 0}; }
        }
        throw std::logic_error("q5 linear_add: admitted problem has no covering route");
    };
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK17408Routes);
}

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q5 linear_add: invalid column interval");
    }
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, min_cols});
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, max_cols});

    return 0;
}

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan resolved = q5_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q5 linear_add: plan does not match the exact problem");
    }
    (void)ws;

    switch (plan.schedule) {
    case Q5LinearAddScheduleId::Split2ExactResidual:
        q5_linear_add_split2_exact_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        q5_linear_add_mma_r64_c16_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        q5_linear_add_mma_r64_c24_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C32:
        q5_linear_add_mma_r64_c32_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        q5_linear_add_mma_r64_c64_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C32S3:
        q5_linear_add_mma_r64_c32_s3_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C32S4:
        q5_linear_add_mma_r64_c32_s4_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        q5_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SmallTMmaResidual:
        q5_linear_add_small_t_mma_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C128Tail:
        launch_wide_with_narrow_tail(x, w, residual_out, ws, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T8W4S2:
        q5_linear_add_sliced_r16_t8_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T16W4S2:
        q5_linear_add_sliced_r16_t16_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T24W4S2:
        q5_linear_add_sliced_r16_t24_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W4S2:
        q5_linear_add_sliced_r32_t32_w4_s2_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise:
        q5_linear_add_sliced_r32_t24_pairwise_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W4S1:
        q5_linear_add_sliced_r32_t32_w4_s1_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W2S2:
        q5_linear_add_sliced_r32_t32_w2_s2_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T64W2S1:
        q5_linear_add_sliced_r32_t64_w2_s1_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR32T32K128:
        q5_linear_add_mma_r32_t32_k128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR32T128:
        q5_linear_add_mma_r32_t128_launch(x, w, residual_out, stream);
        return;
    }
    throw std::logic_error("q5 linear_add: unknown schedule");
}

void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan plan = q5_linear_add_resolve_plan(problem);
    q5_linear_add_execute_plan(plan, x, w, residual_out, ws, stream);
}

} // namespace ninfer::ops::detail
