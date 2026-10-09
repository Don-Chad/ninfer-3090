#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

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
    Q4Q5GdnInputScheduleId schedule;
};

// The 1..6 / 7..32 boundary is measured, and it is not where the kernels' own limits suggest.
// q4_q5_gdn_input_independent.cu accepts T up to 15 -- launch_q4 and launch_q5 both carry a
// dedicated R8C8 route for 5..15 -- so widths 7..15 look like a free extension of the cheap
// direct path. Tried it (2026-09-09, {{1, 15}} / {{16, 32}}) and it is a net regression. Measured
// through DFlash2 draft counts on the 27B, which is the workload that walks these widths one at a
// time; verification width is k+1, and k=4/5 stay inside 1..6 in both tables so they calibrate the
// ~3% between-process drift between the two builds:
//
//   k      4      5      6      7      8     10
//   width  5      6      7      8      9     11
//   direct route to 15, normalised against k=4,5:
//         0.0%  +0.0%  -3.6%  +1.3% -23.8% -13.2%
//
// So the grouped MMA tile wins from width 7 upward and wins overwhelmingly from 9. Width 8 is the
// one place the direct path is competitive, by 1.3%, which is the R8C8 tile fitting exactly; 9
// needs two passes through an 8-wide tile and collapses.
//
// THAT MEASUREMENT TESTED THE WRONG KERNEL, and correcting it moves the boundary from 6 to 8.
//
// `launch_q5` had two direct routes, not one: split4 for T=2..6 and simt_r8_c8 for T=7..15. The
// {{1, 15}} experiment above therefore compared the *grouped tile* against simt_r8_c8, because
// split4 threw above 6. Those two kernels are nothing alike in parallelism -- split4 is one block
// per output row with four warps splitting K, so 12,288 blocks and 12.5 machine-fulls of warps,
// against simt_r8_c8's 1,536 blocks and 3.12, against the grouped MMA tile's 256 and 0.26. TODO
// section 2c establishes that the kernels here are parallelism-starved rather than
// bandwidth-bound, and split4 is the one route that is not.
//
// Instantiating split4 to 8 and re-measuring (2026-09-09, cold, medians of 31 with --spread,
// clocks locked at 1,500 MHz) makes the direct route the outright winner at 7 and 8:
//
//   T    independent (split4)      c8 (was routed here)     independent before (simt_r8_c8)
//   6    207.9  206.8..208.9       243.7  239.6..540.7       188.4
//   7    228.4  226.3..229.4       242.7  240.6..247.8       330.8
//   8    208.9  206.8..209.9       240.6  237.6..244.7       319.5
//   9    435.2  432.1..437.2       529.4  514.0..535.6       486.4
//
// So +5.9% at width 7 and +13.2% at width 8 over the c8 tile that was routed there, with the two
// spreads completely disjoint at both widths, and 31-35% over what the direct route used to cost.
//
// AND THEN ONLY ONE OF THOSE TWO SURVIVED IN SITU, WHICH IS WHY WIDTH 7 STAYS ON c8.
// Measured end to end through the serving path with tools/bench/run_interleaved_ab.py -- two
// binaries alternating inside each repetition, clocks locked at 1,500 MHz, six repetitions, and
// draft count 4 (width 5, on the independent route in both arms) carried as a drift control:
//
//   config                  paired median   positive   control drift
//   k=6, width 7                  -0.78%       0 / 6          +0.00%
//   k=7, width 8                  +2.85%       6 / 6          +0.09%
//
// The control moving 0.00-0.09% is what makes these readable, and both results were identical to
// three figures across all six repetitions. So the cold bench overstates, and at width 7 it
// overstates enough to invert the sign: a 5.9% cold margin is worth -0.78% in production, while a
// 13.2% cold margin is worth +2.85%. That is a calibration point for TODO section 3's warning that
// every boundary in this repository was decided on cold-flush margins -- the practical threshold is
// somewhere between those two, and a cold margin under ~10% should not be trusted to survive.
//
// Hence {1,6} independent / {7,7} c8 / {8,8} independent. The single-width band at 8 looks odd and
// is what the measurement supports: width 8 is the C8 serving cohort, the profile the 27B release
// recommends for multi-user serving, and this kernel is 13.8 ms of its 56.3 ms round, so a 13%
// kernel win landing as ~2.9% of the round is exactly the arithmetic working out. Width 7 is
// DFlash2 at k=6. Both are real workloads; only one of them wants split4.
//
// Why width 8 and not 7, mechanically: split4 costs 228.4 us at width 7 against 208.9 at width 8,
// so it is worse at the odd width, while c8 is flat (242.7 / 240.6) because its cost is set by
// padded width. Eight divides split4's vector loads and seven does not.
//
// Width 9 collapses to 435.2 and that sets the new boundary. It is a register cliff rather than a
// geometric one: split4 carries `__launch_bounds__(128, 10)`, capping it at 51 registers per
// thread, and `acc[kTt]` costs one register per column. Hence the instantiations stop at 8 -- see
// launch_q5_split4_exact. Anyone widening it further must relax MIN_BLOCKS first, which is
// affordable (the kernel has 12.5 machine-fulls of blocks and does not need ten per SM) but is a
// separate measurement.
//
// Two workloads pay for this: DFlash2's verification width is k+1, so its 5->6 cliff *is* the
// 6/7 boundary, and a C8 decode cohort runs at width 8. See TODO sections 2c and 3.
//
// SUPERSEDED 2026-09-13, and by the one comparison the paragraphs above could not make. Everything
// from "THAT MEASUREMENT TESTED THE WRONG KERNEL" down to here was measured on a branch that had
// no small-T kernel, so it compares split4 against the grouped tiles and nothing else. Merging
// that branch with the small-T one puts both kernels in a single binary for the first time, and
// the bench can then ask the question directly. RTX 3090, Windows, 315 W, unlocked clocks, cold,
// median min..p95 of 31 (us):
//
//   T                       6              7              8              9             12
//   small_t_mma          81.9           82.9           82.9           90.1           91.1
//   independent(split4) 169.0          185.3          167.9          476.2          510.0
//   grouped_r64_c8      233.5          235.5          232.4          498.7          496.6
//
// split4 at width 8 is real and reproduces -- 167.9 us against 185.3 at width 7 and 232.4 for the
// c8 tile it displaced, which is the shape of the +2.85% in situ that put it here. It is simply
// beaten 2.03x by small_t_mma at the same width, with the spreads disjoint (84.0 against 165.9).
// small_t_mma wins at every width 1..32 by margins between 2.0x and 5.6x, so the {1,6}/{7,7}/{8,8}
// split is gone and the {1,32} route below replaces it outright.
//
// split4's instantiation at width 8 is deliberately KEPT even though resolve_plan can no longer
// reach it. It stays reachable through execute_schedule, which is what let this comparison happen
// at all, and it is the only non-grouped baseline the bench has below width 16. Do not delete it
// as dead code; it is the control.
//
// Below 32 the tile is chosen by measurement, not by which one exists. Every decode extent lives
// here -- a verification round is k+1 wide (6 at the four draft tokens docs/cli.md recommends) and
// a C8 serving cohort is 8 -- and until 2026-09-09 all of 7..32 took the 32-wide tile, so eight
// live columns issued four times the MMA work they needed.
//
// Measured with bench/ops/q4_q5_gdn_input_schedule_bench.cu, cold, medians of 31 with --spread
// (us, median and min..p95). Every boundary below has its winner's p95 under the runner-up's min,
// so none of it is inside the noise:
//
//   T    independent          c8               c16              c32
//   6    188.4 187..189   240.6 235..247    247.8 245..255   290.8 289..293
//   7    330.8 327..336   238.6 231..250    242.7 238..252   267.3 264..272
//   8    319.5 315..323   236.5 231..248    243.7 237..253   267.3 263..273
//   9    486.4 481..493   504.8 492..526    243.7 240..525   267.3 265..274
//   16          --        506.9 497..523    242.7 237..253   267.3 264..273
//   17          --        750.6 738..1158   495.6 484..513   269.3 266..274
//   32          --        999.4 980..1037   491.5 479..508   264.2 262..268
//
// So c8 wins 7..8 by 10.7-11.5%, c16 wins 9..16 by 8.8-9.2%, and each collapses one column past
// its own width because a second pass costs a whole extra weight read. The staircase is exactly
// the tile widths.
//
// Do not read that 11% as evidence the padding was the main cost. It is not: dropping BN from 32
// to 8 removes 75% of the padded MMA work and buys 11%, because this Op streams ~55 MB of weights
// (4096x5120 q4 plus 12288x5120 q5) whose 64.4 us at 854.2 GB/s no tile choice changes. c8 at
// 236.5 us is 27% of that floor. The padded work was a minor term all along, and the remaining 3.7x
// is the thing worth chasing -- see TODO section 2c.
//
// A tile narrower than the live extent repeats the whole weight pass per column slice, so
// above 16 columns the 32-wide tile covers a decode round in one pass instead of two.
// Above the direct route the cost of a grouped MMA tile is set by its padded column width,
// not by the live token count, so the tile is chosen to be the narrowest one that still
// covers the extent in a single pass. Decode extents (C8 with MTP3 is 32) get the 32-wide
// tile; the 128-wide tile remains the prefill-chunk anchor.
//
// 2026-09-11, RTX 3090 under Linux: SmallTMma -- the Q4 and Q5 small-T MMA kernels
// (q4_ksplit_mma.cuh, q5_small_t_mma.cuh) over query/key and value/z -- is flat across T=1..8 and
// beats everything below it, the T=1 GEMVs included. Cold, median of 31 (us):
//
//   T                  1      2      3      4      5      6      7      8
//   independent    91.1  101.4  111.6  100.4  146.4  164.9  322.5  309.2
//   grouped c8    234.5  234.5  232.4  231.4  231.4  232.4  232.4  231.4
//   small_t        79.7   79.9   78.8   78.8   82.9   82.9   84.0   84.0
//
// At T=4 (an MTP3 verify) that is 74% of the 58.7 us both weights cost to stream once, against
// 58%; at T=1 the independent route's value/z GEMV is 768 blocks of 16 rows for 246 slots.
//
// Its 16- and 32-column tiles against the grouped MMA tiles (us):
//
//   T              9     12     16     20     24     32
//   small_t    100.4  100.4  114.7  164.9  182.3  226.3
//   grouped    238.6  235.7  239.6  261.1  263.2  256.0     (c16 to 16, c32 above)
//
// so it runs to its 32-column limit and the c8/c16/c32 tiles serve nothing below 33. With the
// 16- and 32-column tiles sharing staged activation slabs (KWarps 4; q4_q5_gdn_input_small_t.cu
// has the layout sweep), same bench (us):
//
//   T              9     16     17     24     32
//   small_t     96.3  100.4  137.2  147.5  153.6
//   grouped    240.6  240.6  265.2  268.3  263.2     (c16 to 16, c32 above)
constexpr std::array<RouteSpec, 3> kRoutes{{
    {{1, 32}, Q4Q5GdnInputScheduleId::SmallTMma},
    {{33, 64}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C64},
    {{65, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C8:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c8";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C16:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c16";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C32:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c32";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C64:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c64";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    case Q4Q5GdnInputScheduleId::SmallTMma:
        return "gdn_input_proj.q4_q5.small_t.mma";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r32.c32.s2";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r32.c64.s4";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128S2:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128.s2";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule};
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

void q4_q5_gdn_input_execute_schedule(Q4Q5GdnInputScheduleId schedule, const Tensor& x,
                                      const Weight& qk_weight, const Weight& value_z_weight,
                                      Tensor& qkv, Tensor& z, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C8:
        q4_q5_gdn_input_grouped_mma_c8_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C16:
        q4_q5_gdn_input_grouped_mma_c16_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C32:
        q4_q5_gdn_input_grouped_mma_c32_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C64:
        q4_q5_gdn_input_grouped_mma_c64_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2:
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4:
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128S2:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, schedule,
                                           stream);
        return;
    case Q4Q5GdnInputScheduleId::SmallTMma: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_small_t_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }
    q4_q5_gdn_input_execute_schedule(plan.schedule, x, qk_weight, value_z_weight, qkv, z, stream);
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
