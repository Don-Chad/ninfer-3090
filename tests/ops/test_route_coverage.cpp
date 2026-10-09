// Lists the schedules an Op declares but never routes to.
//
// A schedule nothing selects is where dispatch bugs hide. Both switch fallthroughs found in the
// 2026-09 catch-up -- q4_q5 GroupedHomogeneousPairMmaR32C32S4 with no `return`, and
// GroupedHomogeneousPairMmaR32C64S4 with no `case` at all -- were dormant precisely because the
// live route table never picked the affected ids. Neither is visible to a correctness test: a
// fallthrough runs a second kernel over the first, both compute the same projection, and only the
// cost doubles.
//
// So this test does not fail on unrouted schedules. Keeping an upstream schedule that wins nowhere
// here is a deliberate and correct choice -- deleting it costs merge effort for no measured gain.
// What is not acceptable is not knowing. This prints the set and pins it two ways: the aggregate
// count below, and each Op's exact schedule names against `survey`'s `expected_unrouted` argument.
// The count alone would pass unchanged if a route edit stranded one schedule while un-stranding
// another -- the per-Op name set is what catches that the identity, not just the size, moved.
//
// It also catches the k=2048 case from a different angle: an Op can route to twelve distinct
// schedule ids that are one kernel on this hardware. That does not show up here -- all twelve are
// "routed" -- which is why the comment in q8_pair_plan.cpp spells the collapse out. Unrouted is the
// cheap half of the question; grep the launchers for NINFER_SM8X_COMPAT for the other half.

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

// Column counts to resolve at. Every width to 2304 covers each table's dense low end and its
// narrow exact-tail bands, and the tail values reach the unbounded final route. Resolution is pure
// host logic, so exhaustive is cheaper than clever.
std::vector<std::int32_t> probe_widths() {
    std::vector<std::int32_t> widths;
    for (std::int32_t t = 1; t <= 2304; ++t) { widths.push_back(t); }
    for (const std::int32_t t : {4096, 8192, 16384, 65536, kAnyCols - 1, kAnyCols}) {
        widths.push_back(t);
    }
    return widths;
}

// Walks the enum by integer value until the name function stops recognising it. Every Op here
// returns a "<op>.unknown" sentinel past its last id, which is what bounds the walk -- no header
// needs a Count member it would then have to defend at the next merge.
template <typename Id, typename NameFn>
std::vector<Id> declared_schedules(NameFn name, std::string_view unknown) {
    std::vector<Id> ids;
    for (int raw = 0; raw < 512; ++raw) {
        const Id id = static_cast<Id>(raw);
        if (std::string_view(name(id)) == unknown) { break; }
        ids.push_back(id);
    }
    return ids;
}

struct OpReport {
    std::string op;
    std::size_t declared = 0;
    std::vector<std::string> unrouted;
    std::vector<std::string> expected_unrouted;
    bool identity_ok = true;
};

// One Op, all of its shapes. Surveying per shape would be misleading: the q8_pair k=5120 table
// selects three schedules, so 42 of its 45 ids look "unrouted" there while most are live at
// k=2048. What matters is whether an id is reachable from *any* registered shape of the Op.
//
// `expected_unrouted` is the exact baseline for this Op, compared as a set so reordering the enum
// does not itself fail the test. Update it, in the same commit as the route table that moved,
// naming which schedules changed state -- that is the identity check `kExpectedUnrouted` alone
// cannot make: a route edit that strands one schedule while un-stranding another leaves the total
// unchanged, and only comparing the sets themselves catches that the membership moved.
template <typename Id, typename NameFn, typename ResolveFn>
OpReport survey(const std::string& op, NameFn name, std::string_view unknown,
                const std::vector<std::string>& shape_names,
                const std::vector<ResolveFn>& resolvers,
                std::vector<std::string> expected_unrouted) {
    const std::vector<Id> declared = declared_schedules<Id>(name, unknown);
    std::set<int> routed;
    for (const auto& resolve : resolvers) {
        // Every route table this survey uses is compile-time verified contiguous and closed from
        // column 1 through the unbounded tail (each Op's own `catalog_is_closed`/
        // `routes_are_closed` static_assert), and every shape here is one `*_admits` already
        // accepts. So no probed width can fail to resolve -- a throw here is not "this shape
        // doesn't admit this width", it is one of those guarantees breaking, which must fail the
        // test loudly rather than get folded into "unrouted" by a catch that assumed a gap that
        // cannot exist.
        for (const std::int32_t cols : probe_widths()) {
            routed.insert(static_cast<int>(resolve(cols)));
        }
    }

    OpReport report{op, declared.size(), {}, std::move(expected_unrouted), true};
    for (const Id id : declared) {
        if (routed.count(static_cast<int>(id)) == 0) { report.unrouted.emplace_back(name(id)); }
    }
    std::vector<std::string> actual_sorted = report.unrouted;
    std::vector<std::string> expected_sorted = report.expected_unrouted;
    std::sort(actual_sorted.begin(), actual_sorted.end());
    std::sort(expected_sorted.begin(), expected_sorted.end());
    report.identity_ok = actual_sorted == expected_sorted;
    (void)shape_names;
    return report;
}

} // namespace

int main() {
    namespace detail = ninfer::ops::detail;
    std::vector<OpReport> reports;

    using PairFn = std::function<detail::Q8PairScheduleId(std::int32_t)>;
    reports.push_back(survey<detail::Q8PairScheduleId, decltype(detail::q8_pair_schedule_name),
                             PairFn>(
        "q8_pair", detail::q8_pair_schedule_name, "q8_pair.unknown", {"k=5120", "k=2048"},
        {PairFn([](std::int32_t cols) {
             return detail::q8_pair_resolve_plan({1024, 5120, 5120, cols}).schedule;
         }),
         PairFn([](std::int32_t cols) {
             return detail::q8_pair_resolve_plan({1024, 2048, 2048, cols}).schedule;
         })},
        {"q8_pair.dual_decode.k2048.r8", "q8_pair.dual_decode.k2048.r16",
         "q8_pair.splitk4.mma.r16.c80", "q8_pair.splitk4.mma.r16.c88",
         "q8_pair.splitk4.mma.r16.c96", "q8_pair.splitk4.mma.r16.c104",
         "q8_pair.splitk4.mma.r16.c112", "q8_pair.splitk2.mma.r16.c128",
         "q8_pair.splitk2.mma.r16.c160", "q8_pair.splitk2.mma.r16.c192",
         "q8_pair.splitk2.mma.r16.c224", "q8_pair.splitk2.mma.r16.c256",
         "q8_pair.concat_mma.r32.c80", "q8_pair.concat_mma.r32.c112",
         "q8_pair.concat_mma.r48.c64", "q8_pair.concat_mma.r64.c64",
         "q8_pair.concat_mma.r64.c80", "q8_pair.concat_mma.r96.c80",
         "q8_pair.concat_mma.r96.c112"}));

    using AttnFn = std::function<detail::Q8AttnInputScheduleId(std::int32_t)>;
    reports.push_back(
        survey<detail::Q8AttnInputScheduleId, decltype(detail::q8_attn_input_schedule_name),
               AttnFn>("q8_attn_input", detail::q8_attn_input_schedule_name,
                       "attn_input_proj.q8.unknown", {"target", "companion", "dflash2"},
                       // All three shapes the resolver distinguishes -- see supported_shape() in
                       // q8_attn_input_plan.cpp. Surveying only dflash2 reported 11 of 16
                       // unrouted, because it never asked the other two tables.
                       {AttnFn([](std::int32_t cols) {
                            return detail::q8_attn_input_resolve_plan(
                                       {2048, 4096, 512, 9216, 2048, cols})
                                .schedule;
                        }),
                        AttnFn([](std::int32_t cols) {
                            return detail::q8_attn_input_resolve_plan(
                                       {2048, 4096, 1024, 6144, 2048, cols})
                                .schedule;
                        }),
                        AttnFn([](std::int32_t cols) {
                            return detail::q8_attn_input_resolve_plan(
                                       {5120, 4096, 1024, 6144, 5120, cols})
                                .schedule;
                        })},
                       {"attn_input_proj.q8.simt.r8.c4",
                        "attn_input_proj.q8.dflash2.mma.r16.c64.k128"}));

    using SwigluFn = std::function<detail::Q8LinearSwiGluScheduleId(std::int32_t)>;
    reports.push_back(
        survey<detail::Q8LinearSwiGluScheduleId, decltype(detail::q8_linear_swiglu_schedule_name),
               SwigluFn>("q8_linear_swiglu", detail::q8_linear_swiglu_schedule_name,
                         "linear_swiglu.q8.unknown", {"companion", "dflash2"},
                         {SwigluFn([](std::int32_t cols) {
                              return detail::q8_linear_swiglu_resolve_plan(
                                         {12288, 6144, 2048, 2048, cols})
                                  .schedule;
                          }),
                          SwigluFn([](std::int32_t cols) {
                              return detail::q8_linear_swiglu_resolve_plan(
                                         {34816, 17408, 5120, 5120, cols})
                                  .schedule;
                          })},
                         {"linear_swiglu.q8.dflash2.mma.r32.c64.k128"}));

    using AddFn = std::function<detail::Q5LinearAddScheduleId(std::int32_t)>;
    reports.push_back(
        survey<detail::Q5LinearAddScheduleId, decltype(detail::q5_linear_add_schedule_name), AddFn>(
            "q5_linear_add", detail::q5_linear_add_schedule_name, "linear_add.q5.unknown",
            {"k=6144", "k=17408"},
            {AddFn([](std::int32_t cols) {
                 return detail::q5_linear_add_resolve_plan({5120, 6144, 6144, cols}).schedule;
             }),
             AddFn([](std::int32_t cols) {
                 return detail::q5_linear_add_resolve_plan({5120, 17408, 17408, cols}).schedule;
             })},
            {"linear_add.q5.mma.r64.c16.cta_collective_residual",
             "linear_add.q5.mma.r64.c24.cta_collective_residual",
             "linear_add.q5.mma.r64.c32.cta_collective_residual",
             "linear_add.q5.sliced.r16.t8.w4.s2.residual",
             "linear_add.q5.sliced.r16.t16.w4.s2.residual",
             "linear_add.q5.sliced.r16.t24.w4.s2.residual",
             "linear_add.q5.sliced.r32.t32.w4.s2.residual",
             "linear_add.q5.sliced.r32.t24.w4.s2.pairwise.residual",
             "linear_add.q5.sliced.r32.t32.w4.s1.residual",
             "linear_add.q5.sliced.r32.t32.w2.s2.residual",
             "linear_add.q5.sliced.r32.t64.w2.s1.residual",
             "linear_add.q5.mma.r32.t32.k128.residual",
             "linear_add.q5.mma.r32.t128.residual"}));

    using Q4Q5Fn = std::function<detail::Q4Q5AttnInputScheduleId(std::int32_t)>;
    reports.push_back(
        survey<detail::Q4Q5AttnInputScheduleId, decltype(detail::q4_q5_attn_input_schedule_name),
               Q4Q5Fn>("q4_q5_attn_input", detail::q4_q5_attn_input_schedule_name,
                       "attn_input_proj.q4_q5.unknown", {"35B"},
                       {Q4Q5Fn([](std::int32_t cols) {
                           return detail::q4_q5_attn_input_resolve_plan({5120, 6144, 1024, 5120,
                                                                         cols})
                               .schedule;
                       })},
                       {"attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c32.s4",
                        "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4",
                        "attn_input_proj.q4_q5.mixed.r32.c32.s2",
                        "attn_input_proj.q4_q5.pair.r32.c64.s3",
                        "attn_input_proj.q4_q5.pair.r32.c64.s4",
                        "attn_input_proj.q4_q5.parent_split_fixed"}));

    std::size_t total = 0;
    bool identity_changed = false;
    for (const OpReport& report : reports) {
        std::cout << report.op << ": " << report.unrouted.size() << " of " << report.declared
                  << " unrouted\n";
        for (const std::string& name : report.unrouted) { std::cout << "    " << name << '\n'; }
        if (!report.identity_ok) {
            identity_changed = true;
            std::cerr << report.op << ": unrouted schedule set changed.\n  expected:";
            for (const std::string& name : report.expected_unrouted) { std::cerr << ' ' << name; }
            std::cerr << "\n  actual:  ";
            for (const std::string& name : report.unrouted) { std::cerr << ' ' << name; }
            std::cerr << '\n';
        }
        total += report.unrouted.size();
    }
    if (identity_changed) {
        std::cerr << "route coverage identity changed -- see the per-Op diff above. Update the "
                     "expected_unrouted list in this file for the Op that moved, in the same "
                     "commit as the route table that moved, and say which schedules changed "
                     "state; the aggregate count can stay the same while the membership moves.\n";
        return 1;
    }

    // The pin. Change it deliberately, in the same commit as the route table that moved, and say
    // in that commit which schedules changed state. An unexplained change here means a table edit
    // had a coverage side effect nobody looked at.
    //
    // The inventory as measured, and what is known about each group:
    //
    //   q8_pair            19  decode r8/r16 and splitk c224/c256 have no band in either table;
    //                          the seven concat tile shapes are upstream ids this fork's measured
    //                          boundaries never select; and the eight DualSplitKMediumC80..C192
    //                          ids were stranded by the k=2048 retune (c0c3000e), which collapsed
    //                          {65,192} into one ConcatMmaR32C64 band because all twelve medium
    //                          schedules are the same kernel under NINFER_SM8X_COMPAT. Those eight
    //                          are exactly the routes that retune deleted, which is this test
    //                          doing its job -- the count moved 18 -> 26 and had to be looked at.
    //   q8_attn_input       2  DFlash2MmaR16C64K128 won at no width in the sm_86 sweep. SimtR8C4
    //                          is reachable in principle but no shape's table picks it.
    //   q8_linear_swiglu    1  DFlash2MmaR32C64K128 ties R64C64K128 at 33..44 and wins nowhere.
    //   q5_linear_add       3  MmaResidualR64C32, likewise; and since 2026-09-11 C16 and C24,
    //                          when the small-T MMA took 3..32 at up to 2.9x
    //                          (q5_linear_add_plan.cpp). GemvResidual (unrouted since split2 took
    //                          T=1) was deleted upstream in the 2026-09-17 catch-up: 31 -> 30.
    //                          The 2026-10-09 catch-up added upstream's ten sliced/MMA residual
    //                          tiles (13 here); this fork's measured table keeps split2 and the
    //                          small-T MMA, so none of them is routed on sm_86: 30 -> 41.
    //   q4_q5_attn_input    5  grouped_r32_c64_s4, pair_r32_c64_s3, pair_r32_c64_s4 -- the family
    //                          whose dispatch held both switch bugs the catch-up merge shipped --
    //                          ParentSplitFixed since 2026-09-11, when the small-T MMA took
    //                          1..8 from it (q4_q5_attn_input_plan.cpp), and upstream's
    //                          MixedR32C32S2 from the 2026-10-09 catch-up, never measured here.
    //
    // All of them are kept on purpose: deleting an upstream schedule costs merge effort at every
    // future catch-up for no measured gain here. The point is that the set is written down.
    constexpr std::size_t kExpectedUnrouted = 41;
    if (total != kExpectedUnrouted) {
        std::cerr << "route coverage changed: " << total << " unrouted schedules, expected "
                  << kExpectedUnrouted
                  << ".\nUpdate kExpectedUnrouted in this file together with the route table that "
                     "moved, and record which schedules changed state.\n";
        return 1;
    }
    std::cout << "OK route coverage\n";
    return 0;
}
