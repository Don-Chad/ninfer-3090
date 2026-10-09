// Route-boundary sweep for the Q8 linear-pair k=5120 table (the 27B KV projection pair).
//
// This is the third and last route table the catch-up merge changed, and it changed the most:
//
//   before   {1,4} TwoSimtR8C4   {5,56} TwoSimtR8C8   {57,kAny} DualMmaR32C128
//   after    {1,85} TwoSimtR8C4  {86,960} DualMmaR32C64   {961,kAny} DualMmaR32C128
//
// The middle kernel of the old table no longer exists -- upstream deleted TwoSimtR8C8 along with
// DualMmaR32C80/C96/C112 -- so the table had to be rewritten and could not simply be kept. But the
// rewrite hands a SIMT kernel everything up to 85 columns where this fork previously gave it four,
// and pushes the c128 tile from 57 out to 961, both on sm_120's authority.
//
// Schedules are driven through q8_pair_execute_schedule, the real dispatch minus its
// plan-matches-problem check, rather than through the kernel launches directly: the tiled routes
// slice the token dimension by their own column tile and pick a full-tile variant based on
// alignment, and a bench that reimplemented that would be timing a replica of the Op.

#include "ops/linear_pair/q8/q8_pair_plan.h"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
namespace detail = ninfer::ops::detail;

constexpr std::int32_t kRows = 1024;

void sweep_for_k(std::int32_t k, const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight first =
        ninfer::bench::make_row_split_weight(QType::Q8_G32_FP16, kRows, k, k, 0x51U);
    ninfer::bench::PackedQuantizedWeight second =
        ninfer::bench::make_row_split_weight(QType::Q8_G32_FP16, kRows, k, k, 0x53U);
    ninfer::DeviceBuffer input(static_cast<std::size_t>(k) * max_tokens * 2);
    ninfer::DeviceBuffer first_out(static_cast<std::size_t>(kRows) * max_tokens * 2);
    ninfer::DeviceBuffer second_out(static_cast<std::size_t>(kRows) * max_tokens * 2);

    const auto make = [&](detail::Q8PairScheduleId schedule) {
        return [&, schedule](std::int32_t tokens, cudaStream_t stream) {
            Tensor x(input.p, DType::BF16, {k, tokens});
            Tensor a(first_out.p, DType::BF16, {kRows, tokens});
            Tensor b(second_out.p, DType::BF16, {kRows, tokens});
            detail::q8_pair_execute_schedule(schedule, x, first.weight, second.weight, a, b,
                                             stream);
        };
    };

    // Only the three the k=5120 table can select. The split-K and concat families are reachable
    // from the k=2048 table, but several of them assume that shape, and a kernel run outside the
    // geometry it was written for does not politely throw -- it reads off the end of a buffer and
    // takes the process with it, which is how an earlier sweep of another Op died.
    std::vector<ninfer::bench::SweepEntry> schedules{
        {"two_simt_r8_c4", make(detail::Q8PairScheduleId::TwoSimtR8C4), 0},
        {"dual_mma_r32_c64", make(detail::Q8PairScheduleId::DualMmaR32C64), 0},
        {"dual_mma_r32_c128", make(detail::Q8PairScheduleId::DualMmaR32C128), 0},
    };

    const std::string title             = "q8 linear_pair k=" + std::to_string(k);
    ninfer::bench::SweepOptions options = base;
    options.title                       = title.c_str();
    std::string routed;
    options.routed_name = [k, &routed](std::int32_t tokens) -> const char* {
        const detail::Q8PairProblem problem{kRows, k, k, tokens};
        routed = detail::q8_pair_schedule_name(detail::q8_pair_resolve_plan(problem).schedule);
        return routed.c_str();
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor a(first_out.p, DType::BF16, {kRows, tokens});
        Tensor b(second_out.p, DType::BF16, {kRows, tokens});
        detail::q8_pair_dispatch(x, first.weight, second.weight, a, b, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

// --- k=2048, the 35B DFlash pair ---------------------------------------------------------------
//
// Two things make this table different from k=5120 and both dictate the shape of this sweep.
//
// First, q8_pair_execute_schedule calls require_dflash_row_views whenever k == 2048: the two
// weights must be exact adjacent row views into one 6144-row parent at rows 4096 and 5120, not
// standalone 1024-row matrices. make_row_split_weight already lays the parent out the way that
// check wants -- Q8_G32_FP16 has no high plane, so the scale plane starts exactly at
// 6144*2048 -- so the views are pointer arithmetic rather than a second allocation.
//
// Second, and the reason this table is smaller than it looks on sm_86: under NINFER_SM8X_COMPAT
// q8_pair_splitk_medium_launch discards its schedule argument entirely and loops
// q8_pair_splitk_exact_t over <=32-column chunks. All twelve DualSplitKMediumC* schedules are
// therefore the same code here, and the ten route bands between {33,48} and {161,192} cannot
// differ. Four of them are swept anyway, to show that rather than assert it.

constexpr std::int32_t kDFlashParentRows = 6144;
constexpr std::int32_t kDFlashHidden     = 2048;
constexpr std::int32_t kDFlashViewRows   = 1024;
constexpr std::int32_t kDFlashFirstRow   = 4096;
constexpr std::int32_t kDFlashSecondRow  = 5120;

ninfer::Weight dflash_row_view(const ninfer::bench::PackedQuantizedWeight& parent,
                               std::int32_t row) {
    constexpr std::uint64_t kCodeBytes =
        static_cast<std::uint64_t>(kDFlashParentRows) * kDFlashHidden;
    constexpr std::uint64_t kScaleRowBytes = (kDFlashHidden / 32) * 2;

    const auto* base  = static_cast<const std::uint8_t*>(parent.storage.p);
    ninfer::Weight view = parent.weight;
    view.qdata          = base + static_cast<std::uint64_t>(row) * kDFlashHidden;
    view.scales         = base + kCodeBytes + static_cast<std::uint64_t>(row) * kScaleRowBytes;
    view.n              = kDFlashViewRows;
    view.shape[0]       = kDFlashViewRows;
    view.padded_shape[0] = kDFlashViewRows;
    return view;
}

void sweep_k2048(const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight parent = ninfer::bench::make_row_split_weight(
        QType::Q8_G32_FP16, kDFlashParentRows, kDFlashHidden, kDFlashHidden, 0x51U);
    const ninfer::Weight first  = dflash_row_view(parent, kDFlashFirstRow);
    const ninfer::Weight second = dflash_row_view(parent, kDFlashSecondRow);

    ninfer::DeviceBuffer input(static_cast<std::size_t>(kDFlashHidden) * max_tokens * 2);
    ninfer::DeviceBuffer first_out(static_cast<std::size_t>(kDFlashViewRows) * max_tokens * 2);
    ninfer::DeviceBuffer second_out(static_cast<std::size_t>(kDFlashViewRows) * max_tokens * 2);

    const auto make = [&](detail::Q8PairScheduleId schedule) {
        return [&, schedule](std::int32_t tokens, cudaStream_t stream) {
            Tensor x(input.p, DType::BF16, {kDFlashHidden, tokens});
            Tensor a(first_out.p, DType::BF16, {kDFlashViewRows, tokens});
            Tensor b(second_out.p, DType::BF16, {kDFlashViewRows, tokens});
            detail::q8_pair_execute_schedule(schedule, x, first, second, a, b, stream);
        };
    };

    // Domains, all of them enforced by a throw rather than by luck, so the harness reports "n/a"
    // below a schedule's minimum instead of faulting:
    //   DualDecodeR4        one column at a time; capped so the sweep does not spend minutes
    //                       proving a decode kernel is slow at T=2048.
    //   DualSplitKMmaExactT throws outside T=2..32 (kFirstExactT..kLastExactT).
    //   DualSplitKMedium*   throws below T=33; unbounded above.
    //   Concat*             tiled by launch_tiled, so defined at every width.
    using Id = detail::Q8PairScheduleId;
    std::vector<ninfer::bench::SweepEntry> schedules{
        {"dual_decode_r4", make(Id::DualDecodeR4), 32},
        {"dual_splitk_exact_t", make(Id::DualSplitKMmaExactT), 32},
        {"medium_c48", make(Id::DualSplitKMediumC48), 0},
        {"medium_c64", make(Id::DualSplitKMediumC64), 0},
        {"medium_c128", make(Id::DualSplitKMediumC128), 0},
        {"medium_c192", make(Id::DualSplitKMediumC192), 0},
        {"concat_r32_c64", make(Id::ConcatMmaR32C64), 0},
        {"concat_r32_c96", make(Id::ConcatMmaR32C96), 0},
        {"concat_r32_c128", make(Id::ConcatMmaR32C128), 0},
        {"concat_r48_c96", make(Id::ConcatMmaR48C96), 0},
        {"concat_r48_c112", make(Id::ConcatMmaR48C112), 0},
        {"concat_r48_c128", make(Id::ConcatMmaR48C128), 0},
        {"concat_r64_c96", make(Id::ConcatMmaR64C96), 0},
        {"concat_r64_c128", make(Id::ConcatMmaR64C128), 0},
        {"concat_r96_c64", make(Id::ConcatMmaR96C64), 0},
        {"concat_r96_c96", make(Id::ConcatMmaR96C96), 0},
        {"concat_r128_c64", make(Id::ConcatMmaR128C64), 0},
        {"concat_r128_c80", make(Id::ConcatMmaR128C80), 0},
    };

    ninfer::bench::SweepOptions options = base;
    options.title                       = "q8 linear_pair k=2048 (35B DFlash pair)";
    std::string routed;
    options.routed_name = [&routed](std::int32_t tokens) -> const char* {
        const detail::Q8PairProblem problem{kDFlashViewRows, kDFlashHidden, kDFlashHidden, tokens};
        routed = detail::q8_pair_schedule_name(detail::q8_pair_resolve_plan(problem).schedule);
        return routed.c_str();
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {kDFlashHidden, tokens});
        Tensor a(first_out.p, DType::BF16, {kDFlashViewRows, tokens});
        Tensor b(second_out.p, DType::BF16, {kDFlashViewRows, tokens});
        detail::q8_pair_dispatch(x, first, second, a, b, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::SweepOptions options;
    const bool k2048 = argc > 1 && std::string(argv[1]) == "--k2048";
    if (k2048) { --argc; ++argv; }

    options.tokens =
        k2048 ? std::vector<std::int32_t>{1,   2,   8,   16,  32,  33,  48,  49,  64,   80,  96,
                                          112, 128, 160, 192, 193, 256, 384, 385, 480,  481, 512,
                                          640, 641, 672, 768, 896, 960, 976, 1024, 1280, 1536, 2048}
              : std::vector<std::int32_t>{1,  2,   4,   8,   16,  24,  32,  48,  56,  64,  80,  85,
                                          86, 96,  112, 128, 160, 192, 256, 384, 512, 768, 960, 961,
                                          1024};
    if (!ninfer::bench::parse_sweep_args(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s [--k2048] [--tokens T,...] [--repeat N] [--warmup N] [--spread]\n",
                     argv[0]);
        return 2;
    }
    if (k2048) {
        sweep_k2048(options);
    } else {
        sweep_for_k(5120, options);
    }
    return 0;
}
