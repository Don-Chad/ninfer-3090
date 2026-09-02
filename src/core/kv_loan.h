#pragma once

#include "core/evictable_kv_pool.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// Pages of one KV pool that a window borrows, and the arena granules those pages back. Only
// granules that lie entirely inside the borrowed pages are collected, so a granule shared with a
// live page is never unmapped.
struct KVLoanPlan {
    std::vector<KVPageRun> runs;
    std::vector<std::size_t> granules;
    std::size_t bytes = 0;
};

// Pages one granule spans in a plane, for planes whose stride divides the granularity evenly.
// A plane that needs more than kMaxLoanUnitPages pages per granule (the tiny scale planes) is
// never lent: its pages stay mapped and cost the loan nothing but a few per cent of yield.
inline constexpr std::uint32_t kMaxLoanUnitPages = 256;

[[nodiscard]] inline std::uint32_t kv_loan_unit_pages(const DeviceKVPagePool& pages,
                                                      std::size_t granularity) {
    std::uint32_t unit = 0;
    for (std::size_t index = 0; index < pages.plane_count(); ++index) {
        const auto stride = static_cast<std::size_t>(pages.plane(index).nb[3]);
        if (stride == 0 || granularity % stride != 0) { continue; }
        const std::size_t per_granule = granularity / stride;
        if (per_granule > kMaxLoanUnitPages) { continue; }
        unit = std::max(unit, static_cast<std::uint32_t>(per_granule));
    }
    return unit;
}

// Granules of the arena that the given page range covers entirely, across every lendable plane.
inline void collect_loan_granules(const EvictableKVPool& arena, const DeviceKVPagePool& pages,
                                  std::int32_t begin, std::uint32_t count,
                                  std::vector<std::size_t>& out) {
    const std::size_t granularity = arena.granularity();
    const auto* const base        = static_cast<const std::byte*>(arena.arena().data);
    for (std::size_t index = 0; index < pages.plane_count(); ++index) {
        const auto stride = static_cast<std::size_t>(pages.plane(index).nb[3]);
        if (stride == 0 || granularity % stride != 0 ||
            granularity / stride > kMaxLoanUnitPages) {
            continue;
        }
        const KVPlaneByteRange range = pages.plane_page_range(index, begin, count);
        const auto offset  = static_cast<std::size_t>(static_cast<const std::byte*>(range.base) -
                                                     base);
        const std::size_t first = (offset + granularity - 1) / granularity;
        const std::size_t last  = (offset + range.bytes) / granularity;
        for (std::size_t granule = first; granule < last; ++granule) {
            if (granule < arena.lendable_granules()) { out.push_back(granule); }
        }
    }
}

// Draws a loan of at least `bytes` from the highest free pages, growing one run at a time in
// whole units. Returns an empty plan when the free pages cannot cover the request.
[[nodiscard]] inline KVLoanPlan plan_kv_loan(const EvictableKVPool& arena,
                                             const DeviceKVPagePool& pages, std::size_t bytes) {
    KVLoanPlan plan;
    const std::size_t granularity = arena.granularity();
    const std::uint32_t unit      = kv_loan_unit_pages(pages, granularity);
    if (bytes == 0 || unit == 0 || granularity == 0) { return plan; }
    const std::size_t want = (bytes + granularity - 1) / granularity;
    if (want > arena.window_capacity_bytes() / granularity) { return plan; }

    const std::span<const KVPageRun> free_runs = pages.free_runs();
    std::vector<std::size_t> collected;
    for (auto run = free_runs.rbegin(); run != free_runs.rend() && plan.granules.size() < want;
         ++run) {
        const std::int32_t run_end = run->begin + static_cast<std::int32_t>(run->count);
        std::uint32_t length       = 0;
        collected.clear();
        // Grow a suffix of this run until its granules cover what is still missing: the highest
        // pages go first, which is where the allocator leaves the pool free the longest.
        while (length < run->count && plan.granules.size() + collected.size() < want) {
            length = std::min(run->count, length + unit);
            collected.clear();
            collect_loan_granules(arena, pages, run_end - static_cast<std::int32_t>(length),
                                  length, collected);
        }
        if (collected.empty()) { continue; }
        plan.runs.push_back(
            KVPageRun{.begin = run_end - static_cast<std::int32_t>(length), .count = length});
        plan.granules.insert(plan.granules.end(), collected.begin(), collected.end());
    }
    std::sort(plan.granules.begin(), plan.granules.end());
    plan.granules.erase(std::unique(plan.granules.begin(), plan.granules.end()),
                        plan.granules.end());
    if (plan.granules.size() < want) { return KVLoanPlan{}; }
    // Lease exactly what the window needs; the extra granules stay mapped inside the lent pages.
    plan.granules.resize(want);
    plan.bytes = want * granularity;
    return plan;
}

} // namespace ninfer
