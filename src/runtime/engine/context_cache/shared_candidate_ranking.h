#pragma once

// Which shared-prefix candidates the projection weighs when a request offers more than its
// exhaustive subset search can. Pure arithmetic over per-candidate demand, so the rule is stated
// and testable in one place.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace ninfer::runtime {

struct SharedCandidateRank {
    // Declared by the client or asked for by two or more reuse domains: may force an eviction.
    bool pressure_capable = false;
    // Distinct reuse domains that asked for the prefix. A session asking again is one domain, not
    // several: independent callers are the evidence that a prefix is genuinely shared.
    std::uint32_t reuse_domains = 0;
    // Prefill work a hit saves.
    std::uint64_t rebuild_ns = 0;
};

// The indices of the `limit` strongest candidates, in their original order: pressure-capable
// first, then more distinct reuse domains, then more rebuild work saved; ties keep the earlier
// candidate. With `limit` or fewer candidates, all of them.
[[nodiscard]] inline std::vector<std::size_t>
strongest_shared_candidates(std::span<const SharedCandidateRank> ranks, std::size_t limit) {
    std::vector<std::size_t> kept(ranks.size());
    std::iota(kept.begin(), kept.end(), std::size_t{0});
    if (ranks.size() <= limit) { return kept; }
    std::stable_sort(kept.begin(), kept.end(), [&](std::size_t left, std::size_t right) {
        const SharedCandidateRank& a = ranks[left];
        const SharedCandidateRank& b = ranks[right];
        if (a.pressure_capable != b.pressure_capable) { return a.pressure_capable; }
        if (a.reuse_domains != b.reuse_domains) { return a.reuse_domains > b.reuse_domains; }
        return a.rebuild_ns > b.rebuild_ns;
    });
    kept.resize(limit);
    std::sort(kept.begin(), kept.end());
    return kept;
}

} // namespace ninfer::runtime
