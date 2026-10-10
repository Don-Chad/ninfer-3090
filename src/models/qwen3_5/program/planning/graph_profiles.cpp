#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/execution_context.h"
#include "ninfer/ops/softmax_attention.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
// Resource tiers bound inactive attention work. They are not kernel/topology boundaries.
constexpr std::array<std::uint32_t, 7> kCausalVisibleTiers{128,  512,   2048, 4096,
                                                           8192, 16384, 32768};

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> causal_resource_profiles(std::uint32_t capacity,
                                                            std::uint32_t visible_offset) {
    std::vector<std::uint32_t> ends;
    for (const auto visible : kCausalVisibleTiers)
        if (visible >= visible_offset) ends.push_back(visible - visible_offset);
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> dflash_base_profiles(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6 && draft_window <= 15) {
        add_target_boundary(draft_window <= 11 ? 512U : 1024U);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(max_frontier, ends);
}

// Mirror of the 35B (16 query heads) verify route table, which is the geometry DFlash targets.
bool verify_uses_chunked_small_t(std::uint32_t draft_window, std::uint32_t batch_size,
                                 std::uint32_t max_visible_keys) {
    const std::uint32_t tokens = draft_window + 1;
    if (tokens <= 6) { return false; }
    if (batch_size > 1) { return true; }
    // At batch one the route only depends on the target while the width is inside the verify
    // domain; past it causal_attention_resolve_route returns Prompt for every envelope. Without
    // this line the mirror claims a target dependence the route does not have, which costs a
    // second topology class at widths no verify path can request.
    if (tokens > 16) { return false; }
    const std::uint32_t prompt_visible_limit = tokens <= 12 ? 512U : 1024U;
    return max_visible_keys > prompt_visible_limit;
}

// How every causal attention an MTP round records at one execution frontier executes: the target
// verify and the MTP batch forward (both T=K+1 over E+K+1 keys), then each of the K-1
// autoregressive draft steps (T=1). The round's envelopes carry only a maximum, so this is exactly
// what a profile captured at that frontier records.
std::vector<ops::CausalAttentionLaunchShape> mtp_attention_shapes(
    std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t frontier,
    const MtpGraphAttention& attention) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    std::vector<ops::CausalAttentionLaunchShape> shapes;
    shapes.reserve(draft_window);
    shapes.push_back(ops::causal_softmax_attention_launch_shape(
        attention.geometry, attention.storage,
        {1U, visible(static_cast<std::uint64_t>(frontier) + draft_window + 1ULL)}, 1,
        static_cast<std::int32_t>(draft_window) + 1, {nullptr, attention.multiprocessor_count}));
    for (std::uint32_t step = 0; step + 1 < draft_window; ++step) {
        shapes.push_back(ops::causal_softmax_attention_launch_shape(
            attention.geometry, attention.storage,
            {1U, visible(static_cast<std::uint64_t>(frontier) + draft_window + step + 2ULL)}, 1,
            1, {nullptr, attention.multiprocessor_count}));
    }
    return shapes;
}

// Every frontier f in (lo, hi] whose shapes differ from f-1's. Each attention route is a
// monotone step function of its visible-key maximum, so an interval whose ends agree holds no
// change and bisection finds them all in O(changes * log capacity).
template <class Shapes>
void collect_shape_changes(std::uint32_t lo, std::uint32_t hi, const Shapes& shapes,
                           std::vector<std::uint32_t>& changes) {
    if (hi <= lo || shapes(lo) == shapes(hi)) { return; }
    if (hi - lo == 1U) {
        changes.push_back(hi);
        return;
    }
    const std::uint32_t mid = lo + (hi - lo) / 2U;
    collect_shape_changes(lo, mid, shapes, changes);
    collect_shape_changes(mid, hi, shapes, changes);
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; all tiers share one topology per exact B.
    return causal_resource_profiles(capacity, 1);
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                      std::uint32_t draft_window,
                                                      const MtpGraphAttention& attention) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    // Each profile replays the attention routes its maximum selects across its whole range, so
    // the frontier breaks wherever any route changes. instantiate_graph_family builds one
    // executable per topology class and installs the other profiles of that class through an
    // in-place update, which cannot cross a change of kernel-node count; profiles share a class
    // exactly when every attention in the round enqueues the same number of kernels.
    const std::uint32_t max_frontier = capacity - 1;
    const auto shapes                = [&](std::uint32_t frontier) {
        return mtp_attention_shapes(capacity, draft_window, frontier, attention);
    };
    std::vector<std::uint32_t> changes;
    collect_shape_changes(0U, max_frontier, shapes, changes);
    for (const std::uint32_t change : changes) { ends.push_back(change - 1U); }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());

    std::vector<GraphExecutionProfile> profiles = graph_profiles_through(max_frontier, ends);
    std::vector<std::vector<std::uint32_t>> classes;
    for (GraphExecutionProfile& profile : profiles) {
        std::vector<std::uint32_t> nodes;
        for (const ops::CausalAttentionLaunchShape shape : shapes(profile.max)) {
            nodes.push_back(shape.kernel_nodes);
        }
        const auto found       = std::find(classes.begin(), classes.end(), nodes);
        profile.topology_class = static_cast<std::uint32_t>(found - classes.begin());
        if (found == classes.end()) { classes.push_back(std::move(nodes)); }
    }
    return profiles;
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                         std::uint32_t capacity,
                                                         std::uint32_t draft_window,
                                                         std::uint32_t batch_size) {
    if (capacity == 0 || draft_window == 0 || draft_window > 15) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    if (backend == SpeculativeBackend::DFlash2) {
        auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
        return profiles;
    }
    // The sm_86 verify route table keeps a target-dependent chunked small-T route, so the target
    // still contributes a topology class at its route flip (see verify_uses_chunked_small_t).
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool split_swa           = profile.max > 96U;
        const bool chunked_target =
            verify_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = (chunked_target ? 2U : 0U) | (split_swa ? 1U : 0U);
    }
    return profiles;
}

execution::MtpCausalAttentionEnvelopes mtp_causal_attention_envelopes(std::uint32_t max_frontier,
                                                                      std::uint32_t k,
                                                                      std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    execution::MtpCausalAttentionEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

execution::DFlashEnvelopes dflash_envelopes(std::uint32_t max_frontier, std::uint32_t k) {
    return execution::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, k + 1},
    };
}

} // namespace ninfer::models::qwen3_5::detail
