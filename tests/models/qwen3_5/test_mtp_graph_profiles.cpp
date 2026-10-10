// Contract of mtp_graph_profiles: every profile stays on one attention route, and a topology class
// never mixes the chunked small-T route (two kernels per chunk) with the single-launch routes,
// because instantiate_graph_family instantiates one cudaGraphExec_t per class and installs every
// other profile of that class through cudaGraphExecUpdate, which cannot cross a change of node
// count. An MTP round records the target verify and the MTP batch forward
// at T=K+1 and K-1 autoregressive draft steps at T=1, each over its own visible-key maximum, and
// the route table differs between the 27B and 35B head geometries -- so every one of those
// attentions is checked, for both geometries, rather than the verify alone.

#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "ninfer/types.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::KvCacheStorage;
using ninfer::models::qwen3_5::detail::MtpGraphAttention;
using ninfer::ops::AttentionHeadGeometry;
using ninfer::ops::CausalAttentionExecutionEnvelope;
using ninfer::ops::detail::causal_attention_resolve_route;
using ninfer::ops::detail::causal_attention_route_name;
using ninfer::ops::detail::CausalAttentionRoute;

// The official Qwen3.6/3.8-27B dense and Qwen3.6-35B-A3B MoE full-attention geometries.
constexpr AttentionHeadGeometry kGeometries[] = {{256, 24, 4}, {256, 16, 2}};

// Every KV storage the engine can be configured with. The route table branches on storage, so a
// planner that is right for one of them is not thereby right for the rest.
constexpr KvCacheStorage kStorages[] = {
    KvCacheStorage::BFloat16,     KvCacheStorage::Int8Group64,      KvCacheStorage::Fp8E4M3Row256,
    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value,
    KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, KvCacheStorage::RotatedLloyd4KeyInt4Value,
};

// Routes of every attention one round records at this frontier: verify first, then each AR step.
std::vector<CausalAttentionRoute> routes_at(std::uint32_t capacity, std::uint32_t draft_window,
                                            std::uint32_t frontier,
                                            const MtpGraphAttention& attention) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    std::vector<CausalAttentionRoute> routes;
    routes.push_back(causal_attention_resolve_route(
        attention.geometry.query_heads, static_cast<std::int32_t>(draft_window) + 1, 1,
        attention.storage,
        CausalAttentionExecutionEnvelope{
            1U, visible(static_cast<std::uint64_t>(frontier) + draft_window + 1ULL)}));
    for (std::uint32_t step = 0; step + 1 < draft_window; ++step) {
        routes.push_back(causal_attention_resolve_route(
            attention.geometry.query_heads, 1, 1, attention.storage,
            CausalAttentionExecutionEnvelope{
                1U, visible(static_cast<std::uint64_t>(frontier) + draft_window + step + 2ULL)}));
    }
    return routes;
}

std::string describe(const std::vector<CausalAttentionRoute>& routes) {
    std::string out;
    for (const CausalAttentionRoute route : routes) {
        if (!out.empty()) { out += ','; }
        out += causal_attention_route_name(route);
    }
    return out;
}

int failures = 0;

void check(std::uint32_t capacity, std::uint32_t draft_window, const MtpGraphAttention& attention) {
    const auto profiles =
        ninfer::models::qwen3_5::detail::mtp_graph_profiles(capacity, draft_window, attention);
    const auto label = [&] {
        return "capacity=" + std::to_string(capacity) + " k=" + std::to_string(draft_window) +
               " heads=" + std::to_string(attention.geometry.query_heads) +
               " kv=" + std::to_string(static_cast<int>(attention.storage));
    };
    if (profiles.empty()) {
        std::cerr << label() << ": no profiles\n";
        ++failures;
        return;
    }

    std::uint32_t expected_min = 0;
    for (const auto& profile : profiles) {
        if (profile.min != expected_min || profile.max < profile.min) {
            std::cerr << label() << ": frontier coverage has a hole at [" << profile.min << ","
                      << profile.max << "]\n";
            ++failures;
        }
        expected_min = profile.max + 1;
    }
    if (profiles.back().max != capacity - 1) {
        std::cerr << label() << ": coverage stops at " << profiles.back().max << "\n";
        ++failures;
    }

    // 1. one route per profile: the executable installed for a profile is replayed across the
    //    whole frontier range of that profile, so the range should not straddle a route flip.
    for (const auto& profile : profiles) {
        const auto lo = routes_at(capacity, draft_window, profile.min, attention);
        const auto hi = routes_at(capacity, draft_window, profile.max, attention);
        if (lo != hi) {
            std::cerr << label() << ": profile [" << profile.min << "," << profile.max
                      << "] class " << profile.topology_class << " spans a route flip "
                      << describe(lo) << " -> " << describe(hi) << "\n";
            ++failures;
        }
    }

    // 2. one node structure per class: prompt and small-T each enqueue two kernels and may share
    //    an executable, but the chunked route enqueues two per chunk and must not share one with
    //    them.
    const auto chunked = [&](std::uint32_t frontier) {
        std::vector<bool> out;
        for (const CausalAttentionRoute route :
             routes_at(capacity, draft_window, frontier, attention)) {
            out.push_back(route == CausalAttentionRoute::ChunkedSmallT);
        }
        return out;
    };
    std::map<std::uint32_t, std::pair<std::vector<bool>, std::vector<CausalAttentionRoute>>>
        of_class;
    for (const auto& profile : profiles) {
        const auto routes         = routes_at(capacity, draft_window, profile.max, attention);
        const auto [it, inserted] = of_class.emplace(profile.topology_class,
                                                     std::pair{chunked(profile.max), routes});
        if (!inserted && it->second.first != chunked(profile.max)) {
            std::cerr << label() << ": class " << profile.topology_class << " carries both "
                      << describe(it->second.second) << " and " << describe(routes)
                      << " (profile [" << profile.min << "," << profile.max << "])\n";
            ++failures;
        }
    }
}

} // namespace

int main() {
    // Every window the MTP decode frame admits, and one past it, so a cap raise that outgrows the
    // verify route table is caught here rather than by a failed graph update at startup.
    constexpr std::uint32_t kSweptDraftWindows =
        ninfer::models::qwen3_5::kMtpDecodeMaximumDrafts + 1U;
    for (const AttentionHeadGeometry geometry : kGeometries) {
        for (const std::uint32_t capacity : {2048U, 16384U, 65536U, 262144U}) {
            for (std::uint32_t draft_window = 1; draft_window <= kSweptDraftWindows;
                 ++draft_window) {
                for (const KvCacheStorage storage : kStorages) {
                    // 82 SMs: the RTX 3090 the sm_86 route tables were measured on.
                    check(capacity, draft_window,
                          {.geometry = geometry, .storage = storage, .multiprocessor_count = 82});
                }
            }
        }
    }
    if (failures != 0) {
        std::cerr << failures << " MTP graph-profile contract violation(s)\n";
        return 1;
    }
    std::cout << "MTP graph profiles keep one attention route per profile and one node structure "
                 "per class\n";
    return 0;
}
