#include "models/qwen3_5/program/program_impl.h"

#include "core/device.h"
#include "models/qwen3_5/program/context_work.h"

#include <algorithm>
#include <map>
#include <unordered_map>

namespace ninfer::models::qwen3_5::detail {
namespace {

struct StateDemotion {
    StateImageHandle handle;
    std::uint64_t content_epoch = 0;
    bool host_resident          = false;
    std::vector<CheckpointHandle> references;

    friend bool operator==(const StateDemotion&, const StateDemotion&) = default;
};

struct PageDemotion {
    LogicalKVPageHandle handle;
    std::uint32_t position          = 0;
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
    bool host_resident              = false;
    std::vector<CheckpointHandle> references;

    friend bool operator==(const PageDemotion&, const PageDemotion&) = default;
};

bool checkpoint_prefix(const CheckpointState& earlier, const CheckpointState& later) {
    const auto frontier = earlier.frontier;
    if (frontier > later.frontier || !earlier.identity || !later.identity ||
        earlier.key.identity_tag != later.key.identity_tag ||
        earlier.identity->digests.at(frontier) != later.identity->digests.at(frontier)) {
        return false;
    }
    if (earlier.identity == later.identity) { return true; }
    return std::equal(earlier.identity->ledger.begin(), earlier.identity->ledger.begin() + frontier,
                      later.identity->ledger.begin()) &&
           earlier.identity->prefix_identity.prefix_equals(later.identity->prefix_identity,
                                                           frontier);
}

struct PhysicalPage {
    LogicalKVPageHandle handle;
    std::uint32_t position = 0;
};

struct PhysicalOwners {
    std::vector<CheckpointHandle> sources;
    std::vector<StateImageHandle> states;
    std::vector<PhysicalPage> main;
    std::vector<PhysicalPage> backend;
};

struct HistoryOwners {
    const KVHistory* history = nullptr;
    std::vector<std::uint32_t> checkpoints;
    long references = 0;
};

// Borrowed facts never add a history reference or acquire a reader lease. They exist only
// through one synchronous read-only evaluation; Native mutation invalidates the evaluation.
struct PhysicalFacts {
    explicit PhysicalFacts(const ProgramImpl& program)
        : checkpoint_history(program.checkpoints.size()) {
        std::unordered_map<const KVHistory*, std::uint32_t> indices;
        for (std::uint32_t index = 0; index < program.checkpoints.size(); ++index) {
            const auto& slot = program.checkpoints[index];
            if (!slot.value) { continue; }
            const auto* history = slot.value->kv.get();
            auto [entry, added] =
                indices.try_emplace(history, static_cast<std::uint32_t>(histories.size()));
            if (added) { histories.push_back({history, {}, slot.value->kv.use_count()}); }
            histories[entry->second].checkpoints.push_back(index);
            checkpoint_history[index] = entry->second;
        }
    }

    std::vector<HistoryOwners> histories;
    std::vector<std::uint32_t> checkpoint_history;
};

// Group one physical object's complete holders. Walk each history once per pool, even when
// several checkpoints cover it. Flat page references avoid a heap node/vector for every page.
std::vector<PhysicalOwners> physical_owner_sets(const ProgramImpl& program,
                                                const PhysicalFacts& facts,
                                                std::span<const CheckpointHandle> allowed,
                                                std::span<const CheckpointHandle> excluded,
                                                runtime::ContextResourceUsage shortage) {
    std::vector<bool> eligible(program.checkpoints.size(), false);
    for (const auto handle : allowed) {
        if (program.valid_checkpoint(handle) && !program.checkpoints[handle.index].pins &&
            std::find(excluded.begin(), excluded.end(), handle) == excluded.end()) {
            eligible[handle.index] = true;
        }
    }
    std::map<std::vector<std::uint32_t>, PhysicalOwners> unique;
    for (std::uint32_t index = 0; index < eligible.size(); ++index) {
        if (eligible[index]) { unique.try_emplace(std::vector{index}); }
    }
    const auto permitted = [&](const auto& references) {
        return std::all_of(references.begin(), references.end(),
                           [&](auto index) { return eligible[index]; });
    };
    const bool host = shortage.host_bytes != 0;
    if (host || shortage.state_slots) {
        struct Reference {
            std::uint32_t descriptor, owner;
            StateImageHandle handle;
        };

        std::vector<Reference> references;
        for (std::uint32_t index = 0; index < program.checkpoints.size(); ++index) {
            const auto& slot = program.checkpoints[index];
            if (!slot.value) { continue; }
            const auto state = slot.value->state;
            if (host ? program.state_store->host_resident(state)
                     : program.state_store->device_resident(state)) {
                references.push_back({program.state_store->descriptor_index(state), index, state});
            }
        }
        std::sort(references.begin(), references.end(), [](const auto& a, const auto& b) {
            return a.descriptor != b.descriptor ? a.descriptor < b.descriptor : a.owner < b.owner;
        });
        std::vector<std::uint32_t> owners;
        for (std::size_t begin = 0; begin < references.size();) {
            auto end = begin;
            owners.clear();
            while (end < references.size() &&
                   references[end].descriptor == references[begin].descriptor) {
                owners.push_back(references[end++].owner);
            }
            if (permitted(owners)) { unique[owners].states.push_back(references[begin].handle); }
            begin = end;
        }
    }
    const auto collect_pages = [&](bool main_pool) {
        const auto* addresses =
            main_pool ? program.text_kv_addresses.get() : program.backend_kv_addresses.get();
        const auto* pages =
            main_pool ? program.text_kv_pages.get() : program.backend_kv_pages.get();
        if (!addresses || !pages) { return; }
        struct Reference {
            LogicalKVPageHandle handle;
            std::uint32_t descriptor, position, holders;
        };
        std::vector<std::vector<std::uint32_t>> holder_sets;
        std::vector<Reference> references;
        std::vector<std::uint32_t> boundaries;
        for (const auto& history : facts.histories) {
            if (!main_pool && !history.history->backend) { continue; }
            const auto address = main_pool ? history.history->text : *history.history->backend;
            boundaries.clear();
            boundaries.push_back(0);
            for (const auto owner : history.checkpoints) {
                const auto& record = *program.checkpoints[owner].value;
                boundaries.push_back(
                    kv_pages_for_frontier(main_pool ? record.frontier : record.backend_frontier));
            }
            std::sort(boundaries.begin(), boundaries.end());
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
            for (std::size_t interval = 1; interval < boundaries.size(); ++interval) {
                const auto begin = boundaries[interval - 1], end = boundaries[interval];
                auto& holders = holder_sets.emplace_back();
                for (const auto owner : history.checkpoints) {
                    const auto& record = *program.checkpoints[owner].value;
                    if (kv_pages_for_frontier(main_pool ? record.frontier
                                                        : record.backend_frontier) > begin) {
                        holders.push_back(owner);
                    }
                }
                const auto holder_index = static_cast<std::uint32_t>(holder_sets.size() - 1);
                for (auto position = begin; position < end; ++position) {
                    const auto page = addresses->logical_page(address, position);
                    if (host ? pages->host_resident(page) : pages->device_resident(page)) {
                        references.push_back(
                            {page, pages->descriptor_index(page), position, holder_index});
                    }
                }
            }
        }
        std::sort(references.begin(), references.end(),
                  [](const auto& a, const auto& b) { return a.descriptor < b.descriptor; });
        std::vector<std::uint32_t> owners;
        for (std::size_t begin = 0; begin < references.size();) {
            auto end                 = begin;
            std::uint32_t last_owner = 0, position = 0;
            owners.clear();
            while (end < references.size() &&
                   references[end].descriptor == references[begin].descriptor) {
                const auto& reference = references[end++];
                const auto& holders   = holder_sets[reference.holders];
                owners.insert(owners.end(), holders.begin(), holders.end());
                if (holders.back() >= last_owner) {
                    last_owner = holders.back();
                    position   = reference.position;
                }
            }
            std::sort(owners.begin(), owners.end());
            owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
            if (permitted(owners)) {
                auto& group = unique[owners];
                (main_pool ? group.main : group.backend)
                    .push_back({references[begin].handle, position});
            }
            begin = end;
        }
    };
    if (host || (!shortage.state_slots && shortage.main_kv_pages)) { collect_pages(true); }
    if (host || (!shortage.state_slots && !shortage.main_kv_pages && shortage.backend_kv_pages)) {
        collect_pages(false);
    }
    std::vector<PhysicalOwners> result;
    result.reserve(unique.size());
    for (auto& [indices, owners] : unique) {
        owners.sources.reserve(indices.size());
        for (const auto index : indices) {
            owners.sources.push_back({&program, index, program.checkpoints[index].generation});
        }
        const auto deeper_first = [](const auto& left, const auto& right) {
            return left.position > right.position;
        };
        std::sort(owners.main.begin(), owners.main.end(), deeper_first);
        std::sort(owners.backend.begin(), owners.backend.end(), deeper_first);
        result.push_back(std::move(owners));
    }
    return result;
}

struct ReleaseScratch {
    struct StateReferences {
        StateImageHandle handle;
        std::uint32_t count = 0;
    };

    struct PageReference {
        LogicalKVPageHandle handle;
        std::uint32_t descriptor;
    };

    struct HistoryRelease {
        const KVHistory* history;
        std::uint32_t main, backend;
    };

    std::vector<std::uint32_t> removed, histories;
    std::vector<StateReferences> states;
    std::vector<HistoryRelease> suffixes;
    std::vector<PageReference> pages;
};

std::optional<runtime::ContextResourceUsage>
released_resources(const ProgramImpl& program, const PhysicalFacts& facts,
                   std::span<const CheckpointHandle> handles,
                   runtime::ContextResourceUsage shortage, ReleaseScratch& scratch,
                   std::vector<LogicalKVPageHandle>& released_pages) {
    const bool host    = shortage.host_bytes != 0;
    const bool state   = !host && shortage.state_slots != 0;
    const bool main    = !host && !state && shortage.main_kv_pages != 0;
    const bool backend = !host && !state && !main && shortage.backend_kv_pages != 0;
    auto& [removed, histories, states, suffixes, references] = scratch;
    removed.clear();
    histories.clear();
    states.clear();
    suffixes.clear();
    for (const auto handle : handles) {
        if (!program.valid_checkpoint(handle) || program.checkpoints[handle.index].pins) {
            return std::nullopt;
        }
        if (std::find(removed.begin(), removed.end(), handle.index) != removed.end()) { continue; }
        removed.push_back(handle.index);
        const auto& record = program.checkpoint(handle);
        if (!program.state_store->can_release_checkpoint_owner(record.state)) {
            return std::nullopt;
        }
        if (state || host) {
            const auto found = std::find_if(states.begin(), states.end(), [&](const auto& entry) {
                return entry.handle == record.state;
            });
            if (found == states.end()) {
                states.push_back({record.state, 1});
            } else {
                ++found->count;
            }
        }
        histories.push_back(facts.checkpoint_history[handle.index]);
    }
    std::sort(removed.begin(), removed.end());
    std::sort(histories.begin(), histories.end());
    histories.erase(std::unique(histories.begin(), histories.end()), histories.end());
    runtime::ContextResourceUsage result;
    for (const auto& reference : states) {
        if (reference.count == program.state_store->checkpoint_references(reference.handle) &&
            program.state_store->can_release_after_checkpoint_references(reference.handle,
                                                                         reference.count)) {
            if (state && program.state_store->device_resident(reference.handle)) {
                ++result.state_slots;
            }
            if (host && program.state_store->host_resident(reference.handle)) {
                result.host_bytes += program.state_images->host_layout().image_bytes;
            }
        }
    }
    if (state && !result.state_slots) { return std::nullopt; }
    for (const auto index : histories) {
        const auto& history = facts.histories[index];
        if (history.references != static_cast<long>(history.checkpoints.size())) { continue; }
        std::uint32_t retained_main = 0, retained_backend = 0;
        for (const auto owner : history.checkpoints) {
            if (std::binary_search(removed.begin(), removed.end(), owner)) { continue; }
            const auto& record = *program.checkpoints[owner].value;
            retained_main      = std::max(retained_main, record.frontier);
            retained_backend   = std::max(retained_backend, record.backend_frontier);
        }
        suffixes.push_back({history.history, retained_main, retained_backend});
    }
    const auto scan = [&](bool main_pool, bool account) {
        const auto* addresses =
            main_pool ? program.text_kv_addresses.get() : program.backend_kv_addresses.get();
        const auto* pages =
            main_pool ? program.text_kv_pages.get() : program.backend_kv_pages.get();
        if (!addresses || !pages) { return true; }
        references.clear();
        const auto account_release = [&](auto page) {
            if (pages->source_pins(page) || pages->active_address_references(page)) {
                return false;
            }
            if (!account) { return true; }
            if (host) {
                if (pages->host_resident(page)) {
                    result.host_bytes += main_pool ? program.text_host_kv_page_stride
                                                   : program.backend_host_kv_page_stride;
                }
            } else if (pages->device_resident(page)) {
                ++(main_pool ? result.main_kv_pages : result.backend_kv_pages);
                released_pages.push_back(page);
            }
            return true;
        };
        for (const auto& suffix : suffixes) {
            if (!main_pool && !suffix.history->backend) { continue; }
            const auto address  = main_pool ? suffix.history->text : *suffix.history->backend;
            const auto retained = main_pool ? suffix.main : suffix.backend;
            bool safe           = true;
            if (!addresses->visit_releasable_inactive_suffix(
                    address, retained,
                    [&](auto page) {
                        // A directory contains each logical handle once. A unique address
                        // reference therefore needs no cross-history alias reduction.
                        if (pages->address_references(page) == 1) {
                            safe = account_release(page) && safe;
                            return;
                        }
                        const bool counted = account && (host ? pages->host_resident(page)
                                                              : pages->device_resident(page));
                        // Joint source-lease checks apply to both pools, including non-target
                        // pages.
                        if (counted || pages->source_pins(page)) {
                            references.push_back({page, pages->descriptor_index(page)});
                        }
                    }) ||
                !safe) {
                return false;
            }
        }
        std::sort(references.begin(), references.end(),
                  [](const auto& a, const auto& b) { return a.descriptor < b.descriptor; });
        for (std::size_t begin = 0; begin < references.size();) {
            auto end = begin + 1;
            while (end < references.size() &&
                   references[end].descriptor == references[begin].descriptor) {
                ++end;
            }
            const auto page  = references[begin].handle;
            const auto count = end - begin;
            begin            = end;
            if (count != pages->address_references(page)) { continue; }
            if (!account_release(page)) { return false; }
        }
        return true;
    };
    if (main) {
        if (!scan(true, true) || !result.main_kv_pages || !scan(false, false)) {
            return std::nullopt;
        }
    } else if (backend) {
        if (!scan(false, true) || !result.backend_kv_pages || !scan(true, false)) {
            return std::nullopt;
        }
    } else if (!scan(true, host) || !scan(false, host)) {
        return std::nullopt;
    }
    return result;
}

} // namespace

// A quote carries content identities and complete optional holders. It holds no execution
// lease: start_demote revalidates it before acquiring real destinations.
struct DemotionPlan {
    const ProgramImpl* owner               = nullptr;
    runtime::ContextResourceClass resource = runtime::ContextResourceClass::State;
    std::vector<CheckpointHandle> allowed;
    std::vector<CheckpointHandle> excluded;
    std::optional<StateDemotion> state;
    std::vector<PageDemotion> pages;
    // Each range retains one component's deep-first selection order. Transfers traverse
    // each range forward in logical position without reversing the component ordering.
    std::vector<std::uint32_t> page_group_ends;
};

std::uint32_t ProgramImpl::checkpoint_recovery_frontier(CheckpointHandle retained,
                                                        const RequestBasePlan& base,
                                                        std::uint32_t target) const {
    if (!valid_checkpoint(retained) || checkpoint(retained).frontier > target ||
        !checkpoint_matches(retained, base)) {
        return 0;
    }
    return checkpoint(retained).frontier;
}

struct ReclaimPlan {
    ReclaimPlan(const ProgramImpl& program, runtime::ContextResourceUsage requested)
        : owner(&program), shortage(requested), facts(program) {}

    const ProgramImpl* owner;
    runtime::ContextResourceUsage shortage;
    PhysicalFacts facts;
    mutable std::unordered_map<std::uint64_t, bool> prefixes;

    bool prefix(CheckpointHandle earlier, CheckpointHandle later) const {
        const auto& a = owner->checkpoint(earlier);
        const auto& b = owner->checkpoint(later);
        if (a.frontier > b.frontier || !a.identity || !b.identity ||
            a.key.identity_tag != b.key.identity_tag ||
            a.identity->digests.at(a.frontier) != b.identity->digests.at(a.frontier)) {
            return false;
        }
        if (a.identity == b.identity) { return true; }
        const auto key   = (static_cast<std::uint64_t>(earlier.index) << 32U) | later.index;
        const auto found = prefixes.find(key);
        if (found != prefixes.end()) { return found->second; }
        const bool matches = checkpoint_prefix(a, b);
        prefixes.emplace(key, matches);
        return matches;
    }
};

struct ReleasePlan {
    std::shared_ptr<const ReclaimPlan> evaluation;
    std::vector<CheckpointHandle> sources;
    runtime::ContextResourceUsage released;
    std::vector<LogicalKVPageHandle> pages;
};

template <typename Prefix>
std::uint64_t recovery_loss(const ProgramImpl& program, std::span<const CheckpointHandle> removed,
                            std::span<const CheckpointHandle> surviving, Prefix&& prefix) {
    std::vector<CheckpointHandle> ordered;
    for (const auto handle : removed) {
        if (program.valid_checkpoint(handle) &&
            std::find(ordered.begin(), ordered.end(), handle) == ordered.end()) {
            ordered.push_back(handle);
        }
    }
    std::sort(ordered.begin(), ordered.end(), [&](auto left, auto right) {
        const auto l = program.checkpoint(left).frontier, r = program.checkpoint(right).frontier;
        return l != r ? l > r : left.index < right.index;
    });

    struct LineageLoss {
        CheckpointHandle deepest;
        std::uint32_t tokens;
    };

    std::vector<LineageLoss> lineages;
    for (const auto handle : ordered) {
        const auto& record     = program.checkpoint(handle);
        std::uint32_t fallback = 0;
        for (const auto retained : surviving) {
            if (!program.valid_checkpoint(retained) ||
                std::find(ordered.begin(), ordered.end(), retained) != ordered.end()) {
                continue;
            }
            const auto& candidate = program.checkpoint(retained);
            if (candidate.frontier > fallback && prefix(retained, handle)) {
                fallback = candidate.frontier;
            }
        }
        const auto lost    = record.frontier - fallback;
        const auto lineage = std::find_if(lineages.begin(), lineages.end(), [&](const auto& item) {
            return prefix(handle, item.deepest);
        });
        if (lineage == lineages.end()) {
            lineages.push_back({handle, lost});
        } else {
            lineage->tokens = std::max(lineage->tokens, lost);
        }
    }
    std::uint64_t result = 0;
    for (const auto& lineage : lineages) { result += lineage.tokens; }
    return result;
}

std::uint64_t
ProgramImpl::checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                                      std::span<const CheckpointHandle> surviving) const {
    return recovery_loss(*this, removed, surviving, [&](auto a, auto b) {
        return checkpoint_prefix(checkpoint(a), checkpoint(b));
    });
}

std::optional<ContextDemotion> quote_demotion(const ProgramImpl& program,
                                              const PhysicalOwners& owners,
                                              std::span<const CheckpointHandle> excluded,
                                              runtime::ContextResourceUsage shortage) {
    if (shortage.host_bytes ||
        (!shortage.state_slots && !shortage.main_kv_pages && !shortage.backend_kv_pages)) {
        return std::nullopt;
    }
    const auto resource = shortage.state_slots     ? runtime::ContextResourceClass::State
                          : shortage.main_kv_pages ? runtime::ContextResourceClass::MainKV
                                                   : runtime::ContextResourceClass::BackendKV;
    auto plan           = std::make_shared<DemotionPlan>();
    plan->owner         = &program;
    plan->resource      = resource;
    plan->allowed       = owners.sources;
    plan->excluded.assign(excluded.begin(), excluded.end());
    ContextDemotion candidate;
    if (resource == runtime::ContextResourceClass::State) {
        for (const auto state : owners.states) {
            if (program.state_store->role(state) != StateImageRole::CheckpointImmutable ||
                !program.state_store->can_release_after_checkpoint_references(
                    state, program.state_store->checkpoint_references(state))) {
                continue;
            }
            const auto host = program.state_store->host_resident(state);
            plan->state     = StateDemotion{state, program.state_store->content_epoch(state), host,
                                        owners.sources};
            candidate.released.state_slots = 1;
            candidate.host_bytes = host ? 0 : program.state_images->host_layout().image_bytes;
            break;
        }
        if (!plan->state) { return std::nullopt; }
    } else {
        const auto& pages = resource == runtime::ContextResourceClass::MainKV
                                ? *program.text_kv_pages
                                : *program.backend_kv_pages;
        const auto& objects =
            resource == runtime::ContextResourceClass::MainKV ? owners.main : owners.backend;
        const auto needed = resource == runtime::ContextResourceClass::MainKV
                                ? shortage.main_kv_pages
                                : shortage.backend_kv_pages;
        const auto stride = resource == runtime::ContextResourceClass::MainKV
                                ? program.text_host_kv_page_stride
                                : program.backend_host_kv_page_stride;
        for (const auto& object : objects) {
            const auto page = object.handle;
            if (pages.active_address_references(page) || pages.source_pins(page) ||
                !pages.can_pin_source(page) ||
                (pages.host_resident(page) && !pages.host_replica_current(page))) {
                continue;
            }
            const auto host = pages.host_resident(page);
            plan->pages.push_back({page, object.position, pages.content_epoch(page),
                                   pages.committed_columns(page), host, owners.sources});
            if (!host) { candidate.host_bytes += stride; }
            if (plan->pages.size() == needed) { break; }
        }
        if (plan->pages.empty()) { return std::nullopt; }
        (resource == runtime::ContextResourceClass::MainKV ? candidate.released.main_kv_pages
                                                           : candidate.released.backend_kv_pages) =
            static_cast<std::uint32_t>(plan->pages.size());
    }
    candidate.sources = owners.sources;
    candidate.impl    = std::move(plan);
    return candidate;
}

std::optional<ContextRelease> quote_release(const std::shared_ptr<ReclaimPlan>& evaluation,
                                            std::span<const CheckpointHandle> sources,
                                            ReleaseScratch& scratch) {
    auto plan           = std::make_shared<ReleasePlan>();
    const auto released = released_resources(*evaluation->owner, evaluation->facts, sources,
                                             evaluation->shortage, scratch, plan->pages);
    if (!released) { return std::nullopt; }
    const auto shortage = evaluation->shortage;
    const auto useful   = shortage.host_bytes         ? released->host_bytes
                          : shortage.state_slots      ? released->state_slots
                          : shortage.main_kv_pages    ? released->main_kv_pages
                          : shortage.backend_kv_pages ? released->backend_kv_pages
                                                      : 0;
    if (!useful) { return std::nullopt; }
    plan->evaluation = evaluation;
    plan->sources.assign(sources.begin(), sources.end());
    plan->released = *released;
    return ContextRelease{plan->sources, *released, std::move(plan)};
}

ContextReclaimPlan ProgramImpl::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                             std::span<const CheckpointHandle> excluded,
                                             runtime::ContextResourceUsage shortage) const {
    ContextReclaimPlan result;
    result.impl_ = std::make_shared<ReclaimPlan>(*this, shortage);
    if (context_transaction_) { return result; }
    const auto owners =
        physical_owner_sets(*this, result.impl_->facts, allowed, excluded, shortage);
    ReleaseScratch scratch;
    for (const auto& group : owners) {
        if (auto quote = quote_demotion(*this, group, excluded, shortage)) {
            result.demotions.push_back(std::move(*quote));
        }
        if (auto quote = quote_release(result.impl_, group.sources, scratch)) {
            result.releases.push_back(std::move(*quote));
        }
    }
    return result;
}

std::vector<ContextRelease>
ProgramImpl::plan_releases(std::span<const CheckpointHandle> allowed,
                           std::span<const CheckpointHandle> excluded,
                           runtime::ContextResourceUsage shortage) const {
    if (context_transaction_) { return {}; }
    auto evaluation   = std::make_shared<ReclaimPlan>(*this, shortage);
    const auto owners = physical_owner_sets(*this, evaluation->facts, allowed, excluded, shortage);
    ReleaseScratch scratch;
    std::vector<ContextRelease> result;
    for (const auto& group : owners) {
        if (auto quote = quote_release(evaluation, group.sources, scratch)) {
            result.push_back(std::move(*quote));
        }
    }
    return result;
}

struct DemotionBatch {
    DemotionBatch(std::shared_ptr<const ReclaimPlan> evaluation,
                  runtime::ContextResourceUsage requested)
        : evaluation(std::move(evaluation)), shortage(requested) {
        if (!this->evaluation || shortage.state_slots || shortage.host_bytes ||
            (!shortage.main_kv_pages && !shortage.backend_kv_pages)) {
            valid = false;
            return;
        }
        const auto original = this->evaluation->shortage;
        resource            = shortage.main_kv_pages ? runtime::ContextResourceClass::MainKV
                                                     : runtime::ContextResourceClass::BackendKV;
        valid               = !original.state_slots && !original.host_bytes &&
                (resource == runtime::ContextResourceClass::MainKV
                     ? original.main_kv_pages != 0
                     : !original.main_kv_pages && original.backend_kv_pages != 0);
    }

    std::shared_ptr<const ReclaimPlan> evaluation;
    runtime::ContextResourceUsage shortage;
    runtime::ContextResourceClass resource = runtime::ContextResourceClass::MainKV;
    bool valid                             = true;
    std::vector<std::shared_ptr<const DemotionPlan>> components;
    std::vector<const PageDemotion*> ordered;
    std::vector<std::uint32_t> group_ends;
    std::unordered_map<std::uint32_t, const PageDemotion*> selected;

    const LogicalKVPageStore& page_store() const {
        return resource == runtime::ContextResourceClass::MainKV
                   ? *evaluation->owner->text_kv_pages
                   : *evaluation->owner->backend_kv_pages;
    }

    bool append(const ContextDemotion& quote) {
        if (!valid) { return false; }
        const auto& program = *evaluation->owner;
        if (program.has_context_transaction() || !quote.impl || quote.impl->owner != &program ||
            quote.impl->state || quote.impl->resource != resource || quote.impl->pages.empty() ||
            quote.sources != quote.impl->allowed ||
            (!components.empty() && components.front()->excluded != quote.impl->excluded)) {
            valid = false;
            return false;
        }
        for (const auto source : quote.sources) {
            if (!program.valid_checkpoint(source) || program.checkpoints[source.index].pins) {
                valid = false;
                return false;
            }
        }
        const auto& pages     = page_store();
        const auto& component = *quote.impl;
        std::size_t group     = 0;
        auto end              = component.page_group_ends.empty()
                                    ? static_cast<std::uint32_t>(component.pages.size())
                                    : component.page_group_ends.front();
        for (std::uint32_t i = 0; i < component.pages.size(); ++i) {
            const auto& page = component.pages[i];
            if (!pages.valid(page.handle) || !pages.device_resident(page.handle) ||
                pages.active_address_references(page.handle) || pages.source_pins(page.handle) ||
                !pages.can_pin_source(page.handle) ||
                pages.content_epoch(page.handle) != page.content_epoch ||
                pages.committed_columns(page.handle) != page.committed_columns ||
                pages.host_resident(page.handle) != page.host_resident ||
                (page.host_resident && !pages.host_replica_current(page.handle))) {
                valid = false;
                return false;
            }
            const auto [entry, added] =
                selected.emplace(pages.descriptor_index(page.handle), &page);
            if (!added && *entry->second != page) {
                valid = false;
                return false;
            }
            if (added) { ordered.push_back(&page); }
            if (i + 1 == end) {
                const auto count = static_cast<std::uint32_t>(ordered.size());
                if (count && (group_ends.empty() || group_ends.back() != count)) {
                    group_ends.push_back(count);
                }
                if (++group < component.page_group_ends.size()) {
                    end = component.page_group_ends[group];
                }
            }
        }
        components.push_back(quote.impl);
        return true;
    }

    bool covers(const ContextRelease& quote) const {
        if (!valid || ordered.empty() || !quote.impl || quote.impl->evaluation != evaluation ||
            quote.sources != quote.impl->sources || quote.released != quote.impl->released ||
            quote.impl->pages.empty()) {
            return false;
        }
        const auto requested = quote.impl->evaluation->shortage;
        if (requested.state_slots || requested.host_bytes ||
            (resource == runtime::ContextResourceClass::MainKV
                 ? !requested.main_kv_pages
                 : requested.main_kv_pages || !requested.backend_kv_pages)) {
            return false;
        }
        const auto& program = *evaluation->owner;
        if (program.has_context_transaction()) { return false; }
        for (const auto source : quote.sources) {
            if (!program.valid_checkpoint(source) || program.checkpoints[source.index].pins) {
                return false;
            }
        }
        const auto& pages = page_store();
        for (const auto page : quote.impl->pages) {
            if (!pages.valid(page) || !pages.device_resident(page)) { return false; }
            const auto found = selected.find(pages.descriptor_index(page));
            if (found == selected.end() || found->second->handle != page) { return false; }
        }
        return true;
    }

    std::optional<ContextDemotion> finish() const {
        if (!valid || ordered.empty() || evaluation->owner->has_context_transaction()) {
            return std::nullopt;
        }
        const auto& program  = *evaluation->owner;
        const auto needed    = resource == runtime::ContextResourceClass::MainKV
                                   ? shortage.main_kv_pages
                                   : shortage.backend_kv_pages;
        const auto stride    = resource == runtime::ContextResourceClass::MainKV
                                   ? program.text_host_kv_page_stride
                                   : program.backend_host_kv_page_stride;
        const auto usage     = program.physical_usage();
        const auto available = usage.capacity.host_bytes - usage.occupied.host_bytes;
        auto plan            = std::make_shared<DemotionPlan>();
        plan->owner          = &program;
        plan->resource       = resource;
        plan->excluded       = components.front()->excluded;
        ContextDemotion result;
        for (const auto* page : ordered) {
            const auto bytes = page->host_resident ? 0 : stride;
            if (plan->pages.size() == needed || bytes > available - result.host_bytes) { break; }
            plan->pages.push_back(*page);
            result.host_bytes += bytes;
            plan->allowed.insert(plan->allowed.end(), page->references.begin(),
                                 page->references.end());
        }
        if (plan->pages.empty()) { return std::nullopt; }
        const auto count = static_cast<std::uint32_t>(plan->pages.size());
        for (const auto end : group_ends) {
            if (end < count) { plan->page_group_ends.push_back(end); }
        }
        plan->page_group_ends.push_back(count);
        std::sort(plan->allowed.begin(), plan->allowed.end(),
                  [](auto a, auto b) { return a.index < b.index; });
        plan->allowed.erase(std::unique(plan->allowed.begin(), plan->allowed.end()),
                            plan->allowed.end());
        result.sources = plan->allowed;
        (resource == runtime::ContextResourceClass::MainKV ? result.released.main_kv_pages
                                                           : result.released.backend_kv_pages) =
            count;
        result.impl = std::move(plan);
        return result;
    }
};

bool ProgramImpl::start_demote(const ContextDemotion& quote) {
    if (!quote.impl || quote.impl->owner != this || context_transaction_) { return false; }
    const auto& plan = *quote.impl;
    if (quote.sources != plan.allowed) { return false; }
    for (const auto source : plan.allowed) {
        if (!valid_checkpoint(source) || checkpoints[source.index].pins ||
            std::find(plan.excluded.begin(), plan.excluded.end(), source) != plan.excluded.end()) {
            return false;
        }
    }
    // Validate actual current checkpoint holders before acquiring any destination or lease.
    // KV aliases preserve logical positions: fork/view copy a prefix in place, COW creates
    // a new handle, growth appends fresh handles, and truncation only removes a suffix.
    if (plan.state) {
        if (quote.released != runtime::ContextResourceUsage{.state_slots = 1}) { return false; }
        const auto& expected = *plan.state;
        std::size_t matched  = 0;
        for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
            const auto& slot = checkpoints[index];
            if (!slot.value || slot.value->state != expected.handle) { continue; }
            if (matched == expected.references.size() ||
                expected.references[matched++] != CheckpointHandle{this, index, slot.generation}) {
                return false;
            }
        }
        if (matched != expected.references.size() ||
            state_store->role(expected.handle) != StateImageRole::CheckpointImmutable ||
            !state_store->device_resident(expected.handle) ||
            state_store->content_epoch(expected.handle) != expected.content_epoch ||
            state_store->host_resident(expected.handle) != expected.host_resident ||
            !state_store->can_release_after_checkpoint_references(
                expected.handle, state_store->checkpoint_references(expected.handle)) ||
            quote.host_bytes !=
                (expected.host_resident ? 0 : state_images->host_layout().image_bytes)) {
            return false;
        }
    } else {
        const bool main       = plan.resource == runtime::ContextResourceClass::MainKV;
        const auto* pages     = main ? text_kv_pages.get() : backend_kv_pages.get();
        const auto* addresses = main ? text_kv_addresses.get() : backend_kv_addresses.get();
        if (!pages || !addresses || plan.pages.empty()) { return false; }
        runtime::ContextResourceUsage released;
        (main ? released.main_kv_pages : released.backend_kv_pages) =
            static_cast<std::uint32_t>(plan.pages.size());
        if (quote.released != released) { return false; }

        struct Holder {
            CheckpointHandle checkpoint;
            KVAddressSpaceHandle address;
            std::uint32_t pages;
        };

        std::vector<Holder> holders;
        holders.reserve(checkpoints.size());
        for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
            const auto& slot = checkpoints[index];
            if (!slot.value || (!main && !slot.value->kv->backend)) { continue; }
            const auto& record = *slot.value;
            holders.push_back(
                {{this, index, slot.generation},
                 main ? record.kv->text : *record.kv->backend,
                 kv_pages_for_frontier(main ? record.frontier : record.backend_frontier)});
        }
        const auto stride      = main ? text_host_kv_page_stride : backend_host_kv_page_stride;
        std::size_t host_bytes = 0;
        std::vector<std::uint32_t> descriptors;
        descriptors.reserve(plan.pages.size());
        for (const auto& page : plan.pages) {
            if (!pages->valid(page.handle) || !pages->device_resident(page.handle) ||
                pages->active_address_references(page.handle) || pages->source_pins(page.handle) ||
                !pages->can_pin_source(page.handle) ||
                pages->content_epoch(page.handle) != page.content_epoch ||
                pages->committed_columns(page.handle) != page.committed_columns ||
                pages->host_resident(page.handle) != page.host_resident ||
                (page.host_resident && !pages->host_replica_current(page.handle))) {
                return false;
            }
            std::size_t matched = 0;
            for (const auto& holder : holders) {
                if (holder.pages <= page.position ||
                    addresses->logical_page(holder.address, page.position) != page.handle) {
                    continue;
                }
                if (matched == page.references.size() ||
                    page.references[matched++] != holder.checkpoint) {
                    return false;
                }
            }
            if (matched != page.references.size()) { return false; }
            descriptors.push_back(pages->descriptor_index(page.handle));
            if (!page.host_resident) { host_bytes += stride; }
        }
        std::sort(descriptors.begin(), descriptors.end());
        if (std::adjacent_find(descriptors.begin(), descriptors.end()) != descriptors.end() ||
            host_bytes != quote.host_bytes) {
            return false;
        }
    }

    ContextTransaction transaction;
    transaction.kind                  = ContextOperationKind::Demote;
    transaction.preserve_state_device = false;
    if (plan.state) {
        if (plan.state->host_resident) {
            return state_store->drop_device_replica(plan.state->handle);
        }
        auto transfer = state_store->reserve_device_to_host(plan.state->handle);
        if (!transfer) { return false; }
        transaction.state_transfer.emplace(std::move(*transfer));
    } else {
        auto& pages = plan.resource == runtime::ContextResourceClass::MainKV ? *text_kv_pages
                                                                             : *backend_kv_pages;
        std::vector<LogicalKVPageHandle> missing;
        std::size_t begin       = 0;
        const auto append_group = [&](std::size_t end) {
            for (auto index = end; index != begin; --index) {
                const auto& page = plan.pages[index - 1];
                if (!page.host_resident) { missing.push_back(page.handle); }
            }
            begin = end;
        };
        if (plan.page_group_ends.empty()) {
            append_group(plan.pages.size());
        } else {
            for (const auto end : plan.page_group_ends) { append_group(end); }
        }
        if (!missing.empty() && !host_kv_extents) { return false; }
        std::size_t offset = 0;
        while (offset < missing.size()) {
            auto count = missing.size() - offset;
            std::optional<HostKVExtentReservation> destination;
            while (count) {
                auto reservation =
                    host_kv_extents->prepare(pages, std::span(missing).subspan(offset, count));
                if (reservation) {
                    destination.emplace(std::move(*reservation));
                    break;
                }
                count /= 2;
            }
            // All destinations and source leases belong to this local transaction until
            // every component is ready. A failed reservation unwinds without dropping pages.
            if (!destination) { return false; }
            KVTransfer transfer;
            transfer.pages       = &pages;
            transfer.resource    = plan.resource;
            transfer.drop_device = true;
            transfer.logical.assign(missing.begin() + offset, missing.begin() + offset + count);
            transfer.physical = host_kv_extents->device_sources(*destination);
            transfer.host_destination.emplace(std::move(*destination));
            transaction.kv_transfers.push_back(std::move(transfer));
            offset += count;
        }
        // Complete Host replicas already preserve these contents. Once the entire batch is
        // reserved, release their Device copies synchronously, as for a single demotion.
        for (const auto& page : plan.pages) {
            if (page.host_resident && !pages.drop_device_replica(page.handle)) {
                throw std::logic_error("quoted redundant Device page changed during demotion");
            }
        }
        if (transaction.kv_transfers.empty()) { return true; }
    }

    context_transaction_.emplace(std::move(transaction));
    try {
        context_source_ready_.record(device.stream);
        context_source_ready_.wait(device.transfer_stream);
        enqueue_state_backup(*context_transaction_);
        enqueue_context_transfers(*context_transaction_);
    } catch (...) {
        abort_context();
        throw;
    }
    return true;
}

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5 {

ContextDemotionBatch::ContextDemotionBatch(std::unique_ptr<detail::DemotionBatch> impl)
    : impl_(std::move(impl)) {}

ContextDemotionBatch::ContextDemotionBatch(ContextDemotionBatch&&) noexcept            = default;
ContextDemotionBatch& ContextDemotionBatch::operator=(ContextDemotionBatch&&) noexcept = default;
ContextDemotionBatch::~ContextDemotionBatch()                                          = default;

bool ContextDemotionBatch::append(const ContextDemotion& quote) {
    return impl_ && impl_->append(quote);
}

bool ContextDemotionBatch::covers(const ContextRelease& quote) const {
    return impl_ && impl_->covers(quote);
}

std::optional<ContextDemotion> ContextDemotionBatch::finish() const {
    return impl_ ? impl_->finish() : std::nullopt;
}

ContextDemotionBatch
ContextReclaimPlan::begin_kv_batch(runtime::ContextResourceUsage shortage) const {
    return ContextDemotionBatch(std::make_unique<detail::DemotionBatch>(impl_, shortage));
}

std::uint64_t ContextReclaimPlan::recovery_loss(std::span<const CheckpointHandle> removed,
                                                std::span<const CheckpointHandle> surviving) const {
    if (!impl_) { throw std::logic_error("recovery query has no Native evaluation"); }
    return detail::recovery_loss(*impl_->owner, removed, surviving,
                                 [&](auto a, auto b) { return impl_->prefix(a, b); });
}

} // namespace ninfer::models::qwen3_5
