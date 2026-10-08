#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "models/qwen3_5/frontend/graft.h"
#include "runtime/contract/token_constraint.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen3_5 {

namespace execution {
class Parameters;
}

namespace detail {
struct CaptureAssessmentImpl;

} // namespace detail

// Read-only diagnostics sampled from the real Program stores.  This is not an accounting input.
struct PhysicalUsageSnapshot {
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t device_state_slots      = 0;
    std::uint32_t host_state_slots        = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalUsageSnapshot&,
                                                   const PhysicalUsageSnapshot&) noexcept = default;
};

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

// Program-minted shortlist metadata. It only narrows catalog inspection; Program still performs
// exact token, position, media and runtime-mode verification before a checkpoint can be selected.
struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefixShortlistKey,
                                                   PrefixShortlistKey) noexcept = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::uint32_t main_pages       = 0;
    std::uint32_t backend_pages    = 0;

    [[nodiscard]] friend constexpr bool operator==(TargetKVRequirement,
                                                   TargetKVRequirement) noexcept = default;
};

struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] friend bool operator==(const CheckpointSummary&,
                                         const CheckpointSummary&) noexcept = default;
};

struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint;
    std::optional<CheckpointSummary> rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const ContinuationSummary&,
                                         const ContinuationSummary&) noexcept = default;
};

// Where the bulk of a snapshot lives inside its bytes: one region per StateImage and one each for
// the Text and backend KV pages. Everything else (the header with the token ledger, the checkpoint
// directory and the integrity trailer) is small. Two snapshots of one conversation share almost all
// of these regions byte for byte, which is what lets a store keep them once.
struct SessionSnapshotRegion {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
};

// A retained continuation serialized for disk: the session snapshot bytes, the resident depth
// and the session digest (FNV-1a 64 of the token ledger, 16 hex characters).
struct SessionSnapshot {
    std::vector<std::uint8_t> bytes;
    std::uint32_t tokens = 0;
    std::string session_digest;
    // The bulk regions of `bytes`, in order and disjoint.
    std::vector<SessionSnapshotRegion> regions;
    // The prefix key of every checkpoint the snapshot restores (endpoint, rewrite, long anchors),
    // exactly as a later request computes it for its own prompt, so the snapshot can be found by
    // prefix without being opened.
    std::vector<PrefixShortlistKey> checkpoints;
    // The session's own prefix digest at every frontier 0..tokens, so a store can tell that an
    // older image is an earlier state of the same conversation without opening either.
    std::vector<std::array<std::uint64_t, 2>> prefix_digests;
};

struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const SharedPrefixSummary&,
                                         const SharedPrefixSummary&) noexcept = default;
};

namespace detail {

struct SequencePlanImpl;

struct SequencePlannerImpl;

struct AdmissionCandidateImpl;

struct CapturePressureCandidateImpl;

struct RequestBasePlanImpl;

struct PressurePlanningSessionImpl;

class ProgramImpl;

struct RuntimeContractAccess;
} // namespace detail

class SequencePlanner;

class Program;

class PressurePlanningSession;

class CapturePressurePlanningSession;

class CapturePressurePlan;

class CapturePressureCandidate;

// Concrete Qwen execution and resource contracts; model instances supply their own data.
// target selection remains outside this layer and happens once in the closed Engine registry.

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    // What each further device reserves (device 1 first). Empty on one device.
    [[nodiscard]] std::span<const std::size_t> extra_rank_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    // Pinned host bytes of one StateImage slot, which sizes the Host state tier.
    [[nodiscard]] std::size_t host_state_image_bytes() const noexcept;
    // Replaces the Host-side context-cache capacities. Device memory does not depend on them, so the
    // plan stays valid; the Engine uses this to size them once the host's free memory is known.
    void set_host_context_cache(const ContextCacheOptions& resolved);

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;

    friend class detail::ProgramImpl;
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                 const EngineOptions&, std::uint32_t);
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;
    [[nodiscard]] std::optional<runtime::PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl> impl_;
};

class AdmissionCandidate {
public:
    AdmissionCandidate(AdmissionCandidate&&) noexcept;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept;
    ~AdmissionCandidate();

    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept;
    [[nodiscard]] std::optional<std::uint32_t> graft_shared_slot() const noexcept;

public:
    // Family-private construction/storage seam. Exact packages expose only the completed alias;
    // Engine code can inspect summary() but not target planning state.
    explicit AdmissionCandidate(std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::AdmissionCandidateImpl> impl_;

    friend class Program;
    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

// Family-private owning wrapper for the physical capture-pressure candidate. It intentionally has
// no request summary or admission API.

class CapturePressureCandidate {
public:
    CapturePressureCandidate(CapturePressureCandidate&&) noexcept;
    CapturePressureCandidate& operator=(CapturePressureCandidate&&) noexcept;
    ~CapturePressureCandidate();

    CapturePressureCandidate(const CapturePressureCandidate&)            = delete;
    CapturePressureCandidate& operator=(const CapturePressureCandidate&) = delete;

public:
    explicit CapturePressureCandidate(
        std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl_;

    friend class Program;
    friend class PressurePlanningSession;
    friend class CapturePressurePlan;
    friend struct detail::PressurePlanningSessionImpl;
};

// A sealed pressure-only post-state for one active capture. Its payload is Program-private and
// cannot be inspected or executed through the request-admission interface.

class CapturePressurePlan {
public:
    CapturePressurePlan(CapturePressurePlan&&) noexcept            = default;
    CapturePressurePlan& operator=(CapturePressurePlan&&) noexcept = default;
    ~CapturePressurePlan()                                         = default;

    CapturePressurePlan(const CapturePressurePlan&)            = delete;
    CapturePressurePlan& operator=(const CapturePressurePlan&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    CapturePressurePlan(CapturePressureCandidate&& pressure,
                        runtime::ProgramResourceRevision revision) noexcept
        : pressure_(std::move(pressure)), revision_(revision) {}

    CapturePressureCandidate pressure_;
    runtime::ProgramResourceRevision revision_;

    friend class Program;
    friend class PressurePlanningSession;
};

// A sealed Program-owned physical decision.  ResourceManager may retain it and inspect the
// request-level summary, but cannot see allocator quantities, references, reservations, or stage
// deltas.  Start validates the bound Program revision before performing any mutation.

class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept            = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ~ResourcePlan()                                  = default;

    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return admission_.summary();
    }

    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    ResourcePlan(AdmissionCandidate&& admission, runtime::ProgramResourceRevision revision,
                 bool needs_transfer) noexcept
        : admission_(std::move(admission)), revision_(revision), needs_transfer_(needs_transfer) {}

    AdmissionCandidate admission_;
    runtime::ProgramResourceRevision revision_;
    bool needs_transfer_ = false;

    friend class Program;
    friend class PressurePlanningSession;
};

// A Program-minted proof that one FIFO borrower cannot consume the maximum physical entitlement
// reserved for the blocked head.  Common scheduling binds the opaque proof to logical identities
// and a revision; it cannot inspect or reproduce the resource arithmetic.

class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    explicit PersistentBackfillProof(runtime::ProgramResourceRevision revision) noexcept
        : revision_(revision) {}

    runtime::ProgramResourceRevision revision_;

    friend class Program;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess;
};

class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&&)      = delete;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    SharedPrefixHandle& operator=(SharedPrefixHandle&&)      = delete;
    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

class PressureTargetHandle {
public:
    PressureTargetHandle() noexcept = default;

    [[nodiscard]] friend constexpr bool operator==(PressureTargetHandle,
                                                   PressureTargetHandle) noexcept = default;

private:
    const void* session_      = nullptr;
    std::uint32_t generation_ = 0;
    std::uint32_t index_      = 0;

    friend struct detail::PressurePlanningSessionImpl;

    friend class PressurePlanningSession;
};

class PressureConstructionCursor {
public:
    PressureConstructionCursor(PressureConstructionCursor&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)), slot_(other.slot_),
          generation_(other.generation_), release_(other.release_) {}

    PressureConstructionCursor& operator=(PressureConstructionCursor&&)      = delete;
    PressureConstructionCursor(const PressureConstructionCursor&)            = delete;
    PressureConstructionCursor& operator=(const PressureConstructionCursor&) = delete;

    ~PressureConstructionCursor() {
        if (session_) { release_(session_, slot_, generation_); }
    }

private:
    PressureConstructionCursor(const void* session, std::uint32_t slot, std::uint32_t generation,
                               void (*release)(const void*, std::uint32_t, std::uint32_t) noexcept)
        : session_(session), slot_(slot), generation_(generation), release_(release) {}

    const void* session_;
    std::uint32_t slot_;
    std::uint32_t generation_;
    void (*release_)(const void*, std::uint32_t, std::uint32_t) noexcept;

    friend struct detail::PressurePlanningSessionImpl;
};

class AssessedPressureTarget {
public:
    AssessedPressureTarget(AssessedPressureTarget&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          target_index_(other.target_index_), assessment_(other.assessment_),
          assessment_slot_(std::exchange(other.assessment_slot_, 0)),
          assessment_slot_generation_(std::exchange(other.assessment_slot_generation_, 0)),
          release_slot_(std::exchange(other.release_slot_, nullptr)),
          executable_(std::move(other.executable_)),
          capture_executable_(std::move(other.capture_executable_)) {}

    ~AssessedPressureTarget() { reset(); }

    AssessedPressureTarget& operator=(AssessedPressureTarget&& other) noexcept {
        if (this == &other) { return *this; }
        reset();
        session_                    = std::exchange(other.session_, nullptr);
        session_generation_         = std::exchange(other.session_generation_, 0);
        target_index_               = other.target_index_;
        assessment_                 = other.assessment_;
        assessment_slot_            = std::exchange(other.assessment_slot_, 0);
        assessment_slot_generation_ = std::exchange(other.assessment_slot_generation_, 0);
        release_slot_               = std::exchange(other.release_slot_, nullptr);
        executable_                 = std::move(other.executable_);
        capture_executable_         = std::move(other.capture_executable_);
        return *this;
    }

    AssessedPressureTarget(const AssessedPressureTarget&)            = delete;
    AssessedPressureTarget& operator=(const AssessedPressureTarget&) = delete;

    [[nodiscard]] const runtime::PressureTargetAssessment& assessment() const noexcept {
        return assessment_;
    }

private:
    AssessedPressureTarget(const void* session, std::uint32_t session_generation,
                           std::uint32_t target_index, runtime::PressureTargetAssessment assessment,
                           std::uint32_t assessment_slot, std::uint32_t assessment_slot_generation,
                           void (*release_slot)(const void*, std::uint32_t, std::uint32_t) noexcept,
                           std::optional<AdmissionCandidate>&& executable,
                           std::optional<CapturePressureCandidate>&& capture_executable) noexcept
        : session_(session), session_generation_(session_generation), target_index_(target_index),
          assessment_(assessment), assessment_slot_(assessment_slot),
          assessment_slot_generation_(assessment_slot_generation), release_slot_(release_slot),
          executable_(std::move(executable)), capture_executable_(std::move(capture_executable)) {}

    void reset() noexcept {
        if (session_ != nullptr && release_slot_ != nullptr) {
            release_slot_(session_, assessment_slot_, assessment_slot_generation_);
        }
        session_                    = nullptr;
        session_generation_         = 0;
        assessment_slot_            = 0;
        assessment_slot_generation_ = 0;
        release_slot_               = nullptr;
    }

    const void* session_              = nullptr;
    std::uint32_t session_generation_ = 0;
    std::uint32_t target_index_       = 0;
    runtime::PressureTargetAssessment assessment_;
    std::uint32_t assessment_slot_                                            = 0;
    std::uint32_t assessment_slot_generation_                                 = 0;
    void (*release_slot_)(const void*, std::uint32_t, std::uint32_t) noexcept = nullptr;
    std::optional<AdmissionCandidate> executable_;
    std::optional<CapturePressureCandidate> capture_executable_;

    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

class PreparedPressureExpansion {
public:
    PreparedPressureExpansion(PreparedPressureExpansion&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          scratch_generation_(std::exchange(other.scratch_generation_, 0)),
          parent_index_(other.parent_index_), new_canonical_count_(other.new_canonical_count_) {}

    PreparedPressureExpansion& operator=(PreparedPressureExpansion&&)      = delete;
    PreparedPressureExpansion(const PreparedPressureExpansion&)            = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept {
        return new_canonical_count_;
    }

private:
    PreparedPressureExpansion(const void* session, std::uint32_t session_generation,
                              std::uint32_t scratch_generation, std::uint32_t parent_index,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), session_generation_(session_generation),
          scratch_generation_(scratch_generation), parent_index_(parent_index),
          new_canonical_count_(new_canonical_count) {}

    const void* session_               = nullptr;
    std::uint32_t session_generation_  = 0;
    std::uint32_t scratch_generation_  = 0;
    std::uint32_t parent_index_        = 0;
    std::uint32_t new_canonical_count_ = 0;

    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
    bool complete                     = true;
};

class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept;
    ~PressurePlanningSession();

    PressurePlanningSession(const PressurePlanningSession&)            = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle
    identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] PressureTargetHandle maximal_target(runtime::PlanningCandidateId candidate);
    [[nodiscard]] PressureConstructionCursor begin_construction(PressureTargetHandle target,
                                                                bool restore = false);
    [[nodiscard]] runtime::PressureConstructionStep
    next_construction_option(PressureConstructionCursor& cursor);
    void choose_construction(PressureConstructionCursor& cursor,
                             runtime::PressureConstructionOptionId option);
    [[nodiscard]] std::optional<PressureTargetHandle>
    construction_target(const PressureConstructionCursor& cursor);
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion
    prepare_expansion(PressureTargetHandle parent,
                      std::uint32_t maximum_owners = std::numeric_limits<std::uint32_t>::max());
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AssessedPressureTarget& assessed,
                                      const PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<ResourcePlan> seal(AssessedPressureTarget&& assessed,
                                                   const PreparedPrompt& prompt,
                                                   runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<CapturePressurePlan>
    seal_capture(AssessedPressureTarget&& assessed);

private:
    explicit PressurePlanningSession(
        std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept;

    std::unique_ptr<detail::PressurePlanningSessionImpl> impl_;

    friend class Program;
};

// Typed pressure domain for one active capture. The capture candidate remains Program-owned and
// cannot be inspected, sealed, or executed as a request admission candidate.

class CapturePressurePlanningSession {
public:
    CapturePressurePlanningSession(CapturePressurePlanningSession&&) noexcept;
    CapturePressurePlanningSession& operator=(CapturePressurePlanningSession&&) noexcept;
    ~CapturePressurePlanningSession();

    CapturePressurePlanningSession(const CapturePressurePlanningSession&)            = delete;
    CapturePressurePlanningSession& operator=(const CapturePressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle identity_target() const;
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent);
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] std::optional<CapturePressurePlan> seal(AssessedPressureTarget&& assessed);

    [[nodiscard]] static constexpr runtime::PlanningCandidateId candidate_id() noexcept {
        return runtime::PlanningCandidateId{.value = 0};
    }

private:
    CapturePressurePlanningSession(CapturePressureCandidate&& candidate,
                                   PressurePlanningSession&& session) noexcept
        : candidate_(std::move(candidate)), session_(std::move(session)) {}

    CapturePressureCandidate candidate_;
    PressurePlanningSession session_;

    friend class Program;
};

class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_), epoch_(other.epoch_),
          id_(std::exchange(other.id_, 0)) {}

    CaptureOffer& operator=(CaptureOffer&&)      = delete;
    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;

    friend struct detail::RuntimeContractAccess;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;

    friend struct detail::RuntimeContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

enum class CaptureStatePlacement : std::uint8_t {
    DeviceFork,
    HostSnapshot,
};

struct CaptureAssessment {
    CaptureAssessment();

    // Program retains the physical assessment in this opaque package-private payload.
    std::shared_ptr<detail::CaptureAssessmentImpl> implementation;
    PrefixShortlistKey shortlist_key;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
    runtime::PrefillWork protected_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> projected_recovery_work;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier                = 0;
    bool publishes_private                = false;
    bool publishes_shared                 = false;
    bool needs_transfer                   = false;
    bool physically_feasible              = false;
    bool recycles_private_state           = false;
    CaptureStatePlacement state_placement = CaptureStatePlacement::DeviceFork;
};

struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

struct MaterializationVictimResult;
struct MaterializationSharedVictimResult;

struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationSourceResult {
    runtime::PrivateSourceMode mode = runtime::PrivateSourceMode::Retain;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedSourceResult {
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult, ActiveCaptureResult>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    // A prompt-frontier capture becomes valid only after the generated Begin token is committed.
    // Keeping the move-only capability row-aligned avoids exposing provisional prompt state.
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine owns scheduling and logical residency policy. Program owns physical lanes, opaque
    // capabilities, model state and one immutable pending transaction at a time.
    [[nodiscard]] RequestBasePlan plan_request(const PreparedPrompt& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] std::optional<ResourcePlan> seal_identity(const AdmissionCandidate& candidate,
                                                            const PreparedPrompt& prompt,
                                                            runtime::FinalScheduleIntent intent);
    [[nodiscard]] PressurePlanningSession
    begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                            std::span<const runtime::PlanningCandidateId> candidate_ids,
                            std::span<const ContinuationHandle* const> private_owners,
                            std::span<const runtime::PlanningOwnerId> private_owner_ids,
                            std::span<const SharedPrefixHandle* const> shared_owners,
                            std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                      const PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head, const ResourcePlan& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    // True while the sequence's next media item is still encoding in a concurrent overlay Vision
    // window; the Engine gives that lane no prefill unit until it completes.
    [[nodiscard]] bool vision_pending(SequenceHandle sequence) const noexcept;
    // `constraint` (borrowed for this call) restricts the token sampled at prompt completion.
    [[nodiscard]] PrefillProgress
    advance_prefill(SequenceHandle sequence, runtime::TokenMaskSource* constraint = nullptr,
                    runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] CapturePressurePlanningSession
    begin_capture_pressure_planning(const CaptureAssessment& assessment,
                                    std::span<const ContinuationHandle* const> private_owners,
                                    std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                    std::span<const SharedPrefixHandle* const> shared_owners,
                                    std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressurePlan&& pressure, runtime::CancellationFlagView cancellation);
    // `constraints` is empty or row-aligned with `sequences`; each non-null entry (borrowed for
    // this call) restricts every token that row samples, including every speculative
    // verification column.
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      std::span<runtime::TokenMaskSource* const> constraints = {},
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    // Advance each live sequence with its exact target-owned token row. This does not sample or
    // advance sampler RNG/occurrence state; callers own output publication and budget accounting.
    // Each optional execution split is relative to its row's forced-token span.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;

    // Session persistence. Both run only when no context transaction is open. Save copies a
    // catalogued continuation to host bytes without changing it. Restore builds a new catalogued
    // continuation from bytes a server with the same model binding and execution configuration
    // saved; the caller adopts the returned handle into its catalog or releases it.
    [[nodiscard]] SessionSnapshot save_continuation(const ContinuationHandle& continuation,
                                                    std::string_view model_binding);
    [[nodiscard]] ContinuationHandle restore_continuation(std::span<const std::uint8_t> snapshot,
                                                          std::string_view model_binding);
    [[nodiscard]] std::uint32_t
    continuation_depth(const ContinuationHandle& continuation) const noexcept;
    [[nodiscard]] std::string continuation_digest(const ContinuationHandle& continuation) const;
    [[nodiscard]] std::vector<SlotCheckpoint>
    continuation_checkpoints(const ContinuationHandle& continuation) const;
    [[nodiscard]] ContinuationSummary
    continuation_summary(const ContinuationHandle& continuation) const;

    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    // Largest output budget for a prompt of `prompt_tokens` whose admission entitlement (main KV and
    // any MTP/DFlash backend KV, draft window included) fits one lane's share of each pool, so that
    // every configured lane can hold such a request at once; clamped to the remaining context. Reads
    // only fixed startup capacities, so any thread may call it.
    [[nodiscard]] std::uint32_t concurrent_output_budget(std::uint32_t prompt_tokens) const noexcept;
    void reset_memory_peaks() noexcept;

    // Inject a direct_kv or softprompt_kv graft into a synthesized shared-prefix entry.
    // Called once at startup before any request is admitted.
    void inject_graft(const PromptGraft& graft);

    struct GraftCatalogEntry {
        std::string name;
        SharedPrefixHandle handle;
        SharedPrefixSummary summary;
    };
    [[nodiscard]] std::vector<GraftCatalogEntry> graft_catalog_entries();
    void set_graft_rm_slot(const std::string& name, std::uint32_t rm_slot);

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;

    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

namespace detail {

struct RuntimeContractAccess {
    [[nodiscard]] static SequenceHandle make_sequence(const void* owner, runtime::LaneId lane,
                                                      std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static ContinuationHandle
    make_continuation(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        ContinuationHandle out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static SharedPrefixHandle
    make_shared_prefix(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        SharedPrefixHandle out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static CaptureOffer make_capture_offer(const void* owner, runtime::LaneId lane,
                                                         std::uint64_t epoch,
                                                         std::uint64_t id) noexcept {
        CaptureOffer out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        out.id_    = id;
        return out;
    }

    [[nodiscard]] static const void* owner(const SequenceHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static const void* owner(const CaptureOffer& offer) noexcept {
        return offer.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const CaptureOffer& offer) noexcept {
        return offer.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const CaptureOffer& offer) noexcept {
        return offer.epoch_;
    }

    [[nodiscard]] static std::uint64_t id(const CaptureOffer& offer) noexcept { return offer.id_; }

    static void consume(CaptureOffer& offer) noexcept {
        offer.owner_ = nullptr;
        offer.id_    = 0;
    }

    [[nodiscard]] static runtime::LaneId lane(const SequenceHandle& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SequenceHandle& handle) noexcept {
        return handle.epoch_;
    }

    [[nodiscard]] static const void* owner(const ContinuationHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const ContinuationHandle& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const ContinuationHandle& handle) noexcept {
        return handle.generation_;
    }

    static void consume(ContinuationHandle& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static const void* owner(const SharedPrefixHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const SharedPrefixHandle& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SharedPrefixHandle& handle) noexcept {
        return handle.generation_;
    }

    static void consume(SharedPrefixHandle& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                 std::span<const TokenId> tokens, std::span<const std::int32_t> row_counts,
                 std::uint32_t row_stride, runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }
};

} // namespace detail

[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                                    DeviceContext& device,
                                                    const EngineOptions& options,
                                                    std::uint32_t resident_main_pages = 0);

// Overlay Vision residency: sizes one encode window for these options, checks that the evictable
// weight tail covers it and captures the weight pool's window mirror. Call once after load and
// before sequence planning, so the pinned mirror is already charged when KV capacity resolves.
// Returns the window capacity in bytes, or zero under resident residency.
[[nodiscard]] std::size_t prepare_vision_overlay(const execution::Parameters& parameters,
                                                 DeviceContext& device,
                                                 const EngineOptions& options);

[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                                      SequencePlan&& plan, DeviceContext& device,
                                                      const StartupObserver& startup_observer);

} // namespace ninfer::models::qwen3_5
