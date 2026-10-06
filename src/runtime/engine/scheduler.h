#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/engine/admission_policy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::runtime {

template <class Request>
class Scheduler {
public:
    using RequestPtr     = std::shared_ptr<Request>;
    using SequenceHandle = typename Request::SequenceHandle;

    enum class ExecutionAction : std::uint8_t {
        Prefill,
        Decode,
        Wait,
    };

    class AdmissionGrant {
    public:
        AdmissionGrant(AdmissionGrant&&) noexcept            = default;
        AdmissionGrant& operator=(AdmissionGrant&&) noexcept = default;
        AdmissionGrant(const AdmissionGrant&)                = delete;
        AdmissionGrant& operator=(const AdmissionGrant&)     = delete;

        [[nodiscard]] std::uint64_t request_id() const noexcept { return request_id_; }

        [[nodiscard]] BackfillClass backfill_class() const noexcept { return backfill_class_; }

        [[nodiscard]] std::uint64_t protection_epoch() const noexcept { return protection_epoch_; }

        [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept {
            return resource_revision_;
        }

        [[nodiscard]] std::uint64_t service_work_quanta() const noexcept {
            return service_work_quanta_;
        }

    private:
        AdmissionGrant(std::uint64_t request_id, BackfillClass backfill_class,
                       std::uint64_t protection_epoch, ProgramResourceRevision resource_revision,
                       std::uint64_t service_work_quanta) noexcept
            : request_id_(request_id), backfill_class_(backfill_class),
              protection_epoch_(protection_epoch), resource_revision_(resource_revision),
              service_work_quanta_(service_work_quanta) {}

        std::uint64_t request_id_       = 0;
        BackfillClass backfill_class_   = BackfillClass::None;
        std::uint64_t protection_epoch_ = 0;
        ProgramResourceRevision resource_revision_;
        std::uint64_t service_work_quanta_ = 0;

        friend class Scheduler;
    };

    class FifoSnapshot {
    public:
        [[nodiscard]] bool empty() const noexcept { return requests_.empty(); }

        [[nodiscard]] const RequestPtr& head() const {
            if (requests_.empty()) { throw std::logic_error("FIFO snapshot has no head"); }
            return requests_.front();
        }

        [[nodiscard]] std::span<const RequestPtr> backfill_candidates() const noexcept {
            if (requests_.size() <= 1) { return {}; }
            return {requests_.data() + 1, requests_.size() - 1};
        }

    private:
        explicit FifoSnapshot(const std::deque<RequestPtr>& pending)
            : requests_(pending.begin(), pending.end()) {}

        std::vector<RequestPtr> requests_;

        friend class Scheduler;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ControlMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::vector<TokenId> tokens;
        std::uint32_t row_stride = 0;
        std::size_t size         = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    [[nodiscard]] static FifoSnapshot fifo_snapshot(const std::deque<RequestPtr>& pending) {
        return FifoSnapshot(pending);
    }

    template <class Slots>
    [[nodiscard]] RoundMembership build_round_membership(const Slots& slots,
                                                         std::uint32_t max_concurrency) const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_decode_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            if (!request->sequence) {
                throw std::logic_error("decode-ready request has no sequence handle");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.budgets[membership.size]   = RoundBudget{
                  .generated_tokens_remaining =
                    request->output.model_token_budget_remaining(request->budget->remaining()),
            };
            if (membership.budgets[membership.size].generated_tokens_remaining == 0) {
                throw std::logic_error("decode-ready request has no licensed model tokens");
            }
            ++membership.size;
        }
        return membership;
    }

    template <class Slots>
    [[nodiscard]] ControlMembership build_control_membership(const Slots& slots,
                                                             std::uint32_t max_concurrency) const {
        ControlMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_control_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget || !request->sequence) {
                throw std::logic_error("control-ready request has no generation state");
            }
            const std::span<const TokenId> control = request->output.pending_control_tokens();
            if (control.empty() || control.size() > request->budget->remaining()) {
                throw std::logic_error("control-ready request has no admissible control span");
            }
            if (membership.row_stride == 0) {
                membership.row_stride = static_cast<std::uint32_t>(control.size());
                membership.tokens.reserve(static_cast<std::size_t>(membership.row_stride) *
                                          max_concurrency);
            } else if (control.size() != membership.row_stride) {
                throw std::logic_error("compact control membership has ragged target spans");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.tokens.insert(membership.tokens.end(), control.begin(), control.end());
            ++membership.size;
        }
        return membership;
    }

    template <class Slots>
    [[nodiscard]] ActiveAdmissionSet active_admission_set(const Slots& slots,
                                                          std::uint32_t max_concurrency) const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr) { continue; }
            if (request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id     = request->id,
                .backfill_epoch = request->backfill_epoch,
                .backfill_class = request->backfill_class,
            };
        }
        return active;
    }

    static void consume_service_work(Request& request, std::uint64_t work) {
        if (work == 0 || work > request.remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request.remaining_service_work) + " remaining");
        }
        request.remaining_service_work -= work;
    }

    // A lane that owns staged prefill and can run a unit now. `remaining_tokens` is the part of
    // the admitted prompt suffix not yet computed.
    struct PrefillCandidate {
        std::uint32_t lane             = 0;
        std::uint32_t remaining_tokens = 0;
    };

    // At most `max_owners` requests hold staged prefill at once. A lane that has been passed over
    // `max_skip` times is served before any shorter one, so a long prompt cannot starve.
    void configure_prefill(std::uint32_t max_owners, std::uint32_t max_skip) {
        if (max_owners == 0 || max_owners > kMaximumConcurrency || max_skip == 0) {
            throw std::invalid_argument("prefill owner limit and skip bound must be positive");
        }
        if (prefill_owner_count_ != 0) {
            throw std::logic_error("prefill ownership cannot be reconfigured while owned");
        }
        max_prefill_owners_ = max_owners;
        prefill_max_skip_   = max_skip;
    }

    [[nodiscard]] bool should_attempt_admission(bool have_pending, bool admission_check_pending,
                                                bool have_decode, bool previous_unit_was_decode,
                                                bool context_transaction) const noexcept {
        // A newly published lane runs its first unit before the next admission: that unit
        // settles the reused-state fork, which a later materialization or capture must not see.
        return have_pending && admission_check_pending && !context_transaction &&
               prefill_owner_count_ < max_prefill_owners_ && !has_fresh_prefill_lane() &&
               (!have_decode || previous_unit_was_decode);
    }

    [[nodiscard]] ExecutionAction choose_execution(bool have_decode, bool prefill_runnable,
                                                   bool previous_unit_was_decode) const noexcept {
        if (prefill_runnable) {
            return have_decode && !previous_unit_was_decode ? ExecutionAction::Decode
                                                            : ExecutionAction::Prefill;
        }
        return have_decode ? ExecutionAction::Decode : ExecutionAction::Wait;
    }

    [[nodiscard]] bool is_prefill_owner(std::uint32_t lane) const noexcept {
        return lane < kMaximumConcurrency && prefill_owner_[lane];
    }

    [[nodiscard]] std::uint32_t prefill_owner_count() const noexcept {
        return prefill_owner_count_;
    }

    [[nodiscard]] bool is_fresh_prefill_lane(std::uint32_t lane) const noexcept {
        return is_prefill_owner(lane) && prefill_fresh_[lane];
    }

    [[nodiscard]] bool has_fresh_prefill_lane() const noexcept {
        for (std::uint32_t lane = 0; lane < kMaximumConcurrency; ++lane) {
            if (prefill_owner_[lane] && prefill_fresh_[lane]) { return true; }
        }
        return false;
    }

    // Lane for the next prefill unit among `candidates`: a lane that has not run its first unit,
    // else one passed over `max_skip` times (most skipped first), else the shortest remaining
    // suffix. Ties go to the earlier-published lane.
    [[nodiscard]] std::optional<std::uint32_t>
    select_prefill_lane(std::span<const PrefillCandidate> candidates) const {
        const PrefillCandidate* best = nullptr;
        int best_class               = 0;
        for (const PrefillCandidate& candidate : candidates) {
            if (!is_prefill_owner(candidate.lane)) {
                throw std::logic_error("prefill candidate does not own staged prefill");
            }
            const std::uint32_t lane = candidate.lane;
            const int klass          = prefill_fresh_[lane]                          ? 3
                                       : prefill_skips_[lane] >= prefill_max_skip_ ? 2
                                                                                   : 1;
            if (best == nullptr || klass > best_class) {
                best       = &candidate;
                best_class = klass;
                continue;
            }
            if (klass < best_class) { continue; }
            const std::uint32_t best_lane = best->lane;
            bool better                   = false;
            if (klass == 2) {
                better = prefill_skips_[lane] > prefill_skips_[best_lane] ||
                         (prefill_skips_[lane] == prefill_skips_[best_lane] &&
                          prefill_order_[lane] < prefill_order_[best_lane]);
            } else if (klass == 1) {
                better = candidate.remaining_tokens < best->remaining_tokens ||
                         (candidate.remaining_tokens == best->remaining_tokens &&
                          prefill_order_[lane] < prefill_order_[best_lane]);
            } else {
                better = prefill_order_[lane] < prefill_order_[best_lane];
            }
            if (better) { best = &candidate; }
        }
        if (best == nullptr) { return std::nullopt; }
        return best->lane;
    }

    // Accounts one executed prefill unit on `lane` against the lanes that could have run.
    void record_prefill_unit(std::uint32_t lane,
                             std::span<const PrefillCandidate> candidates) noexcept {
        for (const PrefillCandidate& candidate : candidates) {
            if (candidate.lane != lane &&
                prefill_skips_[candidate.lane] < std::numeric_limits<std::uint32_t>::max()) {
                ++prefill_skips_[candidate.lane];
            }
        }
        if (lane < kMaximumConcurrency) { prefill_skips_[lane] = 0; }
    }

    // A published lane stays fresh until a unit has written its state (or finished the prompt):
    // only then is its reused-state fork settled. A unit that returns without progress, such as a
    // zero-progress capture offer, leaves the lane fresh so it runs again before any admission.
    void mark_prefill_settled(std::uint32_t lane) noexcept {
        if (lane < kMaximumConcurrency) { prefill_fresh_[lane] = false; }
    }

    [[nodiscard]] std::optional<std::uint64_t> protection_epoch() const noexcept {
        return protection_ ? std::optional<std::uint64_t>(protection_->epoch_id) : std::nullopt;
    }

    void set_prefill_lane(std::uint32_t lane) {
        if (lane >= kMaximumConcurrency) { throw std::logic_error("prefill lane is out of range"); }
        if (prefill_owner_[lane]) {
            throw std::logic_error("request already owns staged prefill");
        }
        if (prefill_owner_count_ >= max_prefill_owners_) {
            throw std::logic_error("too many requests own staged prefill");
        }
        prefill_owner_[lane] = true;
        prefill_fresh_[lane] = true;
        prefill_skips_[lane] = 0;
        prefill_order_[lane] = next_prefill_order_++;
        ++prefill_owner_count_;
    }

    void clear_prefill_lane(std::uint32_t lane) {
        if (!is_prefill_owner(lane)) {
            throw std::logic_error("request does not own staged prefill");
        }
        prefill_owner_[lane] = false;
        prefill_fresh_[lane] = false;
        prefill_skips_[lane] = 0;
        --prefill_owner_count_;
    }

    void observe_fifo_head(std::optional<std::uint64_t> request_id) noexcept {
        if (fifo_head_id_ == request_id) { return; }
        fifo_head_id_ = request_id;
        protection_.reset();
    }

    void on_waiting_removed(std::uint64_t request_id) noexcept {
        if (fifo_head_id_ && *fifo_head_id_ == request_id) {
            fifo_head_id_.reset();
            protection_.reset();
        }
    }

    [[nodiscard]] AdmissionGrant grant_head(std::uint64_t request_id,
                                            std::uint64_t service_work_quanta) const {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id || request_id == 0 ||
            service_work_quanta == 0) {
            throw std::logic_error("head admission is not bound to the observed FIFO head");
        }
        return AdmissionGrant(request_id, BackfillClass::None, 0, {}, service_work_quanta);
    }

    [[nodiscard]] bool protect_blocked_head(std::uint64_t request_id,
                                            std::span<const ActiveAdmissionSnapshot> active,
                                            ProgramResourceRevision resource_revision) {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id) {
            throw std::logic_error("blocked admission does not match the observed FIFO head");
        }
        if (!protection_) {
            protection_.emplace(make_admission_protection(next_protection_epoch_++, request_id,
                                                          resource_revision, active));
        } else if (protection_->head_request_id != request_id) {
            throw std::logic_error("protected head changed without a FIFO transition");
        } else {
            rebind_admission_protection(*protection_, active, resource_revision);
        }
        return protection_has_live_donor(*protection_, active);
    }

    [[nodiscard]] std::optional<AdmissionGrant>
    qualify_backfill(std::uint64_t request_id, std::uint64_t service_work_quanta,
                     std::span<const ActiveAdmissionSnapshot> active,
                     ProgramResourceRevision program_proof_revision) const {
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_) {
            throw std::logic_error("backfill qualification has no open protected head");
        }
        if (request_id == 0 || request_id == *fifo_head_id_ || service_work_quanta == 0) {
            throw std::logic_error("backfill candidate has invalid scheduling identity");
        }
        if (persistent_backfill_is_authorized(*protection_, request_id, active,
                                              program_proof_revision)) {
            return AdmissionGrant(request_id, BackfillClass::Persistent, protection_->epoch_id,
                                  program_proof_revision, service_work_quanta);
        }
        return std::nullopt;
    }

    [[nodiscard]] bool validate_grant(const AdmissionGrant& grant) const noexcept {
        if (grant.request_id_ == 0 || grant.service_work_quanta_ == 0) { return false; }
        if (grant.backfill_class_ == BackfillClass::None) {
            return grant.protection_epoch_ == 0 && grant.resource_revision_.value == 0 &&
                   fifo_head_id_ && *fifo_head_id_ == grant.request_id_;
        }
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_ ||
            protection_->epoch_id != grant.protection_epoch_ ||
            protection_->resource_revision != grant.resource_revision_ ||
            grant.request_id_ == *fifo_head_id_) {
            return false;
        }
        return grant.backfill_class_ == BackfillClass::Persistent;
    }

    void commit_admission(AdmissionGrant&& grant) {
        if (!validate_grant(grant)) {
            throw std::logic_error("admission grant is stale or inconsistent");
        }
        if (grant.backfill_class_ == BackfillClass::None) {
            fifo_head_id_.reset();
            protection_.reset();
            grant.request_id_          = 0;
            grant.service_work_quanta_ = 0;
            return;
        }
        grant.request_id_          = 0;
        grant.service_work_quanta_ = 0;
    }

    void reset() noexcept {
        prefill_owner_.fill(false);
        prefill_fresh_.fill(false);
        prefill_skips_.fill(0);
        prefill_owner_count_ = 0;
        fifo_head_id_.reset();
        protection_.reset();
    }

private:
    std::array<bool, kMaximumConcurrency> prefill_owner_{};
    std::array<bool, kMaximumConcurrency> prefill_fresh_{};
    std::array<std::uint32_t, kMaximumConcurrency> prefill_skips_{};
    std::array<std::uint64_t, kMaximumConcurrency> prefill_order_{};
    std::uint32_t prefill_owner_count_ = 0;
    std::uint32_t max_prefill_owners_  = 1;
    std::uint32_t prefill_max_skip_    = 8;
    std::uint64_t next_prefill_order_  = 0;
    std::optional<std::uint64_t> fifo_head_id_;
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
};

} // namespace ninfer::runtime
