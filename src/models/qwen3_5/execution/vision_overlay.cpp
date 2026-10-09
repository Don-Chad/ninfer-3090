#include "models/qwen3_5/execution/vision_overlay.h"

#include "core/kv_loan.h"
#include "models/qwen3_5/execution/vision.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr std::size_t kStagingAlignment = 256;

std::size_t staging_align(std::size_t bytes) {
    return (bytes + kStagingAlignment - 1) / kStagingAlignment * kStagingAlignment;
}

// Moves pointers of one pinned group onto the staging that holds that group. Every non-null
// pointer must address the group's own extent, which also proves the prepared parameters borrow
// the pinned block rather than some other backing.
class GroupRebase {
public:
    GroupRebase(std::span<const std::byte> block, PinnedRange range, std::byte* target)
        : begin_(block.data() + range.offset), bytes_(range.bytes), target_(target) {}

    [[nodiscard]] const void* pointer(const void* source) const {
        if (source == nullptr) { return nullptr; }
        const auto* byte = static_cast<const std::byte*>(source);
        if (byte < begin_ || byte > begin_ + bytes_) {
            throw std::logic_error("Vision overlay weight does not address its pinned group");
        }
        return target_ + (byte - begin_);
    }

    void operator()(Weight& weight) const {
        weight.payload = pointer(weight.payload);
        weight.qdata   = pointer(weight.qdata);
        weight.qhigh   = pointer(weight.qhigh);
        weight.scales  = pointer(weight.scales);
    }

    void operator()(LinearParameters& linear) const { (*this)(linear.weight); }

    void operator()(Tensor& tensor) const {
        tensor.data = const_cast<void*>(pointer(tensor.data));
    }

    void operator()(NormParameters& norm) const {
        (*this)(norm.weight);
        (*this)(norm.bias);
    }

private:
    const std::byte* begin_ = nullptr;
    std::size_t bytes_      = 0;
    std::byte* target_      = nullptr;
};

} // namespace

// ---------------------------------------------------------------------------------------------
// VisionWindow and VisionResidencyBroker

VisionWindow::~VisionWindow() { close(); }

VisionWindow::VisionWindow(VisionWindow&& other) noexcept
    : broker_(other.broker_), tier_(other.tier_), weights_(std::move(other.weights_)),
      kv_(std::move(other.kv_)), runs_(std::move(other.runs_)) {
    other.broker_ = nullptr;
}

VisionWindow& VisionWindow::operator=(VisionWindow&& other) noexcept {
    if (this != &other) {
        close();
        broker_       = other.broker_;
        tier_         = other.tier_;
        weights_      = std::move(other.weights_);
        kv_           = std::move(other.kv_);
        runs_         = std::move(other.runs_);
        other.broker_ = nullptr;
    }
    return *this;
}

void VisionWindow::close() noexcept {
    if (broker_ == nullptr) { return; }
    VisionResidencyBroker* const broker = broker_;
    broker_                             = nullptr;
    if (tier_ == Tier::KvGranules) {
        kv_.close();
        broker->return_loan(runs_);
    } else {
        weights_.close();
    }
}

void VisionResidencyBroker::enable_kv_tier(EvictableKVPool& arena, DeviceKVPagePool& pages,
                                           std::function<bool()> can_submit) {
    kv_arena_   = &arena;
    kv_pages_   = &pages;
    can_submit_ = std::move(can_submit);
}

std::uint32_t VisionResidencyBroker::kv_loan_pages(std::size_t bytes) const {
    if (kv_arena_ == nullptr || kv_pages_ == nullptr || kv_arena_->poisoned()) { return 0; }
    const KVLoanPlan plan = plan_kv_loan(*kv_arena_, *kv_pages_, bytes, false);
    std::uint32_t pages   = 0;
    for (const KVPageRun& run : plan.runs) { pages += run.count; }
    return pages;
}

std::uint32_t VisionResidencyBroker::kv_loan_minimum_pages(std::size_t bytes) const {
    // A window wider than the lendable KV can never be funded there, however much is reclaimed.
    if (kv_arena_ == nullptr || kv_pages_ == nullptr || bytes == 0 ||
        bytes > kv_arena_->window_capacity_bytes()) {
        return 0;
    }
    const std::size_t granularity = kv_arena_->granularity();
    std::size_t page_bytes        = 0;
    for (std::size_t index = 0; index < kv_pages_->plane_count(); ++index) {
        const auto stride = static_cast<std::size_t>(kv_pages_->plane(index).nb[3]);
        if (stride == 0 || granularity % stride != 0 || granularity / stride > kMaxLoanUnitPages) {
            continue;
        }
        page_bytes += stride;
    }
    if (page_bytes == 0) { return 0; }
    const std::size_t granules = (bytes + granularity - 1) / granularity;
    return static_cast<std::uint32_t>((granules * granularity + page_bytes - 1) / page_bytes);
}

std::uint32_t VisionResidencyBroker::available_kv_pages() const noexcept {
    return kv_pages_ != nullptr ? kv_pages_->available_pages() : 0;
}

bool VisionResidencyBroker::can_submit() const { return !can_submit_ || can_submit_(); }

bool VisionResidencyBroker::drain() {
    if (submitted_ == nullptr) { return false; }
    submitted_->finish();
    return true;
}

std::optional<VisionWindow> VisionResidencyBroker::try_acquire_kv(std::size_t bytes) {
    require_closed();
    if (kv_arena_ == nullptr || kv_pages_ == nullptr || kv_arena_->poisoned()) {
        return std::nullopt;
    }
    KVLoanPlan plan = plan_kv_loan(*kv_arena_, *kv_pages_, bytes);
    if (plan.granules.empty()) { return std::nullopt; }
    std::uint32_t total_loan_pages = 0;
    for (const KVPageRun& run : plan.runs) { total_loan_pages += run.count; }
    if (total_loan_pages > kv_pages_->available_pages()) { return std::nullopt; }
    // Lending mutates the page pool before any VisionWindow owns the loan, so a throw part-way
    // through this loop would strand the runs already lent: capacity the pool never gets back,
    // because no destructor knows about them. Unwind what this loop did before the failure leaves.
    std::size_t lent = 0;
    try {
        for (; lent < plan.runs.size(); ++lent) {
            kv_pages_->lend_pages(plan.runs[lent].begin, plan.runs[lent].count);
        }
    } catch (...) {
        while (lent-- > 0) {
            try {
                kv_pages_->return_pages(plan.runs[lent].begin, plan.runs[lent].count);
            } catch (...) { std::terminate(); }
        }
        throw;
    }
    VisionWindow window;
    window.broker_ = this;
    window.tier_   = VisionWindow::Tier::KvGranules;
    window.runs_   = std::move(plan.runs);
    // From here the window owns the loan: if the lease throws, unwinding destroys `window`, whose
    // close() returns every run.
    window.kv_ = kv_arena_->lease(plan.granules, device_.stream);
    return window;
}

VisionWindow VisionResidencyBroker::acquire(std::size_t bytes) {
    std::optional<VisionWindow> borrowed = try_acquire_kv(bytes);
    if (borrowed) { return std::move(*borrowed); }
    VisionWindow window;
    window.broker_  = this;
    window.tier_    = VisionWindow::Tier::WeightTail;
    window.weights_ = pool_.evict(bytes, device_.stream);
    return window;
}

bool VisionResidencyBroker::poisoned() const noexcept {
    return pool_.poisoned() || (kv_arena_ != nullptr && kv_arena_->poisoned());
}

bool VisionResidencyBroker::window_open() const noexcept {
    return pool_.transaction_open() || (kv_arena_ != nullptr && kv_arena_->lease_open());
}

void VisionResidencyBroker::require_closed() {
    // A submitted window belongs to an encode that is already running; finishing it costs at most
    // its remaining encode time and keeps its embeddings for the lane that submitted it.
    (void)drain();
    if (window_open()) { throw std::logic_error("a Vision overlay window is already open"); }
}

void VisionResidencyBroker::return_loan(std::vector<KVPageRun>& runs) noexcept {
    if (kv_pages_ == nullptr) { return; }
    for (const KVPageRun& run : runs) {
        try {
            kv_pages_->return_pages(run.begin, run.count);
        } catch (...) { std::terminate(); }
    }
    runs.clear();
}

// ---------------------------------------------------------------------------------------------
// PinnedResultPool

PinnedResultPool::Handle::Handle(Handle&& other) noexcept
    : pool_(other.pool_), index_(other.index_), bytes_(other.bytes_) {
    other.pool_ = nullptr;
}

PinnedResultPool::Handle& PinnedResultPool::Handle::operator=(Handle&& other) noexcept {
    if (this != &other) {
        release();
        pool_       = other.pool_;
        index_      = other.index_;
        bytes_      = other.bytes_;
        other.pool_ = nullptr;
    }
    return *this;
}

void PinnedResultPool::Handle::release() noexcept {
    if (pool_ != nullptr) {
        pool_->free_.push_back(index_);
        pool_ = nullptr;
    }
}

PinnedResultPool::PinnedResultPool(std::size_t initial, std::size_t maximum,
                                   std::size_t slot_bytes)
    : maximum_(maximum), slot_bytes_(slot_bytes) {
    if (initial == 0 || maximum < initial || slot_bytes == 0) {
        throw std::invalid_argument("pinned Vision result pool needs at least one slot");
    }
    buffers_.reserve(maximum);
    free_.reserve(maximum);
    for (std::size_t index = 0; index < initial; ++index) {
        buffers_.push_back(std::make_unique<PinnedHostBuffer>(slot_bytes));
        free_.push_back(index);
    }
}

PinnedResultPool::Handle PinnedResultPool::acquire() {
    if (free_.empty() && buffers_.size() < maximum_) {
        buffers_.push_back(std::make_unique<PinnedHostBuffer>(slot_bytes_));
        free_.push_back(buffers_.size() - 1);
    }
    if (free_.empty()) { throw std::logic_error("pinned Vision result pool has no free slot"); }
    const std::size_t index = free_.back();
    free_.pop_back();
    const PinnedHostBuffer& buffer = *buffers_[index];
    return Handle(*this, index,
                  std::span<std::byte>(static_cast<std::byte*>(buffer.data()), buffer.size()));
}

// ---------------------------------------------------------------------------------------------
// Window planning

VisionWorkspacePlan plan_vision_window_workspace(const Parameters& parameters,
                                                 std::uint32_t max_merged_tokens) {
    const VisionConfig& config = parameters.model.config().vision.value();
    const VisionParameters& vision = parameters.vision.value();
    // A general region sized by the encode itself places the handoff right after the encode
    // scratch instead of after the Text workspace.
    const VisionWorkspacePlan compact =
        VisionContext::plan_workspace(config, vision, max_merged_tokens, 1);
    const std::size_t encode_bytes =
        VisionContext::workspace_bytes(config, vision,
                                       static_cast<std::size_t>(max_merged_tokens) *
                                           config.spatial_merge_size * config.spatial_merge_size,
                                       max_merged_tokens, compact);
    return VisionContext::plan_workspace(config, vision, max_merged_tokens, encode_bytes);
}

std::size_t vision_window_bytes(const VisionOverlayLayout& layout,
                                const VisionWorkspacePlan& window_plan) {
    return staging_align(layout.staging_bytes) + window_plan.capacity_bytes;
}

// ---------------------------------------------------------------------------------------------
// VisionWeightStream

VisionWeightStream::VisionWeightStream(DeviceContext& device, const VisionOverlayLayout& layout,
                                       std::span<const std::byte> pinned_block, std::byte* staging)
    : device_(device), layout_(layout), block_(pinned_block) {
    prelude_ = staging;
    merger_  = prelude_ + staging_align(layout.prelude.bytes);
    slot_[0] = merger_ + staging_align(layout.merger.bytes);
    slot_[1] = slot_[0] + staging_align(layout.slot_bytes);
    // Nothing below may escape without releasing what it already created: construction that throws
    // runs no destructor for this object, so each failed setup would strand its events until the
    // driver runs out. The caller drains the transfer stream before it returns the staging, which
    // covers the copies this constructor may already have enqueued.
    try {
        for (cudaEvent_t& event : uploaded_) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&prelude_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&merger_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_fence_, cudaEventDisableTiming));

        const cudaStream_t copy = device_.transfer_stream;
        CUDA_CHECK(cudaMemcpyAsync(prelude_, block_.data() + layout.prelude.offset,
                                   layout.prelude.bytes, cudaMemcpyHostToDevice, copy));
        CUDA_CHECK(cudaEventRecord(prelude_event_, copy));
        CUDA_CHECK(cudaMemcpyAsync(merger_, block_.data() + layout.merger.offset,
                                   layout.merger.bytes, cudaMemcpyHostToDevice, copy));
        CUDA_CHECK(cudaEventRecord(merger_event_, copy));
    } catch (...) {
        destroy_events();
        throw;
    }
    upload_bytes_ = layout.prelude.bytes + layout.merger.bytes;
}

VisionWeightStream::~VisionWeightStream() {
    // The owning session drains both streams first; the events are idle here.
    destroy_events();
}

void VisionWeightStream::destroy_events() noexcept {
    for (cudaEvent_t& event : uploaded_) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
        event = nullptr;
    }
    for (cudaEvent_t* event : {&prelude_event_, &merger_event_, &compute_fence_}) {
        if (*event != nullptr) { (void)cudaEventDestroy(*event); }
        *event = nullptr;
    }
}

VisionParameters VisionWeightStream::window_parameters(const VisionParameters& host) const {
    if (host.layers.size() != layout_.layers.size()) {
        throw std::logic_error("Vision overlay layout does not match the prepared tower");
    }
    VisionParameters out = host;
    const GroupRebase prelude(block_, layout_.prelude, prelude_);
    prelude(out.patch_embedding);
    prelude(out.patch_embedding_bias);
    prelude(out.position_embedding);
    for (std::size_t index = 0; index < out.layers.size(); ++index) {
        const GroupRebase rebase(block_, layout_.layers[index], slot_[index % 2]);
        VisionBlockParameters& layer = out.layers[index];
        rebase(layer.norm1);
        rebase(layer.norm2);
        rebase(layer.qkv);
        rebase(layer.qkv_bias);
        rebase(layer.output);
        rebase(layer.fc1);
        rebase(layer.fc2);
        rebase(layer.output_bias);
        rebase(layer.fc1_bias);
        rebase(layer.fc2_bias);
    }
    const GroupRebase merger(block_, layout_.merger, merger_);
    merger(out.merger_norm);
    merger(out.merger_fc1);
    merger(out.merger_fc2);
    merger(out.merger_fc1_bias);
    merger(out.merger_fc2_bias);
    return out;
}

void VisionWeightStream::reset(cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(device_.transfer_stream, compute_fence_, 0));
    next_upload_ = 0;
    upload_next_layer();
    upload_next_layer();
}

void VisionWeightStream::prelude_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
}

void VisionWeightStream::merger_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
}

void VisionWeightStream::arrive(std::uint32_t layer, cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
    if (next_upload_ == layer + 1 && next_upload_ < layout_.layers.size()) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(device_.transfer_stream, compute_fence_, 0));
        upload_next_layer();
    }
}

void VisionWeightStream::upload_next_layer() {
    if (next_upload_ >= layout_.layers.size()) { return; }
    const std::uint32_t layer = next_upload_++;
    const PinnedRange range   = layout_.layers[layer];
    CUDA_CHECK(cudaMemcpyAsync(slot_[layer % 2], block_.data() + range.offset, range.bytes,
                               cudaMemcpyHostToDevice, device_.transfer_stream));
    CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], device_.transfer_stream));
    upload_bytes_ += range.bytes;
}

// ---------------------------------------------------------------------------------------------
// VisionOverlaySession

VisionOverlaySession::VisionOverlaySession(DeviceContext& device, VisionResidencyBroker& broker,
                                           const Parameters& parameters,
                                           const VisionWorkspacePlan& window_plan,
                                           PinnedResultPool::Handle result)
    : device_(device), broker_(broker), parameters_(parameters),
      layout_(parameters.model.vision_overlay().value()), window_plan_(window_plan),
      result_(std::move(result)), completion_(device), main_fence_(device) {
    if (vision_window_bytes(layout_, window_plan_) > broker_.window_capacity_bytes()) {
        throw std::logic_error("Vision overlay window plan exceeds the pool window capacity");
    }
    if (result_.bytes().size() < window_plan_.handoff_capacity_bytes) {
        throw std::logic_error("pinned Vision result slot is smaller than the item handoff");
    }
}

VisionOverlaySession::~VisionOverlaySession() { abandon(); }

void VisionOverlaySession::abandon() noexcept {
    if (broker_.submitted_ == this) { broker_.submitted_ = nullptr; }
    ahead_ = false;
    if (!pending_ && !window_.open()) { return; }
    // The encode still owns the borrowed memory and the staging events; wait it out before the
    // weight stream and the window are destroyed. Failures here are not recoverable and must not
    // escape.
    if (encode_stream_ != nullptr) { (void)cudaStreamSynchronize(encode_stream_); }
    (void)cudaStreamSynchronize(device_.transfer_stream);
    pending_ = false;
    weights_.reset();
    window_parameters_.reset();
    window_.close();
}

VisionWorkspacePlan
VisionOverlaySession::item_plan(const qwen3_5::VisionItemControl& control) const {
    // Size the window for this item, not the planned maximum: fewer chunks to remap and fewer
    // bytes to restore. The planned window bounds it.
    if (control.merged_count == 0 || control.merged_count > window_plan_.max_merged_tokens) {
        throw std::invalid_argument("Vision item exceeds the overlay window budget");
    }
    VisionWorkspacePlan plan =
        plan_vision_window_workspace(parameters_, static_cast<std::uint32_t>(control.merged_count));
    if (plan.capacity_bytes > window_plan_.capacity_bytes) {
        throw std::invalid_argument("Vision item exceeds the overlay window budget");
    }
    return plan;
}

std::size_t VisionOverlaySession::window_bytes(const qwen3_5::VisionItemControl& control) const {
    return vision_window_bytes(layout_, item_plan(control));
}

void VisionOverlaySession::begin(VisionWindow&& window, std::span<const std::uint16_t> patches,
                                 const qwen3_5::VisionItemControl& control,
                                 const VisionWorkspacePlan& item_plan) {
    const std::size_t staging = staging_align(layout_.staging_bytes);
    window_                   = std::move(window);
    auto* const base          = static_cast<std::byte*>(window_.memory().data);
    const DeviceSpan backing{base + staging, window_.memory().bytes - staging};
    // A KV-funded window leaves the Text weights mapped, so its encode runs on the Vision stream
    // beside other lanes' work. It starts after everything already submitted on the main stream:
    // the borrowed granules held free pages, but a page released by work still in flight must not
    // be overwritten before that work retires.
    if (window_.tier() == VisionWindow::Tier::KvGranules) {
        encode_stream_ = device_.vision_stream;
        main_fence_.record(device_.stream);
        main_fence_.wait(encode_stream_);
    } else {
        encode_stream_ = device_.stream;
    }
    weights_.emplace(device_, layout_, parameters_.model.pinned_weights(), base);
    window_parameters_.emplace(weights_->window_parameters(parameters_.vision.value()));
    weights_->reset(encode_stream_);
    const VisionContext context(device_, parameters_.model.config().vision.value(),
                                *window_parameters_, encode_stream_);
    Tensor output = VisionContext::bind_output(backing, item_plan, control.merged_count);
    context.encode(VisionItemView{patches, &control}, output, backing, item_plan, &*weights_);
    result_bytes_ = output.bytes();
    if (result_bytes_ > result_.bytes().size()) {
        throw std::logic_error("Vision item embeddings exceed the pinned result slot");
    }
    CUDA_CHECK(cudaMemcpyAsync(result_.bytes().data(), output.data, result_bytes_,
                               cudaMemcpyDeviceToHost, encode_stream_));
    completion_.record(encode_stream_);
    pending_ = true;
}

bool VisionOverlaySession::submit_item(std::span<const std::uint16_t> patches,
                                       const qwen3_5::VisionItemControl& control) {
    if (pending_) { throw std::logic_error("a Vision item is already in flight"); }
    // Early submission is opportunistic: another window keeps this item for its own unit.
    if (broker_.window_open() || !broker_.can_submit()) { return false; }
    const VisionWorkspacePlan plan = item_plan(control);
    // Only a KV-funded window may stay open across unit boundaries: a weight-tail window unmaps
    // Text weights, which would stop every other lane.
    std::optional<VisionWindow> borrowed =
        broker_.try_acquire_kv(staging_align(layout_.staging_bytes) + plan.capacity_bytes);
    if (!borrowed) { return false; }
    window_start_ = Clock::now();
    try {
        begin(std::move(*borrowed), patches, control, plan);
    } catch (...) {
        abandon();
        throw;
    }
    broker_.submitted_ = this;
    ahead_             = true;
    return true;
}

void VisionOverlaySession::finish() {
    if (!pending_) { return; }
    if (broker_.submitted_ == this) { broker_.submitted_ = nullptr; }
    // Both producers into the borrowed extent are drained before it is handed back: the encode on
    // `encode_stream_`, whose completion event is recorded after the result copy, and the layer
    // uploads on the transfer stream. Nothing below enqueues again -- `weights_.reset()` is
    // std::optional's, so it destroys idle events rather than re-entering the member reset() that
    // issues the first two layer copies.
    try {
        completion_.synchronize();
        CUDA_CHECK(cudaStreamSynchronize(device_.transfer_stream));
    } catch (...) {
        abandon();
        throw;
    }
    stats_.staged_bytes += weights_->uploaded_bytes();
    weights_.reset();
    window_parameters_.reset();
    const bool exclusive       = window_.tier() == VisionWindow::Tier::WeightTail;
    const double open_seconds  = window_.open_seconds();
    const std::size_t borrowed = window_.borrowed_bytes();
    window_.close();
    pending_ = false;
    if (broker_.poisoned()) {
        throw std::runtime_error(
            "Vision overlay window failed to return its memory; the model state is no longer "
            "trustworthy");
    }
    stats_.window_seconds += std::chrono::duration<double>(Clock::now() - window_start_).count();
    stats_.evict_seconds += open_seconds;
    stats_.restore_seconds += window_.close_seconds();
    stats_.evicted_bytes += borrowed;
    stats_.windows += 1;
    stats_.exclusive_windows += exclusive ? 1U : 0U;
    stats_.ahead_windows += ahead_ ? 1U : 0U;
    ahead_ = false;
}

std::span<const std::byte> VisionOverlaySession::take_submitted() {
    finish();
    return {result_.bytes().data(), result_bytes_};
}

std::span<const std::byte>
VisionOverlaySession::encode_item(std::span<const std::uint16_t> patches,
                                  const qwen3_5::VisionItemControl& control) {
    if (pending_) { throw std::logic_error("a Vision item is already in flight"); }
    const VisionWorkspacePlan plan = item_plan(control);
    window_start_                  = Clock::now();
    VisionWindow window = broker_.acquire(staging_align(layout_.staging_bytes) + plan.capacity_bytes);
    try {
        begin(std::move(window), patches, control, plan);
    } catch (...) {
        abandon();
        throw;
    }
    finish();
    return {result_.bytes().data(), result_bytes_};
}

} // namespace ninfer::models::qwen3_5::execution
