#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_kv_pool.h"
#include "core/evictable_weight_pool.h"
#include "core/kv_loan.h"
#include "core/paged_kv_cache.h"
#include <ninfer/targets/qwen3_6/vision.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

struct VisionOverlayWindowStats {
    double window_seconds     = 0.0;
    double evict_seconds      = 0.0;
    double restore_seconds    = 0.0;
    std::size_t evicted_bytes = 0;
    std::size_t staged_bytes  = 0;
    std::uint32_t windows     = 0;
    // Windows that had to borrow the weight tail, which stalls every other lane.
    std::uint32_t exclusive_windows = 0;
};

class VisionResidencyBroker;

// One open window, served either from free KV granules (tier 1: the text weights stay mapped, so
// other lanes keep decoding) or from the evict-ranked weight tail (tier 2: exclusive, always
// available because the tail is far larger than the largest window).
class VisionWindow {
public:
    enum class Tier : std::uint8_t { KvGranules, WeightTail };

    VisionWindow() noexcept = default;
    ~VisionWindow();

    VisionWindow(const VisionWindow&)            = delete;
    VisionWindow& operator=(const VisionWindow&) = delete;
    VisionWindow(VisionWindow&& other) noexcept;
    VisionWindow& operator=(VisionWindow&& other) noexcept;

    [[nodiscard]] bool open() const noexcept { return broker_ != nullptr; }
    [[nodiscard]] Tier tier() const noexcept { return tier_; }
    [[nodiscard]] DeviceSpan memory() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.leased() : weights_.leased();
    }
    [[nodiscard]] double open_seconds() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.stats().lease_seconds : weights_.stats().evict_seconds;
    }
    [[nodiscard]] double close_seconds() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.stats().return_seconds
                                         : weights_.stats().restore_seconds;
    }
    [[nodiscard]] std::size_t borrowed_bytes() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.stats().mapped_bytes : weights_.stats().mapped_bytes;
    }

    // Returns the borrowed memory. Never throws: a failure poisons the owning pool.
    void close() noexcept;

private:
    friend class VisionResidencyBroker;

    VisionResidencyBroker* broker_ = nullptr;
    Tier tier_                     = Tier::WeightTail;
    EvictableWeightPool::Transaction weights_;
    EvictableKVPool::Transaction kv_;
    std::vector<KVPageRun> runs_;
};

// Program-owned arbiter of the single overlay window a device can hold. Tier 1 is offered only
// while the KV tier is enabled and the free pages cover the request; otherwise the window falls
// back to the weight tail, which makes it exclusive by construction: the text weights it borrows
// are unmapped until the window closes.
class VisionResidencyBroker {
public:
    VisionResidencyBroker(DeviceContext& device, EvictableWeightPool& pool) noexcept
        : device_(device), pool_(pool) {}

    // Offers free KV granules as the preferred currency. `can_lend` refuses a loan while the
    // Program cannot afford a capacity change; `on_change` bumps the resource revision.
    void enable_kv_tier(EvictableKVPool& arena, DeviceKVPagePool& pages,
                        std::function<bool()> can_lend, std::function<void()> on_change) {
        kv_arena_   = &arena;
        kv_pages_   = &pages;
        can_lend_   = std::move(can_lend);
        on_change_  = std::move(on_change);
    }

    // Free KV granules only. Empty when the pool cannot fund the window, which is the signal
    // that an asynchronous encode is not available for this item.
    [[nodiscard]] std::optional<VisionWindow> try_acquire_kv(std::size_t bytes) {
        require_closed();
        if (kv_arena_ == nullptr || kv_pages_ == nullptr || kv_arena_->poisoned() ||
            (can_lend_ && !can_lend_())) {
            return std::nullopt;
        }
        KVLoanPlan plan = plan_kv_loan(*kv_arena_, *kv_pages_, bytes);
        if (plan.granules.empty()) { return std::nullopt; }
        for (const KVPageRun& run : plan.runs) { kv_pages_->lend_pages(run.begin, run.count); }
        if (on_change_) { on_change_(); }
        VisionWindow window;
        window.broker_ = this;
        window.tier_   = VisionWindow::Tier::KvGranules;
        window.runs_   = std::move(plan.runs);
        window.kv_     = kv_arena_->lease(plan.granules, device_.stream);
        ++tier1_windows_;
        return window;
    }

    // Free KV granules when they cover the window, the evict-ranked weight tail otherwise.
    [[nodiscard]] VisionWindow acquire(std::size_t bytes) {
        std::optional<VisionWindow> borrowed = try_acquire_kv(bytes);
        if (borrowed) { return std::move(*borrowed); }
        VisionWindow window;
        window.broker_  = this;
        window.tier_    = VisionWindow::Tier::WeightTail;
        window.weights_ = pool_.evict(bytes, device_.stream);
        ++tier2_windows_;
        return window;
    }

    [[nodiscard]] bool poisoned() const noexcept {
        return pool_.poisoned() || (kv_arena_ != nullptr && kv_arena_->poisoned());
    }
    [[nodiscard]] std::size_t window_capacity_bytes() const noexcept {
        return pool_.window_capacity_bytes();
    }
    [[nodiscard]] std::uint32_t tier1_windows() const noexcept { return tier1_windows_; }
    [[nodiscard]] std::uint32_t tier2_windows() const noexcept { return tier2_windows_; }

private:
    friend class VisionWindow;

    void require_closed() const {
        if (pool_.transaction_open() || (kv_arena_ != nullptr && kv_arena_->lease_open())) {
            throw std::logic_error("a vision overlay window is already open");
        }
    }

    void return_loan(std::vector<KVPageRun>& runs) noexcept {
        if (kv_pages_ == nullptr) { return; }
        for (const KVPageRun& run : runs) {
            try {
                kv_pages_->return_pages(run.begin, run.count);
            } catch (...) { std::terminate(); }
        }
        runs.clear();
        if (on_change_) { on_change_(); }
    }

    DeviceContext& device_;
    EvictableWeightPool& pool_;
    EvictableKVPool* kv_arena_      = nullptr;
    DeviceKVPagePool* kv_pages_     = nullptr;
    std::function<bool()> can_lend_ = nullptr;
    std::function<void()> on_change_ = nullptr;
    std::uint32_t tier1_windows_    = 0;
    std::uint32_t tier2_windows_    = 0;
};

inline VisionWindow::~VisionWindow() { close(); }

inline VisionWindow::VisionWindow(VisionWindow&& other) noexcept
    : broker_(other.broker_), tier_(other.tier_), weights_(std::move(other.weights_)),
      kv_(std::move(other.kv_)), runs_(std::move(other.runs_)) {
    other.broker_ = nullptr;
}

inline VisionWindow& VisionWindow::operator=(VisionWindow&& other) noexcept {
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

inline void VisionWindow::close() noexcept {
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

// Pinned slots that receive the embeddings of one item each; a session holds one slot for its
// lifetime, so at most max_concurrency slots are ever needed and no window allocates host memory.
class PinnedResultPool {
public:
    class Handle {
    public:
        Handle() noexcept = default;
        ~Handle() { release(); }
        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& other) noexcept
            : pool_(other.pool_), index_(other.index_), bytes_(other.bytes_) {
            other.pool_ = nullptr;
        }
        Handle& operator=(Handle&& other) noexcept {
            if (this != &other) {
                release();
                pool_       = other.pool_;
                index_      = other.index_;
                bytes_      = other.bytes_;
                other.pool_ = nullptr;
            }
            return *this;
        }

        [[nodiscard]] std::span<std::byte> bytes() const noexcept { return bytes_; }

    private:
        friend class PinnedResultPool;
        Handle(PinnedResultPool& pool, std::size_t index, std::span<std::byte> bytes) noexcept
            : pool_(&pool), index_(index), bytes_(bytes) {}
        void release() noexcept {
            if (pool_ != nullptr) {
                pool_->free_.push_back(index_);
                pool_ = nullptr;
            }
        }

        PinnedResultPool* pool_ = nullptr;
        std::size_t index_      = 0;
        std::span<std::byte> bytes_;
    };

    PinnedResultPool(std::size_t slots, std::size_t slot_bytes) {
        if (slots == 0 || slot_bytes == 0) {
            throw std::invalid_argument("pinned vision result pool needs at least one slot");
        }
        buffers_.reserve(slots);
        free_.reserve(slots);
        for (std::size_t index = 0; index < slots; ++index) {
            buffers_.push_back(std::make_unique<PinnedHostBuffer>(slot_bytes));
            free_.push_back(index);
        }
    }

    [[nodiscard]] Handle acquire() {
        if (free_.empty()) {
            throw std::logic_error("pinned vision result pool has no free slot");
        }
        const std::size_t index = free_.back();
        free_.pop_back();
        PinnedHostBuffer& buffer = *buffers_[index];
        return Handle(*this, index,
                      std::span<std::byte>(static_cast<std::byte*>(buffer.data()), buffer.size()));
    }

    [[nodiscard]] std::size_t slot_bytes() const noexcept {
        return buffers_.empty() ? 0 : buffers_.front()->size();
    }

private:
    std::vector<std::unique_ptr<PinnedHostBuffer>> buffers_;
    std::vector<std::size_t> free_;
};

class VisionWeightStream;
class VisionOverlaySession;

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
