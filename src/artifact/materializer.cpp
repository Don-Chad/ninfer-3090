#include "artifact/materializer.h"

#include "artifact/framing.h"
#include "artifact/transcode.h"
#include "artifact/reader.h"
#include "core/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024 * 1024;
constexpr std::size_t kMaximumSlotCount = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                            cudaGetErrorString(status));
    }
}

class Slot {
public:
    // cudaMallocHost only guarantees sub-page alignment once other pinned allocations exist in
    // the process (observed on Windows); direct I/O requires page alignment. Over-allocate by one
    // page and align the staging pointer inside the pinned block.
    explicit Slot(std::size_t bytes) : buffer(bytes + kPayloadAlignment) {
        const auto raw     = reinterpret_cast<std::uintptr_t>(buffer.data());
        const auto aligned = (raw + kPayloadAlignment - 1) / kPayloadAlignment * kPayloadAlignment;
        aligned_data       = reinterpret_cast<std::byte*>(aligned);
        check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "create weight staging completion event");
    }

    [[nodiscard]] std::byte* data() const noexcept { return aligned_data; }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            check_cuda(cudaEventSynchronize(event), "wait for weight staging transfer");
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    std::byte* aligned_data = nullptr;
    cudaEvent_t event       = nullptr;
    bool pending            = false;
};

// Also covers failure between a queued copy and its event record. Destruct before slots.
struct TransferCompletion {
    cudaStream_t stream;
    bool pending = true;

    ~TransferCompletion() {
        if (pending) { (void)cudaStreamSynchronize(stream); }
    }

    void finish() {
        check_cuda(cudaStreamSynchronize(stream), "complete weight upload");
        pending = false;
    }
};

struct CopyRange {
    std::size_t file       = 0;
    std::uint64_t begin    = 0;
    std::uint64_t end      = 0;
    std::byte* destination = nullptr;
};

// Binds a pipeline rank for a scope, and does nothing at all for the primary one: rank 0 is
// already current, and leaving the single-device path without even a redundant cudaSetDevice is
// what keeps an offload-free load exactly what it was.
class ScopedRank {
public:
    ScopedRank(DeviceContext& device, std::size_t rank) {
        if (rank != 0) { bound_.emplace(device, rank); }
    }

    ScopedRank(const ScopedRank&)            = delete;
    ScopedRank& operator=(const ScopedRank&) = delete;

private:
    std::optional<ScopedDeviceRank> bound_;
};

struct ReadSpan {
    std::size_t file    = 0;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

float read_divisor(const Reader& reader, ObjectHandle handle, const WeightGeometry& geometry,
                   std::span<const std::byte> host, MaterializationStats& stats) {
    if (geometry.format != QType::NVFP4) { return 0.0F; }
    std::array<std::byte, 4> word{};
    if (!host.empty()) {
        std::copy_n(host.data() + geometry.divisor_offset, word.size(), word.data());
    } else {
        const auto& object = reader.directory().tensor(handle);
        reader.read_into(checked_add(object.offset, geometry.divisor_offset, "weight divisor"),
                         word);
        stats.read_bytes = checked_add(stats.read_bytes, word.size(), "read bytes");
    }
    const auto value = std::bit_cast<float>(read_u32_le(word.data()));
    if (!std::isfinite(value) || value <= 0) {
        throw ArtifactError(reader.directory().tensor(handle).id +
                            ": invalid NVFP4 weight divisor");
    }
    return value;
}

} // namespace

const WeightParent& MaterializedArtifact::device_parent(ObjectHandle handle) const {
    if (!has_device(handle)) { throw ArtifactError("object has no device weight backing"); }
    return *objects_[handle.index].device;
}

const WeightParent& MaterializedArtifact::host_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].host) {
        throw ArtifactError("object has no Host weight backing");
    }
    return *objects_[handle.index].host;
}

std::span<const std::byte> MaterializedArtifact::host_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].host_data.empty()) {
        throw ArtifactError("object has no retained Host bytes");
    }
    return objects_[handle.index].host_data;
}

bool MaterializedArtifact::has_device(ObjectHandle handle) const noexcept {
    return handle.index < objects_.size() && objects_[handle.index].device.has_value();
}

const WeightParent& MaterializedArtifact::pinned_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].pinned) {
        throw ArtifactError("object has no pinned Host weight backing");
    }
    return *objects_[handle.index].pinned;
}

std::span<const std::byte> MaterializedArtifact::pinned_block() const noexcept {
    if (!pinned_) { return {}; }
    return {static_cast<const std::byte*>(pinned_->data()), pinned_->size()};
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer,
                                 std::unique_ptr<EvictableWeightPool> backing) {
    if (plan.source != &reader || plan.object_count != reader.directory().objects.size()) {
        throw ArtifactError("materialization plan belongs to another load session");
    }
    const StartupObserver no_observer;
    const auto& observer = startup_observer ? *startup_observer : no_observer;
    std::uint64_t total  = 0;
    for (const auto& placement : plan.device_objects) {
        total = checked_add(total, placement.bytes, "device payload bytes");
    }
    StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize, StartupProgressUnit::Bytes,
                            total);
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    out.stats_.file_bytes            = reader.file_bytes();
    out.stats_.read_bytes            = plan.prior_read_bytes;
    out.stats_.owned_value_bytes     = plan.owned_value_bytes;
    const std::size_t rank_count     = plan.device_rank_count();
    out.stats_.device_capacity_bytes = plan.device_capacity(0);
    out.stats_.device_object_count   = plan.device_objects.size();
    out.stats_.host_object_count     = plan.host_objects.size();
    // The pool backs rank 0's arena alone: its window is borrowed by a Vision encode on the primary
    // device, and later ranks get ordinary arenas below. The planner already refuses an evictable
    // placement anywhere but rank 0.
    out.arenas_.resize(rank_count);
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        const auto capacity = plan.device_capacity(rank);
        if (capacity > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError("device backing exceeds size_t");
        }
        if (rank == 0 && backing) {
            if (backing->arena().bytes < capacity) {
                throw ArtifactError("eviction pool arena is smaller than the materialization plan");
            }
            out.pool_      = std::move(backing);
            out.arenas_[0] = std::make_unique<DeviceArena>(out.pool_->arena());
        } else if (capacity) {
            // Each rank's arena must live in that rank's memory, so it is allocated while that
            // device is current.
            const ScopedRank bound(device, rank);
            out.arenas_[rank] = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
        }
        // Plan offsets are authoritative: an evictable tail starts at an aligned boundary that a
        // bump allocator would not reproduce. The arena accounts the whole planned extent.
        if (out.arenas_[rank] && capacity) {
            (void)out.arenas_[rank]->alloc_bytes(static_cast<std::size_t>(capacity), 1);
        }
        if (rank != 0) {
            out.stats_.offloaded_device_capacity_bytes = checked_add(
                out.stats_.offloaded_device_capacity_bytes, capacity, "offloaded device capacity");
        }
    }
    if (plan.pinned_capacity_bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("pinned Host backing exceeds size_t");
    }
    if (!plan.pinned_objects.empty()) {
        out.pinned_ = std::make_unique<PinnedHostBuffer>(
            static_cast<std::size_t>(plan.pinned_capacity_bytes));
        auto* const block          = static_cast<std::byte*>(out.pinned_->data());
        std::uint64_t previous_end = 0;
        for (const auto& placement : plan.pinned_objects) {
            reader.validate_object(placement.object);
            const auto& geometry = reader.geometry(placement.object);
            auto& storage        = out.objects_.at(placement.object.index);
            if (storage.pinned || geometry.bytes != placement.bytes ||
                placement.offset < previous_end || placement.alignment == 0 ||
                placement.offset % placement.alignment != 0 ||
                placement.offset > plan.pinned_capacity_bytes ||
                placement.bytes > plan.pinned_capacity_bytes - placement.offset) {
                throw ArtifactError("invalid or duplicate pinned Host placement");
            }
            previous_end = placement.offset + placement.bytes;
            const std::span<std::byte> destination(block + placement.offset,
                                                   static_cast<std::size_t>(placement.bytes));
            reader.read_into(reader.directory().tensor(placement.object).offset, destination);
            out.stats_.read_bytes =
                checked_add(out.stats_.read_bytes, placement.bytes, "pinned read bytes");
            const auto divisor =
                read_divisor(reader, placement.object, geometry, destination, out.stats_);
            storage.pinned = WeightParent{geometry, destination.data(), divisor};
        }
        out.stats_.pinned_bytes        = plan.pinned_capacity_bytes;
        out.stats_.pinned_object_count = plan.pinned_objects.size();
    }
    for (auto& placement : plan.host_objects) {
        reader.validate_object(placement.object);
        auto& storage = out.objects_.at(placement.object.index);
        if (!storage.host_data.empty()) { throw ArtifactError("duplicate Host placement"); }
        const auto& object = reader.directory().object(placement.object);
        if (placement.data.empty()) {
            placement.data = reader.read_object(placement.object);
            out.stats_.read_bytes =
                checked_add(out.stats_.read_bytes, placement.data.size(), "Host read bytes");
        }
        if (placement.data.size() != object_bytes(object)) {
            throw ArtifactError("Host placement size differs from object");
        }
        storage.host_data = std::move(placement.data);
        out.stats_.retained_host_bytes =
            checked_add(out.stats_.retained_host_bytes, storage.host_data.size(), "retained bytes");
        if (std::holds_alternative<TensorObject>(object)) {
            const auto& geometry = reader.geometry(placement.object);
            const auto divisor =
                read_divisor(reader, placement.object, geometry, storage.host_data, out.stats_);
            storage.host = WeightParent{geometry, storage.host_data.data(), divisor};
        }
    }
    std::vector<std::vector<CopyRange>> ranges_by_rank(rank_count);
    std::vector<const DevicePlacement*> transcoded;
    std::vector<std::uint64_t> previous_device_end(rank_count, 0);
    for (const auto& placement : plan.device_objects) {
        const auto& stored_geometry = reader.geometry(placement.object);
        const auto geometry =
            placement.transcode
                ? weight_geometry(*placement.transcode, QuantLayout::RowSplit,
                                  reader.directory().tensor(placement.object).shape)
                : stored_geometry;
        auto& object = out.objects_.at(placement.object.index);
        if (placement.rank >= rank_count) {
            throw ArtifactError("device placement names a rank the plan has no arena for");
        }
        const auto& arena    = out.arenas_[placement.rank];
        const auto rank_capacity = plan.device_capacity(placement.rank);
        if (object.device || !arena || geometry.bytes != placement.bytes ||
            (placement.transcode && object.host)) {
            throw ArtifactError("invalid or duplicate device placement");
        }
        if (placement.offset < previous_device_end[placement.rank] || placement.alignment == 0 ||
            placement.offset % placement.alignment != 0 || placement.offset > rank_capacity ||
            placement.bytes > rank_capacity - placement.offset) {
            throw ArtifactError("device offset differs from materialization plan");
        }
        previous_device_end[placement.rank] = placement.offset + placement.bytes;
        const DeviceSpan storage{static_cast<std::byte*>(arena->base()) + placement.offset,
                                 static_cast<std::size_t>(placement.bytes)};
        const auto divisor = object.host
                                 ? object.host->weight_scale_divisor
                                 : read_divisor(reader, placement.object, geometry, {}, out.stats_);
        object.device = WeightParent{geometry, static_cast<const std::byte*>(storage.data),
                                     divisor, stored_geometry.format};
        if (placement.transcode) {
            transcoded.push_back(&placement);
            continue;
        }
        const auto& descriptor = reader.directory().tensor(placement.object);
        for (const auto& segment : reader.segments(descriptor.offset, descriptor.bytes)) {
            ranges_by_rank[placement.rank].push_back(
                {segment.file_index, segment.file_offset,
                 checked_add(segment.file_offset, segment.bytes, "copy range"),
                 static_cast<std::byte*>(storage.data) + segment.destination_offset});
        }
    }
    // Transcoded objects are read whole, requantized on the host and uploaded synchronously.
    // They are the vocabulary-sized exceptions a startup option asked for, so bounded extra Host
    // memory (one object at a time) is acceptable and the direct-I/O pipeline stays unchanged.
    std::uint64_t transcoded_bytes = 0;
    for (const DevicePlacement* placement : transcoded) {
        const auto source = reader.read_object(placement->object);
        out.stats_.read_bytes =
            checked_add(out.stats_.read_bytes, source.size(), "transcode read bytes");
        std::vector<std::byte> encoded(static_cast<std::size_t>(placement->bytes));
        transcode_row_split(*placement->transcode,
                            reader.directory().tensor(placement->object).shape, source, encoded);
        const auto& parent = *out.objects_.at(placement->object.index).device;
        const ScopedRank bound(device, placement->rank);
        check_cuda(cudaMemcpy(const_cast<std::byte*>(parent.data), encoded.data(), encoded.size(),
                              cudaMemcpyHostToDevice),
                   "upload transcoded weight bytes");
        transcoded_bytes = checked_add(transcoded_bytes, encoded.size(), "transcoded bytes");
        phase.progress(transcoded_bytes, total);
    }
    out.stats_.h2d_bytes = transcoded_bytes;
    std::uint64_t copied = 0;
    const auto start     = std::chrono::steady_clock::now();
    bool uploaded_any    = false;
    // One staging pass per rank, each on its own device: a host-to-device copy and the event that
    // retires its slot both belong to the destination device, so they cannot be shared across
    // ranks. The ranks hold disjoint objects, so no payload is read twice. With one rank this is
    // the original single pass, on the original stream, with the original slot pool.
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        auto& ranges = ranges_by_rank[rank];
        if (ranges.empty()) { continue; }
        uploaded_any = true;
        const ScopedRank bound(device, rank);
        const cudaStream_t transfer_stream = device.transfer_stream_for_rank(rank);
        std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
            return std::tie(a.file, a.begin) < std::tie(b.file, b.begin);
        });
        std::vector<ReadSpan> spans;
        std::uint64_t aligned_bytes = 0;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const auto& range = ranges[i];
            if (i && ranges[i - 1].file == range.file && ranges[i - 1].end > range.begin) {
                throw ArtifactError("device source ranges overlap");
            }
            const auto begin = range.begin / kPayloadAlignment * kPayloadAlignment;
            if (spans.empty() || spans.back().file != range.file ||
                begin > align_up(spans.back().end, kPayloadAlignment, "direct range")) {
                spans.push_back({range.file, begin, range.end});
            } else {
                spans.back().end = std::max(spans.back().end, range.end);
            }
        }
        for (const auto& span : spans) {
            aligned_bytes =
                checked_add(aligned_bytes,
                            align_up(span.end - span.begin, kPayloadAlignment, "direct range bytes"),
                            "direct bytes");
        }
        const auto slot_bytes =
            static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_bytes));
        const auto slot_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_bytes - 1) / slot_bytes));
        std::vector<std::unique_ptr<Slot>> slots;
        // Passes run one after another and free their slots in between, so the process high-water
        // mark is the largest pass, not their sum.
        out.stats_.peak_staging_bytes =
            std::max<std::uint64_t>(out.stats_.peak_staging_bytes, slot_bytes * slot_count);
        StartupPhaseScope pin_phase(observer, StartupPhase::WeightsStagingPin,
                                    StartupProgressUnit::Bytes, slot_bytes * slot_count);
        for (std::size_t i = 0; i < slot_count; ++i) {
            slots.push_back(std::make_unique<Slot>(slot_bytes));
        }
        pin_phase.complete();
        TransferCompletion completion{transfer_stream};
        std::size_t next_slot  = 0;
        std::size_t next_range = 0;
        for (const auto& span : spans) {
            for (auto source = span.begin; source < span.end; source += slot_bytes) {
                auto& slot = *slots[next_slot++ % slot_count];
                slot.wait();
                const auto remaining = span.end - source;
                const auto request   = static_cast<std::size_t>(std::min<std::uint64_t>(
                    slot_bytes, align_up(remaining, kPayloadAlignment, "direct block bytes")));
                const auto received  = reader.read_direct(
                    span.file, source, {slot.data(), request});
                if (received < std::min<std::uint64_t>(request, remaining)) {
                    throw ArtifactError("direct read ended before the required payload");
                }
                out.stats_.read_bytes = checked_add(out.stats_.read_bytes, received, "read bytes");
                const auto chunk_end  = checked_add(source, received, "read block end");
                while (next_range < ranges.size() && ranges[next_range].file == span.file &&
                       ranges[next_range].begin < chunk_end) {
                    const auto& range = ranges[next_range];
                    const auto begin  = std::max(source, range.begin);
                    const auto end    = std::min(chunk_end, range.end);
                    if (begin < end) {
                        check_cuda(cudaMemcpyAsync(range.destination + (begin - range.begin),
                                                   slot.data() +
                                                       (begin - source),
                                                   static_cast<std::size_t>(end - begin),
                                                   cudaMemcpyHostToDevice, transfer_stream),
                                   "upload weight bytes");
                        copied = checked_add(copied, end - begin, "copied bytes");
                    }
                    if (range.end > chunk_end) { break; }
                    ++next_range;
                }
                check_cuda(cudaEventRecord(slot.event, transfer_stream),
                           "record weight staging completion");
                slot.pending = true;
                phase.progress(checked_add(copied, transcoded_bytes, "uploaded bytes"), total);
            }
        }
        for (const auto& slot : slots) { slot->wait(); }
        completion.finish();
        if (next_range != ranges.size()) { throw ArtifactError("incomplete device upload"); }
    }
    if (!uploaded_any) {
        phase.complete(transcoded_bytes, total);
        return out;
    }
    if (checked_add(copied, transcoded_bytes, "uploaded bytes") != total) {
        throw ArtifactError("incomplete device upload");
    }
    out.stats_.h2d_bytes = checked_add(copied, transcoded_bytes, "uploaded bytes");
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    phase.complete(checked_add(copied, transcoded_bytes, "uploaded bytes"), total);
    return out;
}

} // namespace ninfer::artifact
