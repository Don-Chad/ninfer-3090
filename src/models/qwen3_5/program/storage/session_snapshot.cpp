#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"
#include "core/host_kv_arena.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Retained-session snapshot format.
//
// A snapshot is the complete host image of one catalogued continuation: its resident prefix
// (ledger, exact identity and shortlist digests), its checkpoint directory (endpoint, rewrite
// checkpoint, long anchors), a deduplicated table of the StateImages those checkpoints name, and
// the paged Text and backend KV payload in logical page order. The KV payload covers every
// checkpoint frontier, because every checkpoint lies at or below the committed frontier.
//
// Byte order is the host's. A snapshot binds to the exact model binding and execution
// configuration of the server that wrote it; restore rejects any mismatch rather than
// reinterpreting bytes. Restore degrades gracefully: an optional checkpoint whose StateImage fits
// neither the Device nor the Host state pool, or whose anchor ordinal exceeds this server's anchor
// capacity, is dropped, while the endpoint is mandatory.

namespace ninfer::models::qwen3_5::detail {
namespace {

constexpr char kSessionSnapshotMagic[8]         = {'N', 'I', 'N', 'F', 'S', 'E', 'S', '1'};
constexpr std::uint32_t kSessionSnapshotVersion = 1;
constexpr std::size_t kMaximumModelBindingBytes = 4096;
constexpr std::uint32_t kMaximumSnapshotAnchors = 64;

class SnapshotWriter {
public:
    explicit SnapshotWriter(std::vector<std::uint8_t>& out) : out_(out) {}

    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out_.insert(out_.end(), begin, begin + count);
    }

    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }

    // Sizes a payload region the caller fills once the whole image is sized, so no asynchronous
    // copy ever records a destination the vector later reallocates away.
    std::size_t reserve_payload(std::size_t count) {
        const std::size_t offset = out_.size();
        out_.resize(out_.size() + count);
        return offset;
    }

private:
    std::vector<std::uint8_t>& out_;
};

class SnapshotReader {
public:
    explicit SnapshotReader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        std::memcpy(out, data_.data() + cursor_, count);
        cursor_ += count;
    }

    template <class T>
    [[nodiscard]] T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        bytes(&value, sizeof(T));
        return value;
    }

    // Borrows a payload region without copying; valid for the snapshot's lifetime.
    [[nodiscard]] const std::uint8_t* payload(std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        const std::uint8_t* region = data_.data() + cursor_;
        cursor_ += count;
        return region;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

template <class T>
void write_vector(SnapshotWriter& writer, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    writer.pod<std::uint64_t>(values.size());
    writer.bytes(values.data(), values.size() * sizeof(T));
}

template <class T>
std::vector<T> read_vector(SnapshotReader& reader, std::size_t maximum_count, const char* label) {
    const std::uint64_t count = reader.pod<std::uint64_t>();
    if (count > maximum_count) {
        throw std::invalid_argument(std::string("session snapshot ") + label +
                                    " count is out of range");
    }
    std::vector<T> values(static_cast<std::size_t>(count));
    reader.bytes(values.data(), values.size() * sizeof(T));
    return values;
}

void write_vision_items(SnapshotWriter& writer, const std::vector<VisionItem>& items) {
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(items.size()));
    for (const VisionItem& item : items) {
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(item.modality));
        writer.pod<std::int32_t>(item.grid.temporal);
        writer.pod<std::int32_t>(item.grid.height);
        writer.pod<std::int32_t>(item.grid.width);
        writer.pod<std::uint64_t>(item.patch_begin);
        writer.pod<std::uint64_t>(item.patch_count);
        writer.bytes(item.content_digest.data(), item.content_digest.size());
        write_vector(writer, item.timestamps);
        writer.pod<std::uint32_t>(static_cast<std::uint32_t>(item.token_spans.size()));
        for (const TokenSpan& span : item.token_spans) {
            writer.pod<std::uint64_t>(span.begin);
            writer.pod<std::uint64_t>(span.count);
        }
    }
}

std::vector<VisionItem> read_vision_items(SnapshotReader& reader, std::size_t tokens) {
    const std::uint32_t count = reader.pod<std::uint32_t>();
    if (count > tokens) {
        throw std::invalid_argument("session snapshot vision item count is out of range");
    }
    std::vector<VisionItem> items(count);
    for (VisionItem& item : items) {
        const std::uint8_t modality = reader.pod<std::uint8_t>();
        if (modality != static_cast<std::uint8_t>(PromptModality::Image) &&
            modality != static_cast<std::uint8_t>(PromptModality::Video)) {
            throw std::invalid_argument("session snapshot vision item modality is invalid");
        }
        item.modality      = static_cast<PromptModality>(modality);
        item.grid.temporal = reader.pod<std::int32_t>();
        item.grid.height   = reader.pod<std::int32_t>();
        item.grid.width    = reader.pod<std::int32_t>();
        item.patch_begin   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        item.patch_count   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        reader.bytes(item.content_digest.data(), item.content_digest.size());
        item.timestamps = read_vector<double>(reader, tokens, "vision timestamp");
        const std::uint32_t spans = reader.pod<std::uint32_t>();
        if (spans > tokens) {
            throw std::invalid_argument("session snapshot vision span count is out of range");
        }
        item.token_spans.resize(spans);
        for (TokenSpan& span : item.token_spans) {
            span.begin = static_cast<std::size_t>(reader.pod<std::uint64_t>());
            span.count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        }
    }
    return items;
}

// Everything a restoring server must match exactly for the payload bytes to mean the same thing.
struct SnapshotConfig {
    std::uint32_t kv_storage          = 0;
    std::uint32_t speculative_backend = 0;
    std::uint32_t proposal_head       = 0;
    std::uint32_t draft_window        = 0;
    std::uint32_t page_tokens         = 0;
    std::uint64_t state_image_bytes   = 0;
    std::uint32_t text_plane_count    = 0;
    std::uint64_t text_page_stride    = 0;
    std::uint32_t backend_plane_count = 0;
    std::uint64_t backend_page_stride = 0;

    [[nodiscard]] friend bool operator==(const SnapshotConfig&,
                                         const SnapshotConfig&) noexcept = default;
};

void write_config(SnapshotWriter& writer, const SnapshotConfig& config) {
    writer.pod(config.kv_storage);
    writer.pod(config.speculative_backend);
    writer.pod(config.proposal_head);
    writer.pod(config.draft_window);
    writer.pod(config.page_tokens);
    writer.pod(config.state_image_bytes);
    writer.pod(config.text_plane_count);
    writer.pod(config.text_page_stride);
    writer.pod(config.backend_plane_count);
    writer.pod(config.backend_page_stride);
}

SnapshotConfig read_config(SnapshotReader& reader) {
    SnapshotConfig config;
    config.kv_storage          = reader.pod<std::uint32_t>();
    config.speculative_backend = reader.pod<std::uint32_t>();
    config.proposal_head       = reader.pod<std::uint32_t>();
    config.draft_window        = reader.pod<std::uint32_t>();
    config.page_tokens         = reader.pod<std::uint32_t>();
    config.state_image_bytes   = reader.pod<std::uint64_t>();
    config.text_plane_count    = reader.pod<std::uint32_t>();
    config.text_page_stride    = reader.pod<std::uint64_t>();
    config.backend_plane_count = reader.pod<std::uint32_t>();
    config.backend_page_stride = reader.pod<std::uint64_t>();
    return config;
}

struct SnapshotSession {
    std::uint32_t tokens                     = 0;
    std::uint32_t execution_frontier         = 0;
    std::uint32_t text_kv_valid              = 0;
    std::uint32_t mtp_kv_valid               = 0;
    std::int32_t rope_delta                  = 0;
    std::uint8_t tail_hidden_valid           = 0;
    std::uint32_t text_committed_frontier    = 0;
    std::uint32_t backend_committed_frontier = 0;
    std::uint32_t text_pages                 = 0;
    std::uint32_t backend_pages              = 0;
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;
};

void write_work(SnapshotWriter& writer, const runtime::PrefillWork& work) {
    writer.pod(work.chunks);
    writer.pod(work.tokens);
    writer.pod(work.attention_pairs);
    writer.pod(work.vision_items);
    writer.pod(work.vision_patches);
}

runtime::PrefillWork read_work(SnapshotReader& reader) {
    runtime::PrefillWork work;
    work.chunks          = reader.pod<std::uint64_t>();
    work.tokens          = reader.pod<std::uint64_t>();
    work.attention_pairs = reader.pod<std::uint64_t>();
    work.vision_items    = reader.pod<std::uint64_t>();
    work.vision_patches  = reader.pod<std::uint64_t>();
    return work;
}

void write_session(SnapshotWriter& writer, const SnapshotSession& session) {
    writer.pod(session.tokens);
    writer.pod(session.execution_frontier);
    writer.pod(session.text_kv_valid);
    writer.pod(session.mtp_kv_valid);
    writer.pod(session.rope_delta);
    writer.pod(session.tail_hidden_valid);
    writer.pod(session.text_committed_frontier);
    writer.pod(session.backend_committed_frontier);
    writer.pod(session.text_pages);
    writer.pod(session.backend_pages);
    write_work(writer, session.rebuild_work);
    writer.pod(session.rebuild_tail_begin);
}

SnapshotSession read_session(SnapshotReader& reader) {
    SnapshotSession session;
    session.tokens                     = reader.pod<std::uint32_t>();
    session.execution_frontier         = reader.pod<std::uint32_t>();
    session.text_kv_valid              = reader.pod<std::uint32_t>();
    session.mtp_kv_valid               = reader.pod<std::uint32_t>();
    session.rope_delta                 = reader.pod<std::int32_t>();
    session.tail_hidden_valid          = reader.pod<std::uint8_t>();
    session.text_committed_frontier    = reader.pod<std::uint32_t>();
    session.backend_committed_frontier = reader.pod<std::uint32_t>();
    session.text_pages                 = reader.pod<std::uint32_t>();
    session.backend_pages              = reader.pod<std::uint32_t>();
    session.rebuild_work               = read_work(reader);
    session.rebuild_tail_begin         = reader.pod<std::uint32_t>();
    return session;
}

// Integrity check over the whole snapshot, stored as a trailing u64. A bit flip that keeps every
// length and domain valid would otherwise restore silently corrupted KV or state. Word-wise, so a
// multi-GB snapshot hashes at memory speed; not cryptographic, since the files are trusted local
// state and the threat is corruption, not tampering.
std::uint64_t snapshot_checksum(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash    = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(bytes.size());
    const std::size_t words = bytes.size() / sizeof(std::uint64_t);
    for (std::size_t index = 0; index < words; ++index) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + index * sizeof(std::uint64_t), sizeof(word));
        hash = (hash ^ word) * 0xff51afd7ed558ccdULL;
        hash ^= hash >> 32U;
    }
    for (std::size_t index = words * sizeof(std::uint64_t); index < bytes.size(); ++index) {
        hash = (hash ^ bytes[index]) * 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> 29U;
    }
    return hash;
}

void synchronize_streams(RankStreams streams) {
    for (std::size_t rank = 0; rank < streams.size(); ++rank) {
        CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }
}

} // namespace

std::string ledger_prefix_digest(std::span<const TokenId> tokens) {
    // FNV-1a 64 over the little-endian token id bytes. Deterministic across processes on one
    // byte order, which snapshot compatibility already requires.
    std::uint64_t hash = 1469598103934665603ULL;
    for (const TokenId token : tokens) {
        const auto value = static_cast<std::uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(value >> shift);
            hash *= 1099511628211ULL;
        }
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

std::uint32_t
ProgramImpl::continuation_depth(const ContinuationHandle& continuation) const noexcept {
    if (!valid_continuation(continuation)) { return 0; }
    return static_cast<std::uint32_t>(
        continuation_states[ContractAccess::index(continuation)].ledger.size());
}

std::string ProgramImpl::continuation_digest(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return ledger_prefix_digest(sequence.ledger);
}

std::vector<SlotCheckpoint>
ProgramImpl::continuation_checkpoints(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    const auto depth              = static_cast<std::uint32_t>(sequence.ledger.size());

    std::vector<std::uint32_t> frontiers;
    frontiers.reserve(sequence.long_anchors.size() + 2U);
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        frontiers.push_back(anchor.frontier);
    }
    if (sequence.rewrite_checkpoint.valid) {
        frontiers.push_back(sequence.rewrite_checkpoint.frontier);
    }
    // The endpoint restores at the executed frontier: the ledger's last token is sampled but not
    // yet in the KV or the state.
    if (sequence.endpoint_valid) { frontiers.push_back(sequence.execution_frontier); }
    std::sort(frontiers.begin(), frontiers.end());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());

    std::vector<SlotCheckpoint> out;
    out.reserve(frontiers.size());
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier > depth) { continue; }
        out.push_back(SlotCheckpoint{
            .frontier       = frontier,
            .session_digest = ledger_prefix_digest(std::span<const TokenId>(sequence.ledger.data(),
                                                                      frontier)),
        });
    }
    return out;
}

qwen3_5::ContinuationSummary
ProgramImpl::continuation_summary(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    return continuation_summary(continuation_states[ContractAccess::index(continuation)]);
}

SessionSnapshot ProgramImpl::save_continuation(const ContinuationHandle& continuation,
                                               std::string_view model_binding) {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    if (model_binding.size() > kMaximumModelBindingBytes) {
        throw std::invalid_argument("session snapshot model binding is too long");
    }
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("cannot snapshot a session during a resource transaction");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("session persistence does not support the DFlash backend");
    }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];

    const std::size_t tokens = sequence.ledger.size();
    if (tokens == 0 || tokens > capacity || sequence.prefix_identity.size() != tokens ||
        sequence.prefix_digests.size() != tokens || sequence.ledger_frontier != tokens ||
        sequence.execution_frontier > tokens || tokens - sequence.execution_frontier > 1 ||
        !sequence.endpoint_valid || !sequence.kv) {
        throw std::logic_error("retained session ledger and identity are inconsistent");
    }
    if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
        !state_store->valid(sequence.state.read)) {
        throw std::logic_error("retained session state binding is not a settled endpoint");
    }
    const StateImageHostLayout& state_layout = state_images->host_layout();

    // Checkpoint directory: each checkpoint names one entry of a deduplicated StateImage table,
    // since a rewrite checkpoint or anchor may alias the endpoint image.
    std::vector<StateImageHandle> unique_states;
    unique_states.reserve(2U + sequence.long_anchors.size());
    const auto image_index = [&](StateImageHandle handle) -> std::int32_t {
        if (!state_store->valid(handle) ||
            state_store->residency(handle) == StateReplicaResidency::None) {
            throw std::logic_error("retained session StateImage has no published replica");
        }
        for (std::size_t index = 0; index < unique_states.size(); ++index) {
            if (unique_states[index] == handle) { return static_cast<std::int32_t>(index); }
        }
        unique_states.push_back(handle);
        return static_cast<std::int32_t>(unique_states.size() - 1U);
    };
    const std::int32_t endpoint_image = image_index(sequence.state.read);
    std::int32_t rewrite_image        = -1;
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("retained rewrite checkpoint has no StateImage");
        }
        rewrite_image = image_index(*sequence.rewrite_state);
    }
    std::vector<std::int32_t> anchor_images;
    anchor_images.reserve(sequence.long_anchors.size());
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        anchor_images.push_back(image_index(anchor.state));
    }

    const DeviceKVPagePool& text_pool   = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout  = plan_host_kv_page_layout(text_pool.geometry());
    const qwen3_5::PagedKVCache* backend = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry());
    }

    const KVAddressSpaceHandle text_address = sequence.kv->text;
    const std::uint32_t text_committed = text_kv_addresses->committed_frontier(text_address);
    const std::uint32_t text_pages     = kv_pages_for_frontier(text_committed);
    if (text_pages == 0 || text_pages > text_kv_addresses->mapped_pages(text_address) ||
        sequence.execution_frontier > text_committed) {
        throw std::logic_error("retained session Text KV coverage is inconsistent");
    }
    std::uint32_t backend_committed = 0;
    std::uint32_t backend_pages     = 0;
    if (backend != nullptr) {
        if (!sequence.kv->backend) {
            throw std::logic_error("retained session has no backend KV address");
        }
        backend_committed = backend_kv_addresses->committed_frontier(*sequence.kv->backend);
        backend_pages     = kv_pages_for_frontier(backend_committed);
        if (backend_pages > backend_kv_addresses->mapped_pages(*sequence.kv->backend)) {
            throw std::logic_error("retained session backend KV coverage is inconsistent");
        }
    } else if (sequence.kv->backend) {
        throw std::logic_error("retained session backend KV address has no backing cache");
    }

    SnapshotConfig config;
    config.kv_storage          = static_cast<std::uint32_t>(kv_storage);
    config.speculative_backend = static_cast<std::uint32_t>(speculative_backend);
    config.proposal_head       = static_cast<std::uint32_t>(proposal_head);
    config.draft_window        = draft_window;
    config.page_tokens         = static_cast<std::uint32_t>(kPagedKVPageSize);
    config.state_image_bytes   = state_layout.image_bytes;
    config.text_plane_count    = static_cast<std::uint32_t>(text_layout.planes.size());
    config.text_page_stride    = text_layout.page_stride;
    if (backend_layout) {
        config.backend_plane_count = static_cast<std::uint32_t>(backend_layout->planes.size());
        config.backend_page_stride = backend_layout->page_stride;
    }

    SnapshotSession session;
    session.tokens                     = static_cast<std::uint32_t>(tokens);
    session.execution_frontier         = sequence.execution_frontier;
    session.text_kv_valid              = sequence.text_kv_valid;
    session.mtp_kv_valid               = sequence.mtp_kv_valid;
    session.rope_delta                 = sequence.rope_delta;
    session.tail_hidden_valid          = sequence.tail_hidden_valid ? 1 : 0;
    session.text_committed_frontier    = text_committed;
    session.backend_committed_frontier = backend_committed;
    session.text_pages                 = text_pages;
    session.backend_pages              = backend_pages;
    session.rebuild_work               = sequence.rebuild_work;
    session.rebuild_tail_begin         = sequence.rebuild_tail_begin;

    SessionSnapshot snapshot;
    snapshot.tokens         = session.tokens;
    snapshot.session_digest = ledger_prefix_digest(sequence.ledger);
    SnapshotWriter writer(snapshot.bytes);
    writer.bytes(kSessionSnapshotMagic, sizeof(kSessionSnapshotMagic));
    writer.pod(kSessionSnapshotVersion);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(model_binding.size()));
    writer.bytes(model_binding.data(), model_binding.size());
    write_config(writer, config);
    write_session(writer, session);
    write_vector(writer, sequence.ledger);
    write_vector(writer, sequence.prefix_identity.token_types());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        write_vector(writer, sequence.prefix_identity.position_axis(axis));
    }
    write_vision_items(writer, sequence.prefix_identity.vision_items());
    write_vector(writer, sequence.prefix_identity.rewrite_execution_frontiers());
    write_vector(writer, sequence.prefix_digests.image());

    writer.pod<std::uint8_t>(sequence.rewrite_checkpoint.valid ? 1 : 0);
    writer.pod<std::uint8_t>(static_cast<std::uint8_t>(sequence.rewrite_checkpoint.kind));
    writer.pod<std::uint32_t>(sequence.rewrite_checkpoint.frontier);
    write_work(writer, sequence.rewrite_checkpoint.rebuild_work);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(sequence.long_anchors.size()));
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        writer.pod<std::uint32_t>(anchor.frontier);
        writer.pod<std::uint32_t>(anchor.ordinal);
        write_work(writer, anchor.rebuild_work);
    }
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(unique_states.size()));
    writer.pod<std::int32_t>(endpoint_image);
    writer.pod<std::int32_t>(rewrite_image);
    for (const std::int32_t index : anchor_images) { writer.pod<std::int32_t>(index); }

    const std::size_t state_offset =
        writer.reserve_payload(state_layout.image_bytes * unique_states.size());
    const std::size_t text_kv_offset =
        writer.reserve_payload(config.text_page_stride * session.text_pages);
    const std::size_t backend_kv_offset =
        writer.reserve_payload(config.backend_page_stride * session.backend_pages);
    const std::size_t checksum_offset = writer.reserve_payload(sizeof(std::uint64_t));

    // The catalogued images and pages are immutable and no transaction is open, but the unit that
    // last wrote them may still be in flight on the compute streams, which also order these
    // copies behind it.
    const RankStreams streams = compute_streams;
    std::uint8_t* const base  = snapshot.bytes.data();
    for (std::size_t index = 0; index < unique_states.size(); ++index) {
        const StateImageHandle image  = unique_states[index];
        std::uint8_t* const image_out = base + state_offset + index * state_layout.image_bytes;
        if (state_store->residency(image) == StateReplicaResidency::HostOnly) {
            const qwen3_5::HostStateImageConstView view = state_store->host_view(image);
            std::memcpy(image_out, view.data, state_layout.image_bytes);
        } else {
            state_images->copy_to_host(
                state_store->physical_slot(image),
                qwen3_5::HostStateImageView{reinterpret_cast<std::byte*>(image_out),
                                            &state_layout},
                streams);
        }
    }

    // Device-resident page runs go through the pool copier; demoted pages are read from their
    // current Host replicas without touching the Device.
    const auto copy_address_pages = [&](const KVAddressSpaceStore& addresses,
                                        const LogicalKVPageStore& pages,
                                        KVAddressSpaceHandle address, std::uint32_t page_count,
                                        const HostKVPageLayout& layout,
                                        std::size_t payload_offset) {
        const DeviceKVPagePool& pool = pages.physical_pool();
        std::vector<DeviceKVPageHandle> run;
        run.reserve(page_count);
        std::uint32_t run_begin = 0;
        const auto flush_run    = [&] {
            if (run.empty()) { return; }
            pool.copy_to_host(std::span<const DeviceKVPageHandle>(run.data(), run.size()),
                              reinterpret_cast<std::byte*>(
                                  base + payload_offset +
                                  static_cast<std::size_t>(run_begin) * layout.page_stride),
                              layout, streams);
            run.clear();
        };
        for (std::uint32_t page = 0; page < page_count; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.device_resident(logical)) {
                if (run.empty()) { run_begin = page; }
                run.push_back(pages.physical(logical));
                continue;
            }
            flush_run();
            if (!pages.host_resident(logical) || !pages.host_replica_current(logical) ||
                !host_kv_extents) {
                throw std::logic_error("retained session KV page has no current replica");
            }
            const HostKVPageReplica& replica    = pages.host_replica(logical);
            const HostKVAllocationConstView view = host_kv_extents->view(replica.extent);
            if (view.layout().page_stride != layout.page_stride ||
                replica.page_offset >= view.page_count()) {
                throw std::logic_error("retained session Host replica layout is inconsistent");
            }
            std::memcpy(base + payload_offset + static_cast<std::size_t>(page) * layout.page_stride,
                        view.data() +
                            static_cast<std::size_t>(replica.page_offset) * layout.page_stride,
                        layout.page_stride);
        }
        flush_run();
    };
    copy_address_pages(*text_kv_addresses, *text_kv_pages, text_address, session.text_pages,
                       text_layout, text_kv_offset);
    if (backend_layout && session.backend_pages != 0) {
        copy_address_pages(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                           session.backend_pages, *backend_layout, backend_kv_offset);
    }
    synchronize_streams(streams);
    const std::uint64_t checksum =
        snapshot_checksum(std::span<const std::uint8_t>(snapshot.bytes.data(), checksum_offset));
    std::memcpy(snapshot.bytes.data() + checksum_offset, &checksum, sizeof(checksum));

    // The layout a store needs: the bulk regions, and the key of each checkpoint as the request
    // path computes it (from the continuation summary, so the two cannot drift apart).
    for (std::size_t index = 0; index < unique_states.size(); ++index) {
        snapshot.regions.push_back(SessionSnapshotRegion{
            .offset = state_offset + index * state_layout.image_bytes,
            .length = state_layout.image_bytes});
    }
    snapshot.regions.push_back(SessionSnapshotRegion{
        .offset = text_kv_offset, .length = static_cast<std::uint64_t>(config.text_page_stride) *
                                            session.text_pages});
    if (session.backend_pages != 0) {
        snapshot.regions.push_back(SessionSnapshotRegion{
            .offset = backend_kv_offset,
            .length = static_cast<std::uint64_t>(config.backend_page_stride) *
                      session.backend_pages});
    }
    const qwen3_5::ContinuationSummary summary = continuation_summary(sequence);
    if (summary.endpoint) { snapshot.checkpoints.push_back(summary.endpoint->shortlist_key); }
    if (summary.rewrite) { snapshot.checkpoints.push_back(summary.rewrite->shortlist_key); }
    for (const auto& anchor : summary.long_anchors) {
        snapshot.checkpoints.push_back(anchor.shortlist_key);
    }
    snapshot.prefix_digests = sequence.prefix_digests.image();
    return snapshot;
}

ContinuationHandle ProgramImpl::restore_continuation(std::span<const std::uint8_t> snapshot,
                                                     std::string_view model_binding) {
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("cannot restore a session during a resource transaction");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("session persistence does not support the DFlash backend");
    }

    if (snapshot.size() < sizeof(kSessionSnapshotMagic) + sizeof(std::uint64_t)) {
        throw std::invalid_argument("session snapshot is truncated");
    }
    const std::span<const std::uint8_t> body = snapshot.first(snapshot.size() - sizeof(std::uint64_t));
    std::uint64_t stored_checksum            = 0;
    std::memcpy(&stored_checksum, snapshot.data() + body.size(), sizeof(stored_checksum));
    SnapshotReader reader(body);
    char magic[sizeof(kSessionSnapshotMagic)] = {};
    reader.bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kSessionSnapshotMagic, sizeof(magic)) != 0) {
        throw std::invalid_argument("file is not a session snapshot");
    }
    // Verified before anything is allocated or uploaded.
    if (snapshot_checksum(body) != stored_checksum) {
        throw std::invalid_argument("session snapshot is corrupt: checksum mismatch");
    }
    if (reader.pod<std::uint32_t>() != kSessionSnapshotVersion) {
        throw std::invalid_argument("session snapshot version is unsupported");
    }
    const std::uint32_t binding_bytes = reader.pod<std::uint32_t>();
    if (binding_bytes > kMaximumModelBindingBytes) {
        throw std::invalid_argument("session snapshot model binding is too long");
    }
    std::string binding(binding_bytes, '\0');
    reader.bytes(binding.data(), binding_bytes);
    if (binding != model_binding) {
        throw std::invalid_argument("session snapshot was saved for a different model");
    }

    const StateImageHostLayout& state_layout = state_images->host_layout();
    const DeviceKVPagePool& text_pool        = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout       = plan_host_kv_page_layout(text_pool.geometry());
    const qwen3_5::PagedKVCache* backend     = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry());
    }

    SnapshotConfig expected;
    expected.kv_storage          = static_cast<std::uint32_t>(kv_storage);
    expected.speculative_backend = static_cast<std::uint32_t>(speculative_backend);
    expected.proposal_head       = static_cast<std::uint32_t>(proposal_head);
    expected.draft_window        = draft_window;
    expected.page_tokens         = static_cast<std::uint32_t>(kPagedKVPageSize);
    expected.state_image_bytes   = state_layout.image_bytes;
    expected.text_plane_count    = static_cast<std::uint32_t>(text_layout.planes.size());
    expected.text_page_stride    = text_layout.page_stride;
    if (backend_layout) {
        expected.backend_plane_count = static_cast<std::uint32_t>(backend_layout->planes.size());
        expected.backend_page_stride = backend_layout->page_stride;
    }
    const SnapshotConfig config = read_config(reader);
    if (config.kv_storage != expected.kv_storage || config.page_tokens != expected.page_tokens ||
        config.text_plane_count != expected.text_plane_count ||
        config.text_page_stride != expected.text_page_stride) {
        throw std::invalid_argument("session snapshot KV configuration does not match the server");
    }
    if (config.speculative_backend != expected.speculative_backend ||
        config.proposal_head != expected.proposal_head ||
        config.draft_window != expected.draft_window ||
        config.backend_plane_count != expected.backend_plane_count ||
        config.backend_page_stride != expected.backend_page_stride) {
        throw std::invalid_argument(
            "session snapshot speculative configuration does not match the server");
    }
    if (config.state_image_bytes != expected.state_image_bytes) {
        throw std::invalid_argument("session snapshot state geometry does not match the server");
    }

    const SnapshotSession session = read_session(reader);
    if (session.tokens == 0 || session.tokens > capacity) {
        throw std::invalid_argument("session snapshot depth exceeds the server context");
    }
    if (session.execution_frontier > session.tokens ||
        session.tokens - session.execution_frontier > 1 ||
        session.text_kv_valid > session.text_committed_frontier ||
        session.execution_frontier > session.text_committed_frontier ||
        session.text_committed_frontier > capacity ||
        session.mtp_kv_valid > session.backend_committed_frontier ||
        session.backend_committed_frontier > capacity ||
        session.rebuild_work.tokens > session.execution_frontier ||
        session.rebuild_tail_begin > session.execution_frontier) {
        throw std::invalid_argument("session snapshot frontiers are inconsistent");
    }
    if (session.text_pages == 0 ||
        session.text_pages != kv_pages_for_frontier(session.text_committed_frontier) ||
        session.backend_pages != kv_pages_for_frontier(session.backend_committed_frontier) ||
        (!backend_layout && session.backend_pages != 0)) {
        throw std::invalid_argument("session snapshot page counts are out of range");
    }

    std::vector<TokenId> ledger = read_vector<TokenId>(reader, session.tokens, "ledger");
    const std::uint32_t vocabulary = parameters.model.config().text.vocab_size;
    if (ledger.size() != session.tokens) {
        throw std::invalid_argument("session snapshot ledger does not match its depth");
    }
    for (const TokenId id : ledger) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= vocabulary) {
            throw std::invalid_argument("session snapshot ledger token is out of domain");
        }
    }
    std::vector<std::uint8_t> token_types =
        read_vector<std::uint8_t>(reader, session.tokens, "token type");
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) {
        axis = read_vector<std::int32_t>(reader, session.tokens, "position");
    }
    std::vector<VisionItem> vision_items = read_vision_items(reader, session.tokens);
    std::vector<std::uint32_t> rewrite_frontiers =
        read_vector<std::uint32_t>(reader, session.tokens, "rewrite frontier");
    std::vector<std::array<std::uint64_t, 2>> digest_image =
        read_vector<std::array<std::uint64_t, 2>>(
            reader, static_cast<std::size_t>(session.tokens) + 1U, "shortlist digest");
    if (token_types.size() != session.tokens ||
        digest_image.size() != static_cast<std::size_t>(session.tokens) + 1U) {
        throw std::invalid_argument("session snapshot identity does not match its depth");
    }
    if (!vision_items.empty() && !vision_enabled) {
        throw std::invalid_argument("session snapshot holds media but Vision is disabled");
    }

    const std::uint8_t rewrite_valid        = reader.pod<std::uint8_t>();
    const std::uint8_t rewrite_kind         = reader.pod<std::uint8_t>();
    const std::uint32_t rewrite_frontier    = reader.pod<std::uint32_t>();
    const runtime::PrefillWork rewrite_work = read_work(reader);
    if (rewrite_valid > 1 ||
        (rewrite_valid != 0 &&
         (rewrite_frontier == 0 || rewrite_frontier > session.tokens ||
          rewrite_kind > static_cast<std::uint8_t>(RewriteCheckpointKind::ResponseReplay)))) {
        throw std::invalid_argument("session snapshot rewrite checkpoint is inconsistent");
    }
    struct SnapshotAnchor {
        std::uint32_t frontier = 0;
        std::uint32_t ordinal  = 0;
        runtime::PrefillWork rebuild_work;
        std::int32_t image = -1;
    };
    const std::uint32_t anchor_count = reader.pod<std::uint32_t>();
    if (anchor_count > kMaximumSnapshotAnchors) {
        throw std::invalid_argument("session snapshot anchor count is out of range");
    }
    std::vector<SnapshotAnchor> anchors(anchor_count);
    for (SnapshotAnchor& anchor : anchors) {
        anchor.frontier     = reader.pod<std::uint32_t>();
        anchor.ordinal      = reader.pod<std::uint32_t>();
        anchor.rebuild_work = read_work(reader);
        if (anchor.frontier == 0 || anchor.frontier > session.tokens || anchor.ordinal == 0) {
            throw std::invalid_argument("session snapshot long anchor is inconsistent");
        }
    }
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (anchors[previous].ordinal == anchors[index].ordinal) {
                throw std::invalid_argument("session snapshot anchor ordinals are not unique");
            }
        }
    }
    const std::uint32_t image_count = reader.pod<std::uint32_t>();
    if (image_count == 0 || image_count > 2U + anchor_count) {
        throw std::invalid_argument("session snapshot StateImage table is out of range");
    }
    const auto read_image_index = [&](bool required) -> std::int32_t {
        const std::int32_t index = reader.pod<std::int32_t>();
        if (index >= static_cast<std::int32_t>(image_count) || (required && index < 0) ||
            index < -1) {
            throw std::invalid_argument("session snapshot StateImage index is out of range");
        }
        return index;
    };
    const std::int32_t endpoint_image = read_image_index(true);
    const std::int32_t rewrite_image  = read_image_index(false);
    if ((rewrite_valid != 0) != (rewrite_image >= 0)) {
        throw std::invalid_argument("session snapshot rewrite StateImage index is inconsistent");
    }
    for (SnapshotAnchor& anchor : anchors) { anchor.image = read_image_index(true); }

    std::vector<const std::uint8_t*> image_payloads(image_count);
    for (std::uint32_t index = 0; index < image_count; ++index) {
        image_payloads[index] = reader.payload(state_layout.image_bytes);
    }
    const std::uint8_t* text_payload =
        reader.payload(static_cast<std::size_t>(config.text_page_stride) * session.text_pages);
    const std::uint8_t* backend_payload =
        session.backend_pages != 0
            ? reader.payload(static_cast<std::size_t>(config.backend_page_stride) *
                             session.backend_pages)
            : nullptr;
    if (reader.remaining() != 0) {
        throw std::invalid_argument("session snapshot has trailing bytes");
    }

    DeviceKVPagePool& text_device_pool = text_kv_pages->physical_pool();
    if (text_device_pool.available_pages() < session.text_pages ||
        (backend_layout &&
         backend_kv_pages->physical_pool().available_pages() < session.backend_pages)) {
        throw std::invalid_argument(
            "session snapshot does not fit the free KV capacity; evict other sessions first");
    }

    // Page uploads go through a temporarily activated address space, which needs an execution
    // row. Rows belong to lanes and only active requests hold them, so any idle lane's row works.
    std::optional<std::int32_t> free_row;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (requests[lane].lifecycle == Lifecycle::Empty &&
            active_continuations[lane] == continuation_capacity) {
            free_row = static_cast<std::int32_t>(lane);
            break;
        }
    }
    if (!free_row) {
        throw std::invalid_argument("session restore requires an idle execution lane");
    }

    const std::optional<std::uint32_t> slot_index = allocate_continuation_slot();
    if (!slot_index) {
        throw std::invalid_argument(
            "session snapshot does not fit the continuation catalog; evict other sessions first");
    }

    const RankStreams streams = compute_streams;
    std::optional<StateImageHandle> state;
    std::vector<std::optional<StateImageHandle>> extra_images(image_count);
    std::optional<KVAddressSpaceHandle> text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    try {
        // Every restored image prefers a Device slot and falls back to a HostOnly replica, the
        // shape a demoted checkpoint has, when the Device pool is occupied.
        const auto stage_image =
            [&](const std::uint8_t* payload) -> std::optional<StateImageHandle> {
            const qwen3_5::HostStateImageConstView view{
                reinterpret_cast<const std::byte*>(payload), &state_layout};
            std::optional<StateImageHandle> handle = state_store->reserve_reset(streams);
            if (handle) {
                state_images->copy_from_host(view, state_store->physical_slot(*handle), streams);
                return handle;
            }
            return state_store->adopt_host_image(view);
        };
        state = stage_image(image_payloads[static_cast<std::size_t>(endpoint_image)]);
        if (!state) {
            throw std::invalid_argument(
                "session snapshot does not fit the free state capacity; evict other sessions "
                "first");
        }

        const std::uint32_t anchor_capacity =
            context_cache.max_long_anchors_per_continuation.value_or(0);
        const auto upload_image = [&](std::int32_t index) {
            const auto position = static_cast<std::size_t>(index);
            if (index == endpoint_image || extra_images[position]) { return; }
            extra_images[position] = stage_image(image_payloads[position]);
        };
        if (rewrite_valid != 0) { upload_image(rewrite_image); }
        for (const SnapshotAnchor& anchor : anchors) {
            if (anchor.ordinal <= anchor_capacity) { upload_image(anchor.image); }
        }

        const auto build_address = [&](KVAddressSpaceStore& addresses, DeviceKVPagePool& pool,
                                       std::uint32_t page_count, std::uint32_t committed,
                                       const HostKVPageLayout& layout,
                                       const std::uint8_t* payload) -> KVAddressSpaceHandle {
            std::optional<KVAddressSpaceHandle> address = addresses.create_inactive();
            if (!address) {
                throw std::invalid_argument(
                    "session snapshot does not fit the KV address capacity; evict other "
                    "sessions first");
            }
            if (page_count == 0) { return *address; }
            try {
                addresses.activate(*address, page_count, *free_row, streams);
                addresses.ensure_mapped_to_tokens(*address, committed, streams);
                if (addresses.mapped_pages(*address) != page_count) {
                    throw std::invalid_argument(
                        "session snapshot KV page count does not match its committed tokens");
                }
                addresses.commit_frontier(*address, committed);
                std::vector<DeviceKVPageHandle> destinations;
                destinations.reserve(page_count);
                for (std::uint32_t page = 0; page < page_count; ++page) {
                    destinations.push_back(addresses.physical_page(*address, page));
                }
                pool.copy_from_host(reinterpret_cast<const std::byte*>(payload), layout,
                                    std::span<const DeviceKVPageHandle>(destinations.data(),
                                                                        destinations.size()),
                                    streams);
            } catch (...) {
                if (addresses.active(*address)) { addresses.deactivate(*address); }
                (void)addresses.release(*address);
                throw;
            }
            return *address;
        };
        text_address = build_address(*text_kv_addresses, text_device_pool, session.text_pages,
                                     session.text_committed_frontier, text_layout, text_payload);
        if (backend_layout) {
            backend_address = build_address(
                *backend_kv_addresses, backend_kv_pages->physical_pool(), session.backend_pages,
                session.backend_committed_frontier, *backend_layout, backend_payload);
        }
        synchronize_streams(streams);

        // Adoption point: the sequence owns the handles from here, so the locals are disarmed as
        // they are handed over and failure collapses to releasing the continuation slot.
        SequenceState& sequence = continuation_states[*slot_index];
        if (state_store->role(*state) == StateImageRole::ActiveMutable) {
            state_store->freeze(*state);
        }
        sequence.state = ActiveStateBinding{.read = *state, .write = *state};
        state.reset();
        sequence.kv.emplace(SequenceKVBundle{.text = *text_address, .backend = backend_address});
        text_address.reset();
        backend_address.reset();
        sequence.lane = static_cast<std::uint32_t>(*free_row);
        release_sequence_growth_entitlement(sequence);
        unbind_sequence_kv(sequence);

        sequence.ledger.assign(ledger.begin(), ledger.end());
        sequence.prefix_identity.restore(std::move(token_types), std::move(positions),
                                         std::move(vision_items), std::move(rewrite_frontiers));
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1U);
        sequence.prefix_digests.restore(std::move(digest_image));
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1U);
        sequence.execution_frontier      = session.execution_frontier;
        sequence.ledger_frontier         = session.tokens;
        sequence.text_kv_valid           = session.text_kv_valid;
        sequence.mtp_kv_valid            = session.mtp_kv_valid;
        sequence.dflash_context_frontier = 0;
        sequence.rope_delta              = session.rope_delta;
        sequence.mtp_draft_count         = 0;
        sequence.tail_hidden_valid       = session.tail_hidden_valid != 0;
        sequence.endpoint_valid          = true;
        sequence.rewrite_checkpoint      = {};
        sequence.rewrite_state.reset();
        sequence.reserved_state.reset();
        sequence.long_anchors.clear();
        sequence.rebuild_work       = session.rebuild_work;
        sequence.rebuild_tail_begin = session.rebuild_tail_begin;

        // Surviving checkpoints: freeze the uploaded images and give the sequence one checkpoint
        // reference per naming checkpoint; the endpoint keeps zero references, as after finish().
        for (std::optional<StateImageHandle>& image : extra_images) {
            if (image && state_store->role(*image) == StateImageRole::ActiveMutable) {
                state_store->freeze(*image);
            }
        }
        const auto resolve_image = [&](std::int32_t index) -> std::optional<StateImageHandle> {
            if (index == endpoint_image) { return sequence.state.read; }
            return extra_images[static_cast<std::size_t>(index)];
        };
        if (rewrite_valid != 0) {
            if (const std::optional<StateImageHandle> handle = resolve_image(rewrite_image)) {
                sequence.rewrite_state      = *handle;
                sequence.rewrite_checkpoint = RewriteCheckpoint{
                    .valid        = true,
                    .kind         = static_cast<RewriteCheckpointKind>(rewrite_kind),
                    .frontier     = rewrite_frontier,
                    .rebuild_work = rewrite_work,
                };
                state_store->retain_checkpoint_reference(*handle);
            }
        }
        for (const SnapshotAnchor& anchor : anchors) {
            if (anchor.ordinal > anchor_capacity) { continue; }
            const std::optional<StateImageHandle> handle = resolve_image(anchor.image);
            if (!handle) { continue; }
            sequence.long_anchors.push_back(LongAnchorCheckpoint{
                .state        = *handle,
                .frontier     = anchor.frontier,
                .ordinal      = anchor.ordinal,
                .rebuild_work = anchor.rebuild_work,
            });
            state_store->retain_checkpoint_reference(*handle);
        }
        // Images staged for checkpoints that did not survive are not owned by the sequence.
        for (std::optional<StateImageHandle>& image : extra_images) {
            if (image && state_store->checkpoint_references(*image) == 0) {
                (void)state_store->release(*image);
                image.reset();
            }
        }
        refresh_state_views(sequence);

        text_kv_addresses->set_checkpoint_requirement(sequence.kv->text,
                                                      sequence.execution_frontier);
        if (sequence.kv->backend) {
            backend_kv_addresses->set_checkpoint_requirement(*sequence.kv->backend,
                                                             backend_kv_valid(sequence));
        }
        // Mint the summary once so a malformed rebuild surfaces here rather than at catalog
        // adoption.
        (void)continuation_summary(sequence);

        continuation_slots[*slot_index].role = ContinuationSlotRole::Catalogued;
        advance_resource_revision();
        return ContractAccess::make_continuation(this, *slot_index,
                                                 continuation_slots[*slot_index].generation);
    } catch (...) {
        try {
            synchronize_streams(streams);
        } catch (...) {}
        if (backend_address) {
            if (backend_kv_addresses->active(*backend_address)) {
                backend_kv_addresses->deactivate(*backend_address);
            }
            (void)backend_kv_addresses->release(*backend_address);
        }
        if (text_address) {
            if (text_kv_addresses->active(*text_address)) {
                text_kv_addresses->deactivate(*text_address);
            }
            (void)text_kv_addresses->release(*text_address);
        }
        if (state) { (void)state_store->release(*state); }
        // Images the sequence already adopted carry checkpoint references, so this release
        // refuses them and the slot teardown below owns them instead.
        for (std::optional<StateImageHandle>& image : extra_images) {
            if (image) { (void)state_store->release(*image); }
        }
        release_continuation_slot_best_effort(*slot_index);
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
