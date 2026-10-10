#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"
#include "core/host_kv_arena.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Checkpoint image format.
//
// An image is the complete Host representation of a set of retained checkpoints, normally the
// recovery points of one continuation (its endpoint, input-replay point and long anchors). It holds,
// once each, every logical KV page those points cover (Text, and the backend draft KV when there is
// one), every StateImage they name and every exact prefix identity they carry, followed by a table
// that rebuilds each point from those pieces. The payload is exactly what a Host demotion of the same
// points would hold, so importing an image yields Host-resident checkpoints indistinguishable from
// demoted ones: a later binding restores them through the ordinary Host-to-Device path.
//
// Byte order is the host's. An image binds to the Program configuration (KV storage, speculative
// layout, page and StateImage geometry) and to the caller's model binding; import rejects any
// mismatch instead of reinterpreting bytes. A trailing checksum covers the whole image.
//
// KV pages are listed in the order the histories first reach them, so successive images of one
// conversation share the bytes of their common prefix page for page: a content-addressed store keeps
// them once.

namespace ninfer::models::qwen3_5::detail {
namespace {

constexpr char kImageMagic[8]                   = {'N', 'I', 'N', 'F', 'C', 'K', 'P', 'T'};
constexpr std::uint32_t kImageVersion           = 1;
constexpr std::size_t kMaximumModelBindingBytes = 4096;
constexpr std::uint32_t kMaximumImagePoints     = 64;

class ImageWriter {
public:
    explicit ImageWriter(std::vector<std::uint8_t>& out) : out_(out) {}

    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out_.insert(out_.end(), begin, begin + count);
    }

    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }

    template <class T>
    void vector(const std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        pod<std::uint64_t>(values.size());
        bytes(values.data(), values.size() * sizeof(T));
    }

    // Sizes a payload region the caller fills once the whole image is sized, so no asynchronous
    // copy records a destination the vector later reallocates away.
    std::size_t reserve(std::size_t count) {
        const std::size_t offset = out_.size();
        out_.resize(out_.size() + count);
        return offset;
    }

private:
    std::vector<std::uint8_t>& out_;
};

class ImageReader {
public:
    explicit ImageReader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("checkpoint image is truncated");
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

    template <class T>
    [[nodiscard]] std::vector<T> vector(std::size_t maximum, const char* label) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto count = pod<std::uint64_t>();
        if (count > maximum) {
            throw std::invalid_argument(std::string("checkpoint image ") + label +
                                        " count is out of range");
        }
        std::vector<T> values(static_cast<std::size_t>(count));
        bytes(values.data(), values.size() * sizeof(T));
        return values;
    }

    [[nodiscard]] std::span<const std::uint8_t> payload(std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("checkpoint image payload is truncated");
        }
        const auto region = data_.subspan(cursor_, count);
        cursor_ += count;
        return region;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

// Word-wise, so a multi-GB image hashes at memory speed. Not cryptographic: the store is trusted
// local state and the threat is corruption, not tampering.
std::uint64_t image_checksum(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash      = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(bytes.size());
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

// Everything an importing Program must match for the payload bytes to mean the same thing.
struct ImageConfig {
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

    friend bool operator==(const ImageConfig&, const ImageConfig&) noexcept = default;
};

void write_config(ImageWriter& writer, const ImageConfig& config) {
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

ImageConfig read_config(ImageReader& reader) {
    ImageConfig config;
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

void write_vision_items(ImageWriter& writer, const std::vector<VisionItem>& items) {
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(items.size()));
    for (const VisionItem& item : items) {
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(item.modality));
        writer.pod<std::int32_t>(item.grid.temporal);
        writer.pod<std::int32_t>(item.grid.height);
        writer.pod<std::int32_t>(item.grid.width);
        writer.pod<std::uint64_t>(item.patch_begin);
        writer.pod<std::uint64_t>(item.patch_count);
        writer.bytes(item.content_digest.data(), item.content_digest.size());
        writer.vector(item.timestamps);
        writer.pod<std::uint32_t>(static_cast<std::uint32_t>(item.token_spans.size()));
        for (const TokenSpan& span : item.token_spans) {
            writer.pod<std::uint64_t>(span.begin);
            writer.pod<std::uint64_t>(span.count);
        }
    }
}

std::vector<VisionItem> read_vision_items(ImageReader& reader, std::size_t tokens) {
    const auto count = reader.pod<std::uint32_t>();
    if (count > tokens) {
        throw std::invalid_argument("checkpoint image vision item count is out of range");
    }
    std::vector<VisionItem> items(count);
    for (VisionItem& item : items) {
        const auto modality = reader.pod<std::uint8_t>();
        if (modality != static_cast<std::uint8_t>(PromptModality::Image) &&
            modality != static_cast<std::uint8_t>(PromptModality::Video)) {
            throw std::invalid_argument("checkpoint image vision item modality is invalid");
        }
        item.modality      = static_cast<PromptModality>(modality);
        item.grid.temporal = reader.pod<std::int32_t>();
        item.grid.height   = reader.pod<std::int32_t>();
        item.grid.width    = reader.pod<std::int32_t>();
        item.patch_begin   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        item.patch_count   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        reader.bytes(item.content_digest.data(), item.content_digest.size());
        item.timestamps  = reader.vector<double>(tokens, "vision timestamp");
        const auto spans = reader.pod<std::uint32_t>();
        if (spans > tokens) {
            throw std::invalid_argument("checkpoint image vision span count is out of range");
        }
        item.token_spans.resize(spans);
        for (TokenSpan& span : item.token_spans) {
            span.begin = static_cast<std::size_t>(reader.pod<std::uint64_t>());
            span.count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        }
    }
    return items;
}

struct HistoryRecord {
    std::uint32_t text_frontier = 0;
    std::vector<std::uint32_t> text_pages; // indices into the Text page table
    bool has_backend               = false;
    std::uint32_t backend_frontier = 0;
    std::vector<std::uint32_t> backend_pages;
};

struct PointRecord {
    runtime::CheckpointRole role   = runtime::CheckpointRole::Continuation;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    bool tail_hidden_valid         = false;
    std::uint32_t history          = 0;
    std::uint32_t state            = 0;
    std::uint32_t identity         = 0;
    PrefixShortlistKey key;
};

// A fully parsed and validated image. Payload spans borrow the caller's bytes.
struct ParsedImage {
    std::optional<PreparedSessionKey> session;
    std::vector<std::shared_ptr<PreparedCaptureBacking>> identities;
    std::vector<std::uint32_t> text_columns;
    std::vector<std::uint32_t> backend_columns;
    std::vector<HistoryRecord> histories;
    std::vector<PointRecord> points;
    std::uint32_t state_count = 0;
    std::span<const std::uint8_t> text_payload;
    std::span<const std::uint8_t> backend_payload;
    std::span<const std::uint8_t> state_payload;
};

std::size_t aligned_host_bytes(std::size_t bytes) {
    const std::size_t alignment = HostContextArena::alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - alignment) {
        throw std::overflow_error("checkpoint image Host size overflows");
    }
    return (bytes + alignment - 1U) / alignment * alignment;
}

} // namespace

namespace {

ImageConfig program_config(const ProgramImpl& program) {
    ImageConfig config;
    config.kv_storage          = static_cast<std::uint32_t>(program.kv_storage);
    config.speculative_backend = static_cast<std::uint32_t>(program.speculative_backend);
    config.proposal_head       = static_cast<std::uint32_t>(program.proposal_head);
    config.draft_window        = program.draft_window;
    config.page_tokens         = static_cast<std::uint32_t>(kPagedKVPageSize);
    config.state_image_bytes   = program.state_images->host_layout().image_bytes;
    const HostKVPageLayout text =
        plan_host_kv_page_layout(program.text_kv_pages->physical_pool().geometry());
    config.text_plane_count = static_cast<std::uint32_t>(text.planes.size());
    config.text_page_stride = text.page_stride;
    if (program.backend_kv_pages) {
        const HostKVPageLayout backend =
            plan_host_kv_page_layout(program.backend_kv_pages->physical_pool().geometry());
        config.backend_plane_count = static_cast<std::uint32_t>(backend.planes.size());
        config.backend_page_stride = backend.page_stride;
    }
    return config;
}

ParsedImage parse_image(const ProgramImpl& program, std::span<const std::uint8_t> image,
                        std::string_view binding) {
    if (image.size() < sizeof(kImageMagic) + sizeof(std::uint64_t)) {
        throw std::invalid_argument("checkpoint image is truncated");
    }
    const auto body              = image.first(image.size() - sizeof(std::uint64_t));
    std::uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, image.data() + body.size(), sizeof(stored_checksum));
    ImageReader reader(body);
    char magic[sizeof(kImageMagic)] = {};
    reader.bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kImageMagic, sizeof(magic)) != 0) {
        throw std::invalid_argument("bytes are not a checkpoint image");
    }
    if (reader.pod<std::uint32_t>() != kImageVersion) {
        throw std::invalid_argument("checkpoint image version is unsupported");
    }
    const auto binding_bytes = reader.pod<std::uint32_t>();
    if (binding_bytes > kMaximumModelBindingBytes) {
        throw std::invalid_argument("checkpoint image model binding is too long");
    }
    std::string stored_binding(binding_bytes, '\0');
    reader.bytes(stored_binding.data(), binding_bytes);
    if (stored_binding != binding) {
        throw std::invalid_argument("checkpoint image was written for a different model");
    }
    if (read_config(reader) != program_config(program)) {
        throw std::invalid_argument(
            "checkpoint image KV, speculative or state configuration differs from this Program");
    }
    // Verified before anything is allocated or copied.
    if (image_checksum(body) != stored_checksum) {
        throw std::invalid_argument("checkpoint image is corrupt: checksum mismatch");
    }

    ParsedImage parsed;
    if (reader.pod<std::uint8_t>() != 0) {
        PreparedSessionKey key;
        key.size = reader.pod<std::uint16_t>();
        if (key.size == 0 || key.size > key.bytes.size()) {
            throw std::invalid_argument("checkpoint image session key is out of range");
        }
        reader.bytes(key.bytes.data(), key.size);
        parsed.session = key;
    }
    const auto identity_count = reader.pod<std::uint32_t>();
    const auto history_count  = reader.pod<std::uint32_t>();
    const auto text_pages     = reader.pod<std::uint32_t>();
    const auto backend_pages  = reader.pod<std::uint32_t>();
    parsed.state_count        = reader.pod<std::uint32_t>();
    const auto point_count    = reader.pod<std::uint32_t>();
    if (point_count == 0 || point_count > kMaximumImagePoints || identity_count > point_count ||
        history_count > point_count || parsed.state_count > point_count ||
        text_pages > program.text_kv_pages->capacity() ||
        backend_pages > (program.backend_kv_pages ? program.backend_kv_pages->capacity() : 0U)) {
        throw std::invalid_argument("checkpoint image table sizes are out of range");
    }

    const std::uint32_t capacity   = program.capacity;
    const auto vocabulary =
        execution::dimension(program.parameters.model.resources().public_token_count);
    for (std::uint32_t index = 0; index < identity_count; ++index) {
        auto backing    = std::make_shared<PreparedCaptureBacking>();
        backing->ledger = reader.vector<TokenId>(capacity, "ledger");
        for (const TokenId id : backing->ledger) {
            if (id < 0 || id >= vocabulary) {
                throw std::invalid_argument("checkpoint image ledger token is out of domain");
            }
        }
        const std::size_t tokens = backing->ledger.size();
        auto token_types         = reader.vector<std::uint8_t>(tokens, "token type");
        std::array<std::vector<std::int32_t>, 3> positions;
        for (auto& axis : positions) { axis = reader.vector<std::int32_t>(tokens, "position"); }
        auto vision  = read_vision_items(reader, tokens);
        auto rewrite = reader.vector<std::uint32_t>(tokens, "rewrite frontier");
        if (token_types.size() != tokens) {
            throw std::invalid_argument("checkpoint image identity does not match its ledger");
        }
        backing->prefix_identity.restore(std::move(token_types), std::move(positions),
                                         std::move(vision), std::move(rewrite));
        auto digests = reader.vector<std::array<std::uint64_t, 2>>(capacity + 1U, "digest");
        if (digests.size() != tokens + 1U) {
            throw std::invalid_argument("checkpoint image digests do not match their ledger");
        }
        backing->digests.restore(std::move(digests));
        parsed.identities.push_back(std::move(backing));
    }

    const auto read_columns = [&](std::uint32_t count, std::vector<std::uint32_t>& out) {
        out = std::vector<std::uint32_t>(count);
        reader.bytes(out.data(), out.size() * sizeof(std::uint32_t));
        for (const auto columns : out) {
            if (columns == 0 || columns > static_cast<std::uint32_t>(kPagedKVPageSize)) {
                throw std::invalid_argument("checkpoint image page coverage is out of range");
            }
        }
    };
    read_columns(text_pages, parsed.text_columns);
    read_columns(backend_pages, parsed.backend_columns);

    const auto read_pages = [&](std::uint32_t frontier, std::uint32_t table,
                                std::span<const std::uint32_t> columns) {
        const auto count = reader.pod<std::uint32_t>();
        if (count != kv_pages_for_frontier(frontier) || frontier > capacity) {
            throw std::invalid_argument("checkpoint image history coverage is inconsistent");
        }
        std::vector<std::uint32_t> pages(count);
        reader.bytes(pages.data(), pages.size() * sizeof(std::uint32_t));
        const auto page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto page   = pages[index];
            const auto needed = std::min(page_size, frontier - index * page_size);
            if (page >= table || columns[page] < needed ||
                (index + 1U < count && columns[page] != page_size)) {
                throw std::invalid_argument("checkpoint image history page is out of range");
            }
        }
        return pages;
    };
    for (std::uint32_t index = 0; index < history_count; ++index) {
        HistoryRecord history;
        history.text_frontier = reader.pod<std::uint32_t>();
        if (history.text_frontier == 0) {
            throw std::invalid_argument("checkpoint image history is empty");
        }
        history.text_pages  = read_pages(history.text_frontier, text_pages, parsed.text_columns);
        history.has_backend = reader.pod<std::uint8_t>() != 0;
        if (history.has_backend != static_cast<bool>(program.backend_kv_pages)) {
            throw std::invalid_argument("checkpoint image backend history does not match");
        }
        if (history.has_backend) {
            history.backend_frontier = reader.pod<std::uint32_t>();
            history.backend_pages =
                read_pages(history.backend_frontier, backend_pages, parsed.backend_columns);
        }
        parsed.histories.push_back(std::move(history));
    }

    for (std::uint32_t index = 0; index < point_count; ++index) {
        PointRecord point;
        const auto role = reader.pod<std::uint8_t>();
        if (role != static_cast<std::uint8_t>(runtime::CheckpointRole::Continuation) &&
            role != static_cast<std::uint8_t>(runtime::CheckpointRole::InputReplay) &&
            role != static_cast<std::uint8_t>(runtime::CheckpointRole::LongAnchor)) {
            throw std::invalid_argument("checkpoint image point role is not a private point");
        }
        point.role                 = static_cast<runtime::CheckpointRole>(role);
        point.frontier             = reader.pod<std::uint32_t>();
        point.backend_frontier     = reader.pod<std::uint32_t>();
        point.rope_delta           = reader.pod<std::int32_t>();
        point.tail_hidden_valid    = reader.pod<std::uint8_t>() != 0;
        point.history              = reader.pod<std::uint32_t>();
        point.state                = reader.pod<std::uint32_t>();
        point.identity             = reader.pod<std::uint32_t>();
        point.key.digests[0]       = reader.pod<std::uint64_t>();
        point.key.digests[1]       = reader.pod<std::uint64_t>();
        point.key.identity_tag     = reader.pod<std::uint32_t>();
        point.key.frontier         = point.frontier;
        if (point.history >= history_count || point.state >= parsed.state_count ||
            point.identity >= identity_count || point.frontier == 0) {
            throw std::invalid_argument("checkpoint image point references are out of range");
        }
        const auto& history  = parsed.histories[point.history];
        const auto& identity = *parsed.identities[point.identity];
        if (point.frontier > history.text_frontier ||
            point.frontier > identity.prefix_identity.size() ||
            (history.has_backend && point.backend_frontier > history.backend_frontier) ||
            (!history.has_backend && point.backend_frontier != 0) ||
            identity.digests.at(point.frontier) != point.key.digests) {
            throw std::invalid_argument("checkpoint image point is inconsistent");
        }
        parsed.points.push_back(point);
    }

    const auto text_bytes = static_cast<std::size_t>(text_pages) *
                            static_cast<std::size_t>(program_config(program).text_page_stride);
    const auto backend_bytes =
        static_cast<std::size_t>(backend_pages) *
        static_cast<std::size_t>(program_config(program).backend_page_stride);
    const auto state_bytes = static_cast<std::size_t>(parsed.state_count) *
                             program.state_images->host_layout().image_bytes;
    parsed.text_payload    = reader.payload(text_bytes);
    parsed.backend_payload = reader.payload(backend_bytes);
    parsed.state_payload   = reader.payload(state_bytes);
    if (reader.remaining() != 0) {
        throw std::invalid_argument("checkpoint image has trailing bytes");
    }
    return parsed;
}

} // namespace

CheckpointImage ProgramImpl::export_checkpoints(std::span<const CheckpointHandle> handles,
                                                const std::optional<PreparedSessionKey>& session,
                                                std::string_view binding) const {
    if (context_transaction_ || pending_transaction_) {
        throw std::logic_error("checkpoints cannot be exported during a context transaction");
    }
    if (handles.empty() || handles.size() > kMaximumImagePoints) {
        throw std::invalid_argument("checkpoint export needs between one and 64 points");
    }
    if (binding.size() > kMaximumModelBindingBytes) {
        throw std::invalid_argument("checkpoint image model binding is too long");
    }
    // Deepest first: the first key is the image's endpoint.
    std::vector<CheckpointHandle> points;
    for (const auto handle : handles) {
        if (std::find(points.begin(), points.end(), handle) == points.end()) {
            points.push_back(handle);
        }
    }
    for (const auto handle : points) {
        const auto& record = checkpoint(handle);
        if (!record.identity || !record.kv ||
            record.role == runtime::CheckpointRole::SharedPrefix) {
            throw std::invalid_argument("checkpoint is not an exportable private recovery point");
        }
    }
    std::stable_sort(points.begin(), points.end(), [&](auto a, auto b) {
        return checkpoint(a).frontier > checkpoint(b).frontier;
    });

    const ImageConfig config                 = program_config(*this);
    const StateImageHostLayout& state_layout = state_images->host_layout();

    // Deduplicated tables, in first-use order.
    std::vector<const PreparedCaptureBacking*> identities;
    std::vector<const KVHistory*> history_order;
    std::unordered_map<const KVHistory*, std::pair<std::uint32_t, std::uint32_t>> coverage;
    std::vector<StateImageHandle> states;
    const auto index_of = [](auto& table, const auto& value) {
        const auto found = std::find(table.begin(), table.end(), value);
        if (found != table.end()) { return static_cast<std::uint32_t>(found - table.begin()); }
        table.push_back(value);
        return static_cast<std::uint32_t>(table.size() - 1U);
    };
    std::vector<PointRecord> records;
    for (const auto handle : points) {
        const auto& record = checkpoint(handle);
        PointRecord point;
        point.role              = record.role;
        point.frontier          = record.frontier;
        point.backend_frontier  = record.kv->backend ? record.backend_frontier : 0U;
        point.rope_delta        = record.rope_delta;
        point.tail_hidden_valid = record.tail_hidden_valid;
        point.history           = index_of(history_order, record.kv.get());
        point.state             = index_of(states, record.state);
        point.identity          = index_of(identities, record.identity.get());
        point.key               = record.key;
        records.push_back(point);
        auto& covered  = coverage[record.kv.get()];
        covered.first  = std::max(covered.first, record.frontier);
        covered.second = std::max({covered.second, point.backend_frontier,
                                   record.kv->backend
                                       ? backend_frontier_at(speculative_backend, record.frontier)
                                       : 0U});
    }

    // Pages: each logical page once, with the columns its deepest use needs.
    struct PageTable {
        LogicalKVPageStore* pages = nullptr;
        std::vector<LogicalKVPageHandle> handles;
        std::vector<std::uint32_t> columns;
        std::unordered_map<std::uint32_t, std::uint32_t> index; // descriptor -> table index
    };
    PageTable text_table{text_kv_pages.get()};
    PageTable backend_table{backend_kv_pages.get()};
    std::vector<HistoryRecord> histories;
    const auto page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
    const auto collect   = [&](PageTable& table, const KVAddressSpaceStore& addresses,
                             KVAddressSpaceHandle address, std::uint32_t& frontier) {
        frontier = std::min(frontier, addresses.committed_frontier(address));
        std::vector<std::uint32_t> out;
        const auto count = kv_pages_for_frontier(frontier);
        if (count > addresses.mapped_pages(address)) {
            throw std::logic_error("exported history coverage exceeds its mapped pages");
        }
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto page   = addresses.logical_page(address, index);
            const auto needed = std::min(page_size, frontier - index * page_size);
            if (table.pages->committed_columns(page) < needed) {
                throw std::logic_error("exported KV page does not cover its checkpoint");
            }
            const auto descriptor = table.pages->descriptor_index(page);
            auto found            = table.index.find(descriptor);
            if (found == table.index.end()) {
                found = table.index
                            .emplace(descriptor, static_cast<std::uint32_t>(table.handles.size()))
                            .first;
                table.handles.push_back(page);
                table.columns.push_back(needed);
            } else {
                table.columns[found->second] = std::max(table.columns[found->second], needed);
            }
            out.push_back(found->second);
        }
        return out;
    };
    for (const KVHistory* history : history_order) {
        HistoryRecord out;
        out.text_frontier = coverage[history].first;
        out.text_pages = collect(text_table, *text_kv_addresses, history->text, out.text_frontier);
        out.has_backend = history->backend.has_value();
        if (out.has_backend) {
            out.backend_frontier = coverage[history].second;
            out.backend_pages    = collect(backend_table, *backend_kv_addresses, *history->backend,
                                           out.backend_frontier);
        }
        histories.push_back(std::move(out));
    }
    // A shared page's partial tail can be needed in full by another history.
    for (auto* table : {&text_table, &backend_table}) {
        for (std::size_t index = 0; index < table->handles.size(); ++index) {
            table->columns[index] = std::min(table->columns[index],
                                             table->pages->committed_columns(table->handles[index]));
        }
    }

    CheckpointImage image;
    ImageWriter writer(image.bytes);
    writer.bytes(kImageMagic, sizeof(kImageMagic));
    writer.pod(kImageVersion);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(binding.size()));
    writer.bytes(binding.data(), binding.size());
    write_config(writer, config);
    writer.pod<std::uint8_t>(session ? 1 : 0);
    if (session) {
        writer.pod<std::uint16_t>(session->size);
        writer.bytes(session->bytes.data(), session->size);
    }
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(identities.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(histories.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(text_table.handles.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(backend_table.handles.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(states.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(records.size()));
    for (const PreparedCaptureBacking* identity : identities) {
        writer.vector(identity->ledger);
        writer.vector(identity->prefix_identity.token_types());
        for (std::size_t axis = 0; axis < 3; ++axis) {
            writer.vector(identity->prefix_identity.position_axis(axis));
        }
        write_vision_items(writer, identity->prefix_identity.vision_items());
        writer.vector(identity->prefix_identity.rewrite_execution_frontiers());
        writer.vector(identity->digests.image());
    }
    writer.bytes(text_table.columns.data(), text_table.columns.size() * sizeof(std::uint32_t));
    writer.bytes(backend_table.columns.data(),
                 backend_table.columns.size() * sizeof(std::uint32_t));
    for (const HistoryRecord& history : histories) {
        writer.pod(history.text_frontier);
        writer.pod<std::uint32_t>(static_cast<std::uint32_t>(history.text_pages.size()));
        writer.bytes(history.text_pages.data(), history.text_pages.size() * sizeof(std::uint32_t));
        writer.pod<std::uint8_t>(history.has_backend ? 1 : 0);
        if (history.has_backend) {
            writer.pod(history.backend_frontier);
            writer.pod<std::uint32_t>(static_cast<std::uint32_t>(history.backend_pages.size()));
            writer.bytes(history.backend_pages.data(),
                         history.backend_pages.size() * sizeof(std::uint32_t));
        }
    }
    for (const PointRecord& point : records) {
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(point.role));
        writer.pod(point.frontier);
        writer.pod(point.backend_frontier);
        writer.pod(point.rope_delta);
        writer.pod<std::uint8_t>(point.tail_hidden_valid ? 1 : 0);
        writer.pod(point.history);
        writer.pod(point.state);
        writer.pod(point.identity);
        writer.pod(point.key.digests[0]);
        writer.pod(point.key.digests[1]);
        writer.pod(point.key.identity_tag);
    }
    const auto text_offset =
        writer.reserve(text_table.handles.size() * static_cast<std::size_t>(config.text_page_stride));
    const auto backend_offset = writer.reserve(
        backend_table.handles.size() * static_cast<std::size_t>(config.backend_page_stride));
    const auto state_offset   = writer.reserve(states.size() * state_layout.image_bytes);
    const auto checksum_offset = writer.reserve(sizeof(std::uint64_t));
    if (!text_table.handles.empty()) {
        image.regions.push_back(
            {text_offset, text_table.handles.size() * config.text_page_stride});
    }
    if (!backend_table.handles.empty()) {
        image.regions.push_back(
            {backend_offset, backend_table.handles.size() * config.backend_page_stride});
    }
    image.regions.push_back({state_offset, states.size() * state_layout.image_bytes});

    // The pages and images are immutable and no transaction is open, but the unit that last wrote
    // them may still be in flight on the compute stream, which also orders these copies behind it.
    std::uint8_t* const base  = image.bytes.data();
    const RankStreams streams = RankStreams::compute(device);
    bool device_copies        = false;
    const auto copy_pages     = [&](PageTable& table, std::size_t offset, std::size_t stride) {
        if (table.handles.empty()) { return; }
        const DeviceKVPagePool& pool   = table.pages->physical_pool();
        const HostKVPageLayout layout  = plan_host_kv_page_layout(pool.geometry());
        std::vector<DeviceKVPageHandle> run;
        std::size_t run_begin = 0;
        const auto flush      = [&] {
            if (run.empty()) { return; }
            (void)pool.copy_to_host(run, reinterpret_cast<std::byte*>(base + offset +
                                                                       run_begin * stride),
                                    layout, streams);
            device_copies = true;
            run.clear();
        };
        for (std::size_t index = 0; index < table.handles.size(); ++index) {
            const auto page = table.handles[index];
            std::uint8_t* destination = base + offset + index * stride;
            if (table.pages->host_replica_current(page)) {
                flush();
                const auto replica = table.pages->host_replica(page);
                const auto view =
                    host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
                std::memcpy(destination, view.data(), stride);
                continue;
            }
            if (!table.pages->device_resident(page)) {
                throw std::logic_error("exported KV page has no current replica");
            }
            if (run.empty()) { run_begin = index; }
            run.push_back(table.pages->physical(page));
        }
        flush();
    };
    copy_pages(text_table, text_offset, config.text_page_stride);
    copy_pages(backend_table, backend_offset, config.backend_page_stride);
    for (std::size_t index = 0; index < states.size(); ++index) {
        std::uint8_t* const destination = base + state_offset + index * state_layout.image_bytes;
        if (state_store->host_resident(states[index])) {
            std::memcpy(destination, state_store->host_view(states[index]).data,
                        state_layout.image_bytes);
        } else {
            state_images->copy_to_host(
                state_store->physical_slot(states[index]),
                qwen3_5::HostStateImageView{.data   = reinterpret_cast<std::byte*>(destination),
                                            .layout = &state_layout},
                streams);
            device_copies = true;
        }
    }
    if (device_copies) {
        for (std::size_t rank = 0; rank < streams.size(); ++rank) {
            CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
        }
    }
    const std::uint64_t checksum =
        image_checksum(std::span<const std::uint8_t>(base, checksum_offset));
    std::memcpy(base + checksum_offset, &checksum, sizeof(checksum));

    for (const PointRecord& point : records) { image.keys.push_back(point.key); }
    image.tokens = records.front().frontier;
    const auto& deepest = *identities[records.front().identity];
    const auto& digests = deepest.digests.image();
    image.prefix_digests.assign(digests.begin(),
                                digests.begin() + static_cast<std::ptrdiff_t>(image.tokens) + 1);
    return image;
}

std::size_t ProgramImpl::checkpoint_image_host_bytes(std::span<const std::uint8_t> image,
                                                     std::string_view binding) const {
    const ParsedImage parsed = parse_image(*this, image, binding);
    std::size_t bytes        = static_cast<std::size_t>(parsed.state_count) *
                        aligned_host_bytes(state_images->host_layout().image_bytes);
    if (!parsed.text_payload.empty()) { bytes += aligned_host_bytes(parsed.text_payload.size()); }
    if (!parsed.backend_payload.empty()) {
        bytes += aligned_host_bytes(parsed.backend_payload.size());
    }
    return bytes;
}

std::optional<ImportedCheckpoints>
ProgramImpl::import_checkpoints(std::span<const std::uint8_t> image, std::string_view binding) {
    if (context_transaction_ || pending_transaction_) {
        throw std::logic_error("checkpoints cannot be imported during a context transaction");
    }
    const ParsedImage parsed = parse_image(*this, image, binding);
    if (!host_context_arena || !host_kv_extents || !host_state_images) { return std::nullopt; }

    // Everything created so far, released in reverse on any failure. Histories own their
    // addresses: dropping one releases its page references, after which the extents' unreferenced
    // pages are freed.
    struct Rollback {
        ProgramImpl& program;
        std::vector<StateImageHandle> states;
        std::vector<std::pair<LogicalKVPageStore*, std::vector<LogicalKVPageHandle>>> orphan_pages;
        std::vector<HostKVExtentCapability> extents;
        std::vector<std::shared_ptr<KVHistory>> histories;
        std::vector<CheckpointHandle> reserved;
        bool armed = true;

        ~Rollback() {
            if (!armed) { return; }
            for (const auto handle : reserved) {
                program.checkpoints[handle.index].reserved = false;
            }
            histories.clear();
            for (const auto capability : extents) {
                (void)program.host_kv_extents->release(capability);
            }
            (void)program.host_kv_extents->release_unreferenced();
            for (const auto state : states) { (void)program.state_store->release(state); }
        }
    } rollback{*this};

    const std::size_t state_bytes = state_images->host_layout().image_bytes;
    for (std::uint32_t index = 0; index < parsed.state_count; ++index) {
        const auto bytes =
            std::as_bytes(parsed.state_payload.subspan(index * state_bytes, state_bytes));
        const auto state = state_store->import_host(bytes);
        if (!state) { return std::nullopt; }
        rollback.states.push_back(*state);
    }

    // Pages, in as few extents as the free Host space allows.
    const auto import_table = [&](LogicalKVPageStore& pages, std::span<const std::uint32_t> columns,
                                  std::span<const std::uint8_t> payload,
                                  std::vector<LogicalKVPageHandle>& out) {
        const std::size_t stride = host_kv_extents->page_layout(pages).page_stride;
        std::size_t offset       = 0;
        while (offset < columns.size()) {
            std::size_t count = columns.size() - offset;
            std::optional<HostKVExtentStore::ImportedExtent> extent;
            while (count) {
                extent = host_kv_extents->import_pages(
                    pages, columns.subspan(offset, count),
                    std::as_bytes(payload.subspan(offset * stride, count * stride)));
                if (extent) { break; }
                count /= 2;
            }
            if (!extent) { return false; }
            rollback.extents.push_back(extent->capability);
            out.insert(out.end(), extent->pages.begin(), extent->pages.end());
            offset += count;
        }
        return true;
    };
    std::vector<LogicalKVPageHandle> text_pages, backend_pages;
    if (!import_table(*text_kv_pages, parsed.text_columns, parsed.text_payload, text_pages)) {
        return std::nullopt;
    }
    if (backend_kv_pages && !import_table(*backend_kv_pages, parsed.backend_columns,
                                          parsed.backend_payload, backend_pages)) {
        return std::nullopt;
    }

    for (const HistoryRecord& record : parsed.histories) {
        const auto text = text_kv_addresses->create_inactive();
        if (!text) { return std::nullopt; }
        auto history   = std::make_shared<KVHistory>();
        history->owner = this;
        history->text  = *text;
        if (record.has_backend) {
            const auto backend = backend_kv_addresses->create_inactive();
            if (!backend) {
                history->owner = nullptr;
                (void)text_kv_addresses->release(*text);
                return std::nullopt;
            }
            history->backend = *backend;
        }
        rollback.histories.push_back(history);
        std::vector<LogicalKVPageHandle> members;
        for (const auto page : record.text_pages) { members.push_back(text_pages[page]); }
        text_kv_addresses->adopt_inactive(history->text, members, record.text_frontier);
        if (record.has_backend) {
            members.clear();
            for (const auto page : record.backend_pages) { members.push_back(backend_pages[page]); }
            backend_kv_addresses->adopt_inactive(*history->backend, members,
                                                 record.backend_frontier);
        }
    }

    for (std::size_t index = 0; index < parsed.points.size(); ++index) {
        const auto handle = reserve_checkpoint();
        if (!handle) { return std::nullopt; }
        rollback.reserved.push_back(*handle);
    }

    // Nothing below can fail: publish the records.
    ImportedCheckpoints imported;
    imported.session = parsed.session;
    for (std::size_t index = 0; index < parsed.points.size(); ++index) {
        const PointRecord& point = parsed.points[index];
        const auto handle        = rollback.reserved[index];
        const auto state         = rollback.states[point.state];
        state_store->retain_checkpoint_reference(state);
        checkpoints[handle.index].value =
            CheckpointState{.kv                = rollback.histories[point.history],
                            .role              = point.role,
                            .state             = state,
                            .identity          = parsed.identities[point.identity],
                            .key               = point.key,
                            .frontier          = point.frontier,
                            .backend_frontier  = point.backend_frontier,
                            .rope_delta        = point.rope_delta,
                            .tail_hidden_valid = point.tail_hidden_valid};
        checkpoints[handle.index].reserved = false;
        imported.points.push_back(handle);
    }
    rollback.armed = false;
    for (const auto& history : rollback.histories) { refresh_history_requirements(history); }
    // An image naming a StateImage no point uses would leave it unowned.
    for (const auto state : rollback.states) {
        if (state_store->checkpoint_references(state) == 0) { (void)state_store->release(state); }
    }
    return imported;
}

} // namespace ninfer::models::qwen3_5::detail
