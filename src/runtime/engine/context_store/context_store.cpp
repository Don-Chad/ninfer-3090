#include "runtime/engine/context_store/context_store.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace ninfer::runtime {
namespace {

namespace fs = std::filesystem;

constexpr char kManifestMagic[8]     = {'N', 'C', 'T', 'X', 'S', 'T', 'O', '1'};
constexpr std::uint32_t kManifestVersion = 1;
// A manifest names at most this many chunks and checkpoints; larger counts mean a corrupt file, not
// a big session (a 1M-token session at 32 MiB chunks is a few thousand chunks).
constexpr std::uint32_t kMaximumChunksPerImage = 1U << 20U;
constexpr std::uint32_t kMaximumCheckpoints    = 1024;
constexpr std::uint32_t kMaximumSegments       = 1U << 16U;

// xxHash64 (Yann Collet, BSD-2-Clause), little-endian hosts. Two independently seeded passes give
// the 128-bit chunk name. This is a content address for trusted local files, not a security
// boundary: it needs to make an accidental collision between two different chunks negligible.
constexpr std::uint64_t kPrime1 = 11400714785074694791ULL;
constexpr std::uint64_t kPrime2 = 14029467366897019727ULL;
constexpr std::uint64_t kPrime3 = 1609587929392839161ULL;
constexpr std::uint64_t kPrime4 = 9650029242287828579ULL;
constexpr std::uint64_t kPrime5 = 2870177450012600261ULL;

constexpr std::uint64_t rotate_left(std::uint64_t value, int bits) noexcept {
    return (value << bits) | (value >> (64 - bits));
}

std::uint64_t read64(const std::uint8_t* p) noexcept {
    std::uint64_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

std::uint32_t read32(const std::uint8_t* p) noexcept {
    std::uint32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

constexpr std::uint64_t xxh_round(std::uint64_t acc, std::uint64_t input) noexcept {
    acc += input * kPrime2;
    acc = rotate_left(acc, 31);
    return acc * kPrime1;
}

constexpr std::uint64_t xxh_merge(std::uint64_t acc, std::uint64_t value) noexcept {
    acc ^= xxh_round(0, value);
    return acc * kPrime1 + kPrime4;
}

std::uint64_t xxh64(const std::uint8_t* p, std::size_t length, std::uint64_t seed) noexcept {
    const std::uint8_t* const end = p + length;
    std::uint64_t h;
    if (length >= 32) {
        const std::uint8_t* const limit = end - 32;
        std::uint64_t v1                = seed + kPrime1 + kPrime2;
        std::uint64_t v2                = seed + kPrime2;
        std::uint64_t v3                = seed;
        std::uint64_t v4                = seed - kPrime1;
        do {
            v1 = xxh_round(v1, read64(p));
            v2 = xxh_round(v2, read64(p + 8));
            v3 = xxh_round(v3, read64(p + 16));
            v4 = xxh_round(v4, read64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotate_left(v1, 1) + rotate_left(v2, 7) + rotate_left(v3, 12) + rotate_left(v4, 18);
        h = xxh_merge(h, v1);
        h = xxh_merge(h, v2);
        h = xxh_merge(h, v3);
        h = xxh_merge(h, v4);
    } else {
        h = seed + kPrime5;
    }
    h += static_cast<std::uint64_t>(length);
    while (p + 8 <= end) {
        h ^= xxh_round(0, read64(p));
        h = rotate_left(h, 27) * kPrime1 + kPrime4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<std::uint64_t>(read32(p)) * kPrime1;
        h = rotate_left(h, 23) * kPrime2 + kPrime3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<std::uint64_t>(*p) * kPrime5;
        h = rotate_left(h, 11) * kPrime1;
        ++p;
    }
    h ^= h >> 33;
    h *= kPrime2;
    h ^= h >> 29;
    h *= kPrime3;
    h ^= h >> 32;
    return h;
}

std::string hex_key(const std::array<std::uint64_t, 2>& hash) {
    char text[33];
    std::snprintf(text, sizeof(text), "%016llx%016llx", static_cast<unsigned long long>(hash[0]),
                  static_cast<unsigned long long>(hash[1]));
    return text;
}

bool valid_id(const std::string& id) {
    if (id.empty() || id.size() > 64) { return false; }
    return std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::int64_t system_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

class Writer {
public:
    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }
    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out.insert(out.end(), begin, begin + count);
    }
    void string(const std::string& text) {
        pod<std::uint32_t>(static_cast<std::uint32_t>(text.size()));
        bytes(text.data(), text.size());
    }
    std::vector<std::uint8_t> out;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) { throw std::runtime_error("manifest is truncated"); }
        std::memcpy(out, data_.data() + cursor_, count);
        cursor_ += count;
    }
    template <class T>
    T pod() {
        T value{};
        bytes(&value, sizeof(T));
        return value;
    }
    std::string string(std::size_t maximum) {
        const auto length = pod<std::uint32_t>();
        if (length > maximum) { throw std::runtime_error("manifest string is out of range"); }
        std::string text(length, '\0');
        bytes(text.data(), length);
        return text;
    }
    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

std::vector<std::uint8_t> read_whole_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    const std::streamsize size = file.tellg();
    if (size < 0) { throw std::runtime_error("cannot size " + path.string()); }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    file.seekg(0);
    if (size != 0 && !file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return data;
}

void remove_quietly(const fs::path& path) noexcept {
    std::error_code ignored;
    fs::remove(path, ignored);
}

} // namespace

std::array<std::uint64_t, 2> ContextStore::hash(std::span<const std::uint8_t> bytes) {
    return {xxh64(bytes.data(), bytes.size(), 0), xxh64(bytes.data(), bytes.size(), 0x9e3779b97f4a7c15ULL)};
}

ContextStore::ContextStore(Options options) : options_(std::move(options)) {
    if (options_.directory.empty()) {
        throw std::invalid_argument("context store directory is empty");
    }
    if (options_.chunk_bytes == 0 ||
        options_.chunk_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("context store chunk size is out of range");
    }
    if (!options_.clock) { options_.clock = system_now_ms; }
    std::error_code error;
    for (const char* sub : {"chunks", "manifests", "tmp"}) {
        fs::create_directories(options_.directory / sub, error);
        if (error) {
            throw std::runtime_error("cannot create context store directory " +
                                     (options_.directory / sub).string() + ": " + error.message());
        }
    }
    {
        std::scoped_lock io(io_mutex_);
        std::scoped_lock lock(mutex_);
        scan();
        maintain_locked(nullptr);
    }
    if (options_.remote) {
        remote_thread_ = std::thread([this] { remote_loop(); });
        enqueue_remote({RemoteTask::Kind::Refresh, std::string()});
    }
}

ContextStore::~ContextStore() {
    {
        std::scoped_lock lock(remote_mutex_);
        remote_stopping_ = true;
    }
    remote_cv_.notify_all();
    if (remote_thread_.joinable()) { remote_thread_.join(); }
}

std::int64_t ContextStore::now_ms() const { return options_.clock(); }

fs::path ContextStore::manifest_path(const std::string& id) const {
    return options_.directory / "manifests" / (id + ".manifest");
}

fs::path ContextStore::used_path(const std::string& id) const {
    return options_.directory / "manifests" / (id + ".used");
}

fs::path ContextStore::chunk_path(const std::array<std::uint64_t, 2>& hash) const {
    const std::string key = hex_key(hash);
    return options_.directory / "chunks" / key.substr(0, 2) / (key + ".chunk");
}

void ContextStore::write_atomically(const fs::path& path, std::span<const std::uint8_t> bytes) const {
    static std::atomic<std::uint64_t> counter{0};
    std::ostringstream name;
    name << std::this_thread::get_id() << '.' << counter.fetch_add(1) << ".tmp";
    const fs::path staging = options_.directory / "tmp" / name.str();
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    {
        std::ofstream file(staging, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        file.flush();
        if (!file.good()) {
            file.close();
            remove_quietly(staging);
            throw std::runtime_error("cannot write " + staging.string());
        }
    }
    fs::rename(staging, path, error);
    if (error) {
        remove_quietly(staging);
        throw std::runtime_error("cannot publish " + path.string() + ": " + error.message());
    }
}

void ContextStore::add_references(const Entry& entry) {
    for (const ChunkRef& chunk : entry.chunks) {
        ChunkUse& use = chunks_[hex_key(chunk.hash)];
        if (use.references++ == 0) {
            use.length = chunk.length;
            // An image a remote holds is registered before its chunks are fetched.
            std::error_code error;
            const auto size = fs::file_size(chunk_path(chunk.hash), error);
            use.present     = !error && size == chunk.length;
            if (use.present) { chunk_total_bytes_ += chunk.length; }
        }
    }
}

void ContextStore::release_references(const Entry& entry) {
    for (const ChunkRef& chunk : entry.chunks) {
        const auto found = chunks_.find(hex_key(chunk.hash));
        if (found == chunks_.end()) { continue; }
        if (--found->second.references == 0) {
            if (found->second.present) { chunk_total_bytes_ -= found->second.length; }
            remove_quietly(chunk_path(chunk.hash));
            chunks_.erase(found);
        }
    }
}

void ContextStore::remove_entry_locked(const std::string& id, bool corrupt) {
    const auto found = entries_.find(id);
    if (found != entries_.end()) {
        release_references(found->second);
        manifest_total_bytes_ -= found->second.manifest_bytes;
        entries_.erase(found);
        if (corrupt) { ++stats_.corrupt_removed; }
    }
    remove_quietly(manifest_path(id));
    remove_quietly(used_path(id));
}

void ContextStore::write_used(const std::string& id, std::int64_t last_used_ms) const {
    const std::string text = std::to_string(last_used_ms);
    try {
        write_atomically(used_path(id),
                         std::span<const std::uint8_t>(
                             reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
    } catch (const std::exception&) {
        // A stale use time only changes which image is evicted first.
    }
}

std::uint64_t ContextStore::used_bytes_locked() const {
    return chunk_total_bytes_ + manifest_total_bytes_;
}

bool ContextStore::read_manifest(const fs::path& path, Entry& entry,
                                 std::vector<Segment>* segments) const {
    try {
        return parse_manifest(read_whole_file(path), entry, segments);
    } catch (const std::exception&) {
        return false;
    }
}

bool ContextStore::parse_manifest(std::span<const std::uint8_t> data, Entry& entry,
                                  std::vector<Segment>* segments) const {
    try {
        if (data.size() < sizeof(kManifestMagic) + sizeof(std::uint64_t)) { return false; }
        const std::span<const std::uint8_t> body(data.data(), data.size() - sizeof(std::uint64_t));
        std::uint64_t stored = 0;
        std::memcpy(&stored, data.data() + body.size(), sizeof(stored));
        if (xxh64(body.data(), body.size(), 0x4e43u) != stored) { return false; }
        Reader reader(body);
        char magic[sizeof(kManifestMagic)];
        reader.bytes(magic, sizeof(magic));
        if (std::memcmp(magic, kManifestMagic, sizeof(magic)) != 0 ||
            reader.pod<std::uint32_t>() != kManifestVersion) {
            return false;
        }
        Info& info       = entry.info;
        info.id          = reader.string(64);
        info.binding     = reader.string(1U << 16U);
        info.tokens      = reader.pod<std::uint32_t>();
        info.image_bytes = reader.pod<std::uint64_t>();
        info.created_ms  = reader.pod<std::int64_t>();
        const auto checkpoint_count = reader.pod<std::uint32_t>();
        if (checkpoint_count > kMaximumCheckpoints) { return false; }
        info.checkpoints.resize(checkpoint_count);
        for (CheckpointKey& key : info.checkpoints) {
            key.frontier     = reader.pod<std::uint32_t>();
            key.digests[0]   = reader.pod<std::uint64_t>();
            key.digests[1]   = reader.pod<std::uint64_t>();
            key.identity_tag = reader.pod<std::uint32_t>();
        }
        const auto segment_count = reader.pod<std::uint32_t>();
        if (segment_count > kMaximumSegments) { return false; }
        std::uint64_t covered = 0;
        std::uint64_t total_chunks = 0;
        for (std::uint32_t index = 0; index < segment_count; ++index) {
            Segment segment;
            segment.chunked = reader.pod<std::uint8_t>() != 0;
            segment.length  = reader.pod<std::uint64_t>();
            if (segment.chunked) {
                const auto chunk_count = reader.pod<std::uint32_t>();
                total_chunks += chunk_count;
                if (total_chunks > kMaximumChunksPerImage) { return false; }
                std::uint64_t chunk_sum = 0;
                segment.chunks.resize(chunk_count);
                for (ChunkRef& chunk : segment.chunks) {
                    chunk.hash[0] = reader.pod<std::uint64_t>();
                    chunk.hash[1] = reader.pod<std::uint64_t>();
                    chunk.length  = reader.pod<std::uint32_t>();
                    chunk_sum += chunk.length;
                    entry.chunks.push_back(chunk);
                }
                if (chunk_sum != segment.length) { return false; }
            } else {
                if (segment.length > reader.remaining()) { return false; }
                if (segments != nullptr) {
                    segment.inline_bytes.resize(static_cast<std::size_t>(segment.length));
                    reader.bytes(segment.inline_bytes.data(), segment.inline_bytes.size());
                } else {
                    std::vector<std::uint8_t> skipped(static_cast<std::size_t>(segment.length));
                    reader.bytes(skipped.data(), skipped.size());
                }
            }
            covered += segment.length;
            if (segments != nullptr) { segments->push_back(std::move(segment)); }
        }
        if (covered != info.image_bytes || reader.remaining() != 0 || !valid_id(info.id)) {
            return false;
        }
        entry.manifest_bytes = data.size();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void ContextStore::scan() {
    std::error_code error;
    for (const auto& item : fs::directory_iterator(options_.directory / "tmp", error)) {
        remove_quietly(item.path());
    }
    for (const auto& item : fs::directory_iterator(options_.directory / "manifests", error)) {
        const fs::path& path = item.path();
        if (path.extension() != ".manifest") { continue; }
        Entry entry;
        const std::string id = path.stem().string();
        if (!valid_id(id) || !read_manifest(path, entry, nullptr) || entry.info.id != id) {
            remove_quietly(path);
            remove_quietly(used_path(id));
            continue;
        }
        entry.info.last_used_ms = entry.info.created_ms;
        try {
            const std::vector<std::uint8_t> used = read_whole_file(used_path(id));
            const std::string text(used.begin(), used.end());
            entry.info.last_used_ms = std::max<std::int64_t>(entry.info.created_ms, std::stoll(text));
        } catch (const std::exception&) {}
        manifest_total_bytes_ += entry.manifest_bytes;
        add_references(entry);
        entries_.emplace(id, std::move(entry));
    }
    // Chunks no manifest names are left over from an interrupted write or an eviction.
    for (const auto& shard : fs::directory_iterator(options_.directory / "chunks", error)) {
        if (!shard.is_directory()) { continue; }
        for (const auto& item : fs::directory_iterator(shard.path(), error)) {
            const std::string key = item.path().stem().string();
            if (item.path().extension() != ".chunk" || chunks_.find(key) == chunks_.end()) {
                remove_quietly(item.path());
            }
        }
    }
    // An image whose chunk is missing or the wrong size can never be reassembled.
    std::vector<std::string> broken;
    for (const auto& [id, entry] : entries_) {
        for (const ChunkRef& chunk : entry.chunks) {
            std::error_code stat_error;
            const auto size = fs::file_size(chunk_path(chunk.hash), stat_error);
            if (stat_error || size != chunk.length) {
                if (options_.remote) {
                    // The remote still has it: a missing chunk is fetched when the image is
                    // loaded, and a short one is discarded so it is fetched again.
                    if (!stat_error) { remove_quietly(chunk_path(chunk.hash)); }
                    continue;
                }
                broken.push_back(id);
                break;
            }
        }
    }
    for (const std::string& id : broken) { remove_entry_locked(id, true); }
}

bool ContextStore::entry_local_locked(const Entry& entry) const {
    for (const ChunkRef& chunk : entry.chunks) {
        const auto found = chunks_.find(hex_key(chunk.hash));
        if (found == chunks_.end() || !found->second.present) { return false; }
    }
    return true;
}

bool ContextStore::entry_has_local_bytes_locked(const Entry& entry) const {
    for (const ChunkRef& chunk : entry.chunks) {
        const auto found = chunks_.find(hex_key(chunk.hash));
        if (found != chunks_.end() && found->second.present) { return true; }
    }
    return false;
}

bool ContextStore::evict_oldest_locked(const std::string* protect) {
    const Entry* oldest = nullptr;
    for (const auto& [id, entry] : entries_) {
        if (protect != nullptr && id == *protect) { continue; }
        // An image the directory holds no chunk of costs only its manifest; dropping it frees
        // nothing and the next refresh would register it again.
        if (options_.remote && !entry_has_local_bytes_locked(entry)) { continue; }
        if (oldest == nullptr || entry.info.last_used_ms < oldest->info.last_used_ms) {
            oldest = &entry;
        }
    }
    if (oldest == nullptr) { return false; }
    const std::string id = oldest->info.id;
    remove_entry_locked(id, false);
    ++stats_.evicted_for_space;
    return true;
}

void ContextStore::maintain_locked(const std::string* protect) {
    if (options_.ttl.count() > 0) {
        const std::int64_t horizon = now_ms() - options_.ttl.count() * 1000;
        std::vector<std::string> expired;
        for (const auto& [id, entry] : entries_) {
            if ((protect == nullptr || id != *protect) && entry.info.last_used_ms < horizon) {
                expired.push_back(id);
            }
        }
        for (const std::string& id : expired) {
            remove_entry_locked(id, false);
            ++stats_.expired;
        }
    }
    if (options_.max_bytes != 0) {
        while (used_bytes_locked() > options_.max_bytes &&
               entries_.size() > (protect != nullptr ? 1U : 0U)) {
            if (!evict_oldest_locked(protect)) { break; }
        }
    }
}

void ContextStore::maintain() {
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    maintain_locked(nullptr);
}

ContextStore::PutResult ContextStore::put(const Description& description,
                                          std::span<const std::uint8_t> image,
                                          std::span<const Region> regions) {
    if (!valid_id(description.id)) { throw std::invalid_argument("context store id is invalid"); }
    if (description.checkpoints.size() > kMaximumCheckpoints) {
        throw std::invalid_argument("too many checkpoint keys");
    }
    std::uint64_t cursor = 0;
    for (const Region& region : regions) {
        if (region.length == 0 || region.offset < cursor ||
            region.offset > image.size() || region.length > image.size() - region.offset) {
            throw std::invalid_argument("context store regions must be ordered and inside the image");
        }
        cursor = region.offset + region.length;
    }

    std::scoped_lock io(io_mutex_);
    PutResult result;
    result.image_bytes = image.size();
    std::vector<Segment> segments;
    std::vector<ChunkRef> all_chunks;
    std::vector<fs::path> written;
    std::unordered_set<std::string> seen_in_put;
    try {
        const auto add_inline = [&](std::uint64_t begin, std::uint64_t end) {
            if (end <= begin) { return; }
            Segment segment;
            segment.length = end - begin;
            segment.inline_bytes.assign(image.begin() + static_cast<std::ptrdiff_t>(begin),
                                        image.begin() + static_cast<std::ptrdiff_t>(end));
            segments.push_back(std::move(segment));
        };
        std::uint64_t position = 0;
        for (const Region& region : regions) {
            add_inline(position, region.offset);
            Segment segment;
            segment.chunked = true;
            segment.length  = region.length;
            for (std::uint64_t done = 0; done < region.length; done += options_.chunk_bytes) {
                const std::uint64_t size = std::min(options_.chunk_bytes, region.length - done);
                const std::span<const std::uint8_t> bytes =
                    image.subspan(static_cast<std::size_t>(region.offset + done),
                                  static_cast<std::size_t>(size));
                ChunkRef chunk{.hash = hash(bytes), .length = static_cast<std::uint32_t>(size)};
                const std::string key = hex_key(chunk.hash);
                bool present;
                {
                    std::scoped_lock lock(mutex_);
                    present = chunks_.find(key) != chunks_.end();
                }
                if (!present && seen_in_put.find(key) == seen_in_put.end()) {
                    write_atomically(chunk_path(chunk.hash), bytes);
                    written.push_back(chunk_path(chunk.hash));
                    result.chunk_bytes_written += size;
                } else {
                    result.chunk_bytes_reused += size;
                }
                seen_in_put.insert(key);
                segment.chunks.push_back(chunk);
                all_chunks.push_back(chunk);
            }
            segments.push_back(std::move(segment));
            position = region.offset + region.length;
        }
        add_inline(position, image.size());

        Writer manifest;
        const std::int64_t now = now_ms();
        manifest.bytes(kManifestMagic, sizeof(kManifestMagic));
        manifest.pod(kManifestVersion);
        manifest.string(description.id);
        manifest.string(description.binding);
        manifest.pod<std::uint32_t>(description.tokens);
        manifest.pod<std::uint64_t>(image.size());
        manifest.pod<std::int64_t>(now);
        manifest.pod<std::uint32_t>(static_cast<std::uint32_t>(description.checkpoints.size()));
        for (const CheckpointKey& key : description.checkpoints) {
            manifest.pod(key.frontier);
            manifest.pod(key.digests[0]);
            manifest.pod(key.digests[1]);
            manifest.pod(key.identity_tag);
        }
        manifest.pod<std::uint32_t>(static_cast<std::uint32_t>(segments.size()));
        for (const Segment& segment : segments) {
            manifest.pod<std::uint8_t>(segment.chunked ? 1 : 0);
            manifest.pod<std::uint64_t>(segment.length);
            if (segment.chunked) {
                manifest.pod<std::uint32_t>(static_cast<std::uint32_t>(segment.chunks.size()));
                for (const ChunkRef& chunk : segment.chunks) {
                    manifest.pod(chunk.hash[0]);
                    manifest.pod(chunk.hash[1]);
                    manifest.pod(chunk.length);
                }
            } else {
                manifest.bytes(segment.inline_bytes.data(), segment.inline_bytes.size());
            }
        }
        const std::uint64_t checksum =
            xxh64(manifest.out.data(), manifest.out.size(), 0x4e43u);
        manifest.pod(checksum);
        write_atomically(manifest_path(description.id), manifest.out);

        Entry entry;
        entry.info = Info{.id           = description.id,
                          .binding      = description.binding,
                          .tokens       = description.tokens,
                          .image_bytes  = image.size(),
                          .created_ms   = now,
                          .last_used_ms = now,
                          .checkpoints  = description.checkpoints};
        entry.manifest_bytes = manifest.out.size();
        entry.chunks         = std::move(all_chunks);

        std::scoped_lock lock(mutex_);
        const auto previous = entries_.find(description.id);
        // Reference the new image before releasing the one it replaces, so shared chunks survive.
        Entry replaced;
        const bool had_previous = previous != entries_.end();
        if (had_previous) {
            replaced = std::move(previous->second);
            entries_.erase(previous);
            manifest_total_bytes_ -= replaced.manifest_bytes;
        }
        manifest_total_bytes_ += entry.manifest_bytes;
        add_references(entry);
        entries_.emplace(description.id, std::move(entry));
        if (had_previous) { release_references(replaced); }
        write_used(description.id, now);
        if (!description.prefix_digests.empty()) {
            std::vector<std::string> superseded;
            for (const auto& [other_id, other] : entries_) {
                if (other_id == description.id || other.info.binding != description.binding ||
                    other.info.tokens > description.tokens || other.info.checkpoints.empty()) {
                    continue;
                }
                const CheckpointKey& endpoint = other.info.checkpoints.front();
                if (!description.checkpoints.empty() &&
                    endpoint.identity_tag != description.checkpoints.front().identity_tag) {
                    continue;
                }
                if (endpoint.frontier < description.prefix_digests.size() &&
                    description.prefix_digests[endpoint.frontier] == endpoint.digests) {
                    superseded.push_back(other_id);
                }
            }
            for (const std::string& other_id : superseded) {
                remove_entry_locked(other_id, false);
                ++stats_.superseded;
            }
        }
        ++stats_.puts;
        stats_.bytes_written += result.chunk_bytes_written;
        stats_.bytes_reused += result.chunk_bytes_reused;
        const std::uint64_t evictions_before = stats_.evicted_for_space + stats_.expired;
        maintain_locked(&description.id);
        result.evicted_images = stats_.evicted_for_space + stats_.expired - evictions_before;
    } catch (...) {
        for (const fs::path& path : written) { remove_quietly(path); }
        std::scoped_lock lock(mutex_);
        ++stats_.put_failures;
        throw;
    }
    enqueue_remote({RemoteTask::Kind::Upload, description.id});
    return result;
}

std::optional<std::vector<std::uint8_t>> ContextStore::load(const std::string& id, bool touch) {
    if (!valid_id(id)) { return std::nullopt; }
    if (options_.remote) {
        bool needs_fetch = false;
        {
            std::scoped_lock lock(mutex_);
            const auto found = entries_.find(id);
            needs_fetch      = found != entries_.end() && !entry_local_locked(found->second);
        }
        if (needs_fetch) {
            // Network time: no lock is held, so puts and lookups carry on meanwhile.
            const FetchStatus status = fetch_image_chunks(id);
            if (status != FetchStatus::Complete) {
                std::scoped_lock io(io_mutex_);
                std::scoped_lock lock(mutex_);
                ++stats_.loads;
                ++stats_.load_misses;
                if (status == FetchStatus::Corrupt) {
                    remove_entry_locked(id, true);
                    std::scoped_lock remote_lock(remote_mutex_);
                    remote_failed_.insert(id);
                }
                return std::nullopt;
            }
        }
    }
    std::scoped_lock io(io_mutex_);
    Entry entry;
    std::vector<Segment> segments;
    {
        std::scoped_lock lock(mutex_);
        ++stats_.loads;
        if (entries_.find(id) == entries_.end()) {
            ++stats_.load_misses;
            return std::nullopt;
        }
    }
    const auto fail = [&]() -> std::optional<std::vector<std::uint8_t>> {
        std::scoped_lock lock(mutex_);
        remove_entry_locked(id, true);
        ++stats_.load_misses;
        return std::nullopt;
    };
    if (!read_manifest(manifest_path(id), entry, &segments)) { return fail(); }
    std::vector<std::uint8_t> image(static_cast<std::size_t>(entry.info.image_bytes));
    std::size_t position = 0;
    for (const Segment& segment : segments) {
        if (!segment.chunked) {
            std::memcpy(image.data() + position, segment.inline_bytes.data(),
                        segment.inline_bytes.size());
            position += segment.inline_bytes.size();
            continue;
        }
        for (const ChunkRef& chunk : segment.chunks) {
            std::ifstream file(chunk_path(chunk.hash), std::ios::binary);
            if (!file ||
                !file.read(reinterpret_cast<char*>(image.data() + position), chunk.length) ||
                hash(std::span<const std::uint8_t>(image.data() + position, chunk.length)) !=
                    chunk.hash) {
                return fail();
            }
            position += chunk.length;
        }
    }
    if (position != image.size()) { return fail(); }
    if (touch) {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found != entries_.end()) {
            found->second.info.last_used_ms = now_ms();
            write_used(id, found->second.info.last_used_ms);
        }
    }
    enqueue_remote({RemoteTask::Kind::Touch, id});
    return image;
}

std::vector<ContextStore::Info> ContextStore::list() const {
    std::scoped_lock lock(mutex_);
    std::vector<Info> out;
    out.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        out.push_back(entry.info);
        out.back().local = entry_local_locked(entry);
    }
    std::sort(out.begin(), out.end(), [](const Info& left, const Info& right) {
        return left.last_used_ms > right.last_used_ms;
    });
    return out;
}

std::optional<ContextStore::Info> ContextStore::find(const std::string& id) const {
    std::scoped_lock lock(mutex_);
    const auto found = entries_.find(id);
    if (found == entries_.end()) { return std::nullopt; }
    Info info  = found->second.info;
    info.local = entry_local_locked(found->second);
    return info;
}

bool ContextStore::erase(const std::string& id) {
    if (!valid_id(id)) { return false; }
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    if (entries_.find(id) == entries_.end()) { return false; }
    remove_entry_locked(id, false);
    return true;
}

void ContextStore::touch(const std::string& id) {
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    const auto found = entries_.find(id);
    if (found == entries_.end()) { return; }
    found->second.info.last_used_ms = now_ms();
    write_used(id, found->second.info.last_used_ms);
    enqueue_remote({RemoteTask::Kind::Touch, id});
}

ContextStore::Stats ContextStore::stats() const {
    std::scoped_lock lock(mutex_);
    Stats out        = stats_;
    out.used_bytes   = used_bytes_locked();
    out.images       = entries_.size();
    if (options_.remote) {
        for (const auto& [id, entry] : entries_) {
            if (!entry_local_locked(entry)) { ++out.remote_images; }
        }
    }
    return out;
}

std::string ContextStore::remote_manifest_key(const std::string& id) const {
    return options_.remote_prefix + "manifests/" + id + ".manifest";
}

std::string ContextStore::remote_chunk_key(const std::array<std::uint64_t, 2>& hash) const {
    const std::string key = hex_key(hash);
    return options_.remote_prefix + "chunks/" + key.substr(0, 2) + "/" + key + ".chunk";
}

void ContextStore::enqueue_remote(RemoteTask task) {
    if (!options_.remote) { return; }
    {
        std::scoped_lock lock(remote_mutex_);
        if (remote_stopping_) { return; }
        for (const RemoteTask& queued : remote_queue_) {
            if (queued.kind == task.kind && queued.id == task.id) { return; }
        }
        remote_queue_.push_back(std::move(task));
    }
    remote_cv_.notify_one();
}

void ContextStore::remote_loop() {
    auto next_refresh = std::chrono::steady_clock::now() + options_.remote_refresh;
    std::unique_lock lock(remote_mutex_);
    for (;;) {
        remote_cv_.wait_until(lock, next_refresh,
                              [&] { return remote_stopping_ || !remote_queue_.empty(); });
        if (remote_stopping_) { return; }
        RemoteTask task{RemoteTask::Kind::Refresh, std::string()};
        if (!remote_queue_.empty()) {
            task = std::move(remote_queue_.front());
            remote_queue_.pop_front();
        } else if (std::chrono::steady_clock::now() >= next_refresh) {
            next_refresh = std::chrono::steady_clock::now() + options_.remote_refresh;
        } else {
            continue;
        }
        remote_busy_ = true;
        lock.unlock();
        try {
            switch (task.kind) {
            case RemoteTask::Kind::Upload: upload_image(task.id); break;
            case RemoteTask::Kind::Refresh: (void)refresh_remote(); break;
            case RemoteTask::Kind::Touch: touch_remote_image(task.id); break;
            case RemoteTask::Kind::Prefetch:
                if (fetch_image_chunks(task.id) == FetchStatus::Complete) {
                    std::scoped_lock io(io_mutex_);
                    std::scoped_lock state(mutex_);
                    maintain_locked(&task.id);
                }
                break;
            }
        } catch (...) {}
        lock.lock();
        remote_busy_ = false;
        if (remote_queue_.empty()) { remote_idle_cv_.notify_all(); }
    }
}

bool ContextStore::flush_remote(std::chrono::steady_clock::time_point deadline) {
    if (!options_.remote) { return true; }
    std::unique_lock lock(remote_mutex_);
    return remote_idle_cv_.wait_until(
        lock, deadline, [&] { return remote_queue_.empty() && !remote_busy_; });
}

bool ContextStore::prefetch(const std::string& id) {
    if (!options_.remote || !valid_id(id)) { return false; }
    {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found == entries_.end() || entry_local_locked(found->second)) { return false; }
    }
    enqueue_remote({RemoteTask::Kind::Prefetch, id});
    return true;
}

void ContextStore::upload_image(const std::string& id) {
    Entry entry;
    {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found == entries_.end()) { return; }
        entry = found->second;
    }
    std::uint64_t objects = 0;
    std::uint64_t bytes   = 0;
    try {
        std::unordered_set<std::string> seen;
        for (const ChunkRef& chunk : entry.chunks) {
            const std::string key = remote_chunk_key(chunk.hash);
            bool known;
            {
                std::scoped_lock lock(remote_mutex_);
                known = uploaded_.count(key) != 0;
            }
            if (known || !seen.insert(key).second) { continue; }
            if (!options_.remote->exists(key)) {
                std::vector<std::uint8_t> data;
                try {
                    data = read_whole_file(chunk_path(chunk.hash));
                } catch (const std::exception&) {
                    continue; // evicted from the directory since: nothing to upload
                }
                if (data.size() != chunk.length) { continue; }
                options_.remote->put(key, data);
                ++objects;
                bytes += data.size();
            }
            std::scoped_lock lock(remote_mutex_);
            uploaded_.insert(key);
        }
        // The manifest goes last: an image another engine sees always has its chunks.
        std::vector<std::uint8_t> manifest;
        try {
            manifest = read_whole_file(manifest_path(id));
        } catch (const std::exception&) {
            return; // replaced or evicted since
        }
        options_.remote->put(remote_manifest_key(id), manifest);
        ++objects;
        bytes += manifest.size();
        std::scoped_lock lock(mutex_);
        stats_.remote_uploads += objects;
        stats_.remote_upload_bytes += bytes;
    } catch (const std::exception&) {
        std::scoped_lock lock(mutex_);
        stats_.remote_uploads += objects;
        stats_.remote_upload_bytes += bytes;
        ++stats_.remote_upload_failures;
    }
}

void ContextStore::touch_remote_image(const std::string& id) {
    std::vector<ChunkRef> chunks;
    {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found == entries_.end()) { return; }
        chunks = found->second.chunks;
    }
    const std::int64_t now = now_ms();
    {
        std::scoped_lock lock(remote_mutex_);
        const auto last = remote_touched_.find(id);
        if (last != remote_touched_.end() &&
            now - last->second < options_.remote_touch_interval.count() * 1000) {
            return;
        }
    }
    try {
        for (const ChunkRef& chunk : chunks) { (void)options_.remote->touch(remote_chunk_key(chunk.hash)); }
        (void)options_.remote->touch(remote_manifest_key(id));
        std::scoped_lock lock(remote_mutex_);
        remote_touched_[id] = now;
    } catch (const std::exception&) {}
}

ContextStore::FetchStatus ContextStore::fetch_image_chunks(const std::string& id) {
    std::vector<ChunkRef> chunks;
    {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found == entries_.end()) { return FetchStatus::Corrupt; }
        chunks = found->second.chunks;
    }
    for (const ChunkRef& chunk : chunks) {
        {
            std::scoped_lock lock(mutex_);
            const auto found = chunks_.find(hex_key(chunk.hash));
            if (found != chunks_.end() && found->second.present) { continue; }
        }
        std::optional<std::vector<std::uint8_t>> data;
        try {
            data = options_.remote->get(remote_chunk_key(chunk.hash));
        } catch (const std::exception&) {
            std::scoped_lock lock(mutex_);
            ++stats_.remote_download_failures;
            return FetchStatus::Transient;
        }
        if (!data || data->size() != chunk.length || hash(*data) != chunk.hash) {
            std::scoped_lock lock(mutex_);
            ++stats_.remote_download_failures;
            return FetchStatus::Corrupt;
        }
        if (!install_chunk(chunk, *data)) { return FetchStatus::Corrupt; }
        std::scoped_lock lock(mutex_);
        ++stats_.remote_downloads;
        stats_.remote_download_bytes += data->size();
    }
    return FetchStatus::Complete;
}

bool ContextStore::install_chunk(const ChunkRef& chunk, std::span<const std::uint8_t> bytes) {
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    const auto found = chunks_.find(hex_key(chunk.hash));
    if (found == chunks_.end() || found->second.references == 0) { return false; } // image removed
    if (found->second.present) { return true; }
    write_atomically(chunk_path(chunk.hash), bytes);
    found->second.present = true;
    chunk_total_bytes_ += chunk.length;
    return true;
}

std::size_t ContextStore::refresh_remote() {
    if (!options_.remote) { return 0; }
    std::size_t added = 0;
    try {
        const std::string manifests = options_.remote_prefix + "manifests/";
        const std::vector<ObjectInfo> objects = options_.remote->list(manifests);
        const std::int64_t horizon =
            options_.ttl.count() > 0 ? now_ms() - options_.ttl.count() * 1000
                                     : std::numeric_limits<std::int64_t>::min();
        for (const ObjectInfo& object : objects) {
            constexpr std::string_view kSuffix = ".manifest";
            if (object.key.size() <= manifests.size() + kSuffix.size() ||
                object.key.compare(object.key.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0) {
                continue;
            }
            const std::string id = object.key.substr(
                manifests.size(), object.key.size() - manifests.size() - kSuffix.size());
            if (!valid_id(id) || object.modified_ms < horizon) { continue; }
            {
                std::scoped_lock lock(mutex_);
                if (entries_.find(id) != entries_.end()) { continue; }
            }
            {
                std::scoped_lock lock(remote_mutex_);
                if (remote_failed_.count(id) != 0) { continue; }
            }
            const std::optional<std::vector<std::uint8_t>> data = options_.remote->get(object.key);
            if (!data) { continue; }
            Entry entry;
            if (!parse_manifest(*data, entry, nullptr) || entry.info.id != id) {
                std::scoped_lock lock(remote_mutex_);
                remote_failed_.insert(id);
                continue;
            }
            entry.info.last_used_ms = std::max(entry.info.created_ms, object.modified_ms);
            std::scoped_lock io(io_mutex_);
            std::scoped_lock lock(mutex_);
            if (entries_.find(id) != entries_.end()) { continue; }
            write_atomically(manifest_path(id), *data);
            manifest_total_bytes_ += entry.manifest_bytes;
            add_references(entry);
            {
                std::scoped_lock remote_lock(remote_mutex_);
                for (const ChunkRef& chunk : entry.chunks) { uploaded_.insert(remote_chunk_key(chunk.hash)); }
            }
            entries_.emplace(id, std::move(entry));
            ++added;
        }
        std::scoped_lock lock(mutex_);
        ++stats_.remote_refreshes;
    } catch (const std::exception&) {
        std::scoped_lock lock(mutex_);
        ++stats_.remote_download_failures;
    }
    return added;
}

} // namespace ninfer::runtime
