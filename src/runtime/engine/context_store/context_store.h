#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime {

// A durable, bounded, incremental store for the byte images of retained sessions.
//
// A session image is several GB for a deep context, and consecutive images of one conversation share
// almost all of it: the older KV pages and the immutable checkpoint states do not change between
// saves, only the newest pages and the endpoint state do. Writing the whole image every time costs
// minutes on a slow disk, so the store splits the declared regions of an image into fixed-size
// chunks addressed by a hash of their content and writes only the chunks it does not already hold.
// Everything outside the regions (the header with the token ledger and identity, and the integrity
// trailer) is small and travels inside the manifest.
//
// The store knows nothing about the image's meaning. The caller declares the regions and the
// checkpoint keys by which a later request finds the image, and decides whether bytes it gets back
// belong to the running model (the image carries its own binding and checksum, which the restore
// validates).
//
// Everything is safe against a crash at any point: chunks are written under a temporary name and
// renamed, the manifest is renamed last, and a start-up scan removes what no manifest names. A
// failure on any single file turns into a miss, never into wrong data.
class ContextStore {
public:
    using Clock = std::function<std::int64_t()>; // Unix milliseconds

    struct Options {
        std::filesystem::path directory;
        // Bytes of chunks and manifests kept; the least recently used images are evicted beyond it.
        // 0 means no limit.
        std::uint64_t max_bytes = 0;
        // Images not used for this long are removed. Zero means no expiry.
        std::chrono::seconds ttl{0};
        std::uint64_t chunk_bytes = std::uint64_t{32} << 20U;
        Clock clock; // defaults to the system clock
    };

    // Where in the image the large, deduplicated bytes are.
    struct Region {
        std::uint64_t offset = 0;
        std::uint64_t length = 0;
    };

    // The key a later request computes for a prefix of its own prompt: if it matches, the image
    // holds a checkpoint at exactly that prefix.
    struct CheckpointKey {
        std::uint32_t frontier = 0;
        std::array<std::uint64_t, 2> digests{};
        std::uint32_t identity_tag = 0;

        [[nodiscard]] friend constexpr bool operator==(const CheckpointKey&,
                                                       const CheckpointKey&) noexcept = default;
    };

    struct Description {
        std::string id;      // lowercase hex, 1-64 characters
        std::string binding; // identity of the model and configuration that produced the image
        std::uint32_t tokens = 0;
        std::vector<CheckpointKey> checkpoints;
        // The image this one replaces, if any; removed once this one is durable.
        std::optional<std::string> supersedes;
    };

    struct Info {
        std::string id;
        std::string binding;
        std::uint32_t tokens        = 0;
        std::uint64_t image_bytes   = 0;
        std::int64_t created_ms     = 0;
        std::int64_t last_used_ms   = 0;
        std::vector<CheckpointKey> checkpoints;
    };

    struct PutResult {
        std::uint64_t image_bytes         = 0;
        std::uint64_t chunk_bytes_written = 0; // new chunk files
        std::uint64_t chunk_bytes_reused  = 0; // chunks the store already held
        std::uint64_t evicted_images      = 0;
    };

    struct Stats {
        std::uint64_t used_bytes        = 0;
        std::uint64_t images            = 0;
        std::uint64_t puts              = 0;
        std::uint64_t put_failures      = 0;
        std::uint64_t bytes_written     = 0;
        std::uint64_t bytes_reused      = 0;
        std::uint64_t loads             = 0;
        std::uint64_t load_misses       = 0;
        std::uint64_t corrupt_removed   = 0;
        std::uint64_t evicted_for_space = 0;
        std::uint64_t expired           = 0;
    };

    // Creates the directory layout, reads the manifests found there and removes whatever is
    // unreadable, unreferenced or left over from an interrupted write.
    explicit ContextStore(Options options);

    // Stores `image` (its `regions` chunked, the rest inline). Idempotent for an id already held:
    // the manifest is replaced. Throws std::runtime_error when the files cannot be written; the
    // store is left consistent and nothing of the failed image remains visible.
    PutResult put(const Description& description, std::span<const std::uint8_t> image,
                  std::span<const Region> regions);

    // Reassembles an image, verifying every chunk. nullopt when it is absent or damaged; a damaged
    // image is removed so it is not offered again. Counts as a use.
    [[nodiscard]] std::optional<std::vector<std::uint8_t>> load(const std::string& id);

    [[nodiscard]] std::vector<Info> list() const;
    [[nodiscard]] std::optional<Info> find(const std::string& id) const;
    bool erase(const std::string& id);
    void touch(const std::string& id);

    // Removes expired images and applies the size limit. Called after each put and at start-up;
    // exposed so a long-idle server can run it.
    void maintain();

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] const Options& options() const noexcept { return options_; }

    // 128-bit content hash used for chunk names.
    [[nodiscard]] static std::array<std::uint64_t, 2> hash(std::span<const std::uint8_t> bytes);

private:
    struct ChunkRef {
        std::array<std::uint64_t, 2> hash{};
        std::uint32_t length = 0;
    };
    struct Segment {
        bool chunked = false;
        std::uint64_t length = 0;
        std::vector<std::uint8_t> inline_bytes; // inline segments only
        std::vector<ChunkRef> chunks;           // chunked segments only
    };
    struct Entry {
        Info info;
        std::uint64_t manifest_bytes = 0;
        std::vector<ChunkRef> chunks; // every chunk the manifest names, in order
    };
    struct ChunkUse {
        std::uint32_t references = 0;
        std::uint32_t length     = 0;
    };

    [[nodiscard]] std::filesystem::path manifest_path(const std::string& id) const;
    [[nodiscard]] std::filesystem::path used_path(const std::string& id) const;
    [[nodiscard]] std::filesystem::path chunk_path(const std::array<std::uint64_t, 2>& hash) const;
    [[nodiscard]] std::int64_t now_ms() const;

    void scan();
    bool read_manifest(const std::filesystem::path& path, Entry& entry,
                       std::vector<Segment>* segments) const;
    void write_atomically(const std::filesystem::path& path,
                          std::span<const std::uint8_t> bytes) const;
    void add_references(const Entry& entry);
    void release_references(const Entry& entry);
    void remove_entry_locked(const std::string& id, bool corrupt);
    void write_used(const std::string& id, std::int64_t last_used_ms) const;
    [[nodiscard]] std::uint64_t used_bytes_locked() const;
    void maintain_locked(const std::string* protect);
    void evict_oldest_locked(const std::string* protect);

    Options options_;
    // Serializes everything that changes files (put, load, erase, touch, maintain), so a chunk is
    // never deleted while another call reads or reuses it. Lookups use only `mutex_` and so are not
    // held up by a long write.
    mutable std::mutex io_mutex_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
    std::unordered_map<std::string, ChunkUse> chunks_; // keyed by hex hash
    std::uint64_t chunk_total_bytes_ = 0;
    std::uint64_t manifest_total_bytes_ = 0;
    Stats stats_;
};

} // namespace ninfer::runtime
