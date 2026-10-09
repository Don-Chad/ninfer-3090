#pragma once

#include "ninfer/object_store.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

        // An optional remote copy of the whole store (an S3-compatible bucket). The directory then
        // acts as a cache of it: every image written is uploaded in the background, images other
        // engines wrote appear in list() and are fetched when loaded, and an image evicted from
        // the directory stays available remotely. The remote is never trusted for correctness:
        // every chunk it returns is verified, and a failure is a miss. Expiry there is the
        // bucket's lifecycle rule; images that are used have their age restarted.
        std::shared_ptr<ObjectStore> remote;
        std::string remote_prefix; // prepended to every key, e.g. "ninfer/"
        std::chrono::seconds remote_refresh{60};
        std::chrono::seconds remote_touch_interval{24 * 3600};
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
        // The first entry is the image's endpoint (the deepest state it can resume from).
        std::vector<CheckpointKey> checkpoints;
        // The image's own prefix digest at every token frontier 0..tokens (not stored). An older
        // image of the same model whose endpoint key this reproduces is an earlier state of the
        // same conversation: this image contains everything it does, so once this one is durable
        // the older one is removed instead of being kept as a near-duplicate until it ages out.
        std::span<const std::array<std::uint64_t, 2>> prefix_digests;
    };

    struct Info {
        std::string id;
        std::string binding;
        std::uint32_t tokens        = 0;
        std::uint64_t image_bytes   = 0;
        std::int64_t created_ms     = 0;
        std::int64_t last_used_ms   = 0;
        std::vector<CheckpointKey> checkpoints;
        // Every chunk of the image is in the directory, so load() does not touch the network.
        bool local = true;
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
        std::uint64_t superseded        = 0;
        // With a remote: images only the remote holds, and the traffic to it.
        std::uint64_t remote_images          = 0;
        std::uint64_t remote_uploads         = 0; // objects uploaded (chunks and manifests)
        std::uint64_t remote_upload_bytes    = 0;
        std::uint64_t remote_upload_failures = 0;
        std::uint64_t remote_downloads       = 0;
        std::uint64_t remote_download_bytes  = 0;
        std::uint64_t remote_download_failures = 0;
        std::uint64_t remote_refreshes       = 0;
    };

    // Creates the directory layout, reads the manifests found there and removes whatever is
    // unreadable, unreferenced or left over from an interrupted write.
    explicit ContextStore(Options options);
    ~ContextStore();
    ContextStore(const ContextStore&)            = delete;
    ContextStore& operator=(const ContextStore&) = delete;

    // Stores `image` (its `regions` chunked, the rest inline). Idempotent for an id already held:
    // the manifest is replaced. Throws std::runtime_error when the files cannot be written; the
    // store is left consistent and nothing of the failed image remains visible.
    PutResult put(const Description& description, std::span<const std::uint8_t> image,
                  std::span<const Region> regions);

    // Reassembles an image, verifying every chunk. nullopt when it is absent or damaged; a damaged
    // image is removed so it is not offered again. Counts as a use.
    // `touch` false reads without refreshing the image's last use; the caller then calls touch()
    // once it has actually used the image.
    [[nodiscard]] std::optional<std::vector<std::uint8_t>> load(const std::string& id,
                                                                bool touch = true);

    [[nodiscard]] std::vector<Info> list() const;
    [[nodiscard]] std::optional<Info> find(const std::string& id) const;
    bool erase(const std::string& id);
    void touch(const std::string& id);

    // Removes expired images and applies the size limit. Called after each put and at start-up;
    // exposed so a long-idle server can run it.
    void maintain();

    // With a remote: stops the background transfers after the work already queued, waiting at most
    // until `deadline`. For shutdown, after the last put. Returns whether the queue drained.
    bool flush_remote(std::chrono::steady_clock::time_point deadline);
    // With a remote: lists it now and registers images it holds that this store does not. Also done
    // in the background every remote_refresh. Returns the number of images added.
    // Lists the bucket and registers images other engines left there. Stops reading manifests at
    // `deadline`; an incomplete listing never removes anything from the index.
    std::size_t refresh_remote(std::chrono::steady_clock::time_point deadline =
                                   std::chrono::steady_clock::time_point::max());
    // With a remote: fetches the image's missing chunks in the background so a later load() is
    // local. False when there is nothing to do.
    bool prefetch(const std::string& id);

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
        // Which write of this id this is. A transfer that outlives the entry it started from
        // must not act on the entry that replaced it.
        std::uint64_t generation = 0;
    };
    struct ChunkUse {
        std::uint32_t references = 0;
        // Queued uploads that still need the file: it outlives its last reference until they finish.
        std::uint32_t pins       = 0;
        std::uint32_t length     = 0;
        bool present             = false; // the chunk file is in the directory
    };
    struct RemoteTask {
        enum class Kind : std::uint8_t { Upload, Refresh, Prefetch, Touch } kind;
        std::string id;
        // Upload: the image as it was when it was written (the live entry may have been replaced
        // or evicted since), with its chunk files pinned until the upload ends.
        std::vector<std::uint8_t> manifest;
        std::vector<ChunkRef> chunks;
    };

    [[nodiscard]] std::filesystem::path manifest_path(const std::string& id) const;
    [[nodiscard]] std::filesystem::path used_path(const std::string& id) const;
    [[nodiscard]] std::filesystem::path chunk_path(const std::array<std::uint64_t, 2>& hash) const;
    [[nodiscard]] std::int64_t now_ms() const;

    void scan();
    bool read_manifest(const std::filesystem::path& path, Entry& entry,
                       std::vector<Segment>* segments) const;
    bool parse_manifest(std::span<const std::uint8_t> data, Entry& entry,
                        std::vector<Segment>* segments) const;
    void write_atomically(const std::filesystem::path& path,
                          std::span<const std::uint8_t> bytes) const;
    void add_references(const Entry& entry);
    void release_references(const Entry& entry);
    void remove_entry_locked(const std::string& id, bool corrupt);
    void write_used(const std::string& id, std::int64_t last_used_ms) const;
    [[nodiscard]] std::uint64_t used_bytes_locked() const;
    void maintain_locked(const std::string* protect);
    bool evict_oldest_locked(const std::string* protect);
    [[nodiscard]] bool entry_local_locked(const Entry& entry) const;
    [[nodiscard]] bool entry_has_local_bytes_locked(const Entry& entry) const;

    // The remote tier (all no-ops without options_.remote).
    [[nodiscard]] std::string remote_manifest_key(const std::string& id) const;
    [[nodiscard]] std::string remote_chunk_key(const std::array<std::uint64_t, 2>& hash) const;
    void enqueue_remote(RemoteTask task);
    void enqueue_upload(const std::string& id, std::vector<std::uint8_t> manifest,
                        std::vector<ChunkRef> chunks);
    void unpin_chunks_locked(const std::vector<ChunkRef>& chunks);
    void remote_loop();
    void upload_image(RemoteTask& task);
    void touch_remote_image(const std::string& id);
    enum class FetchStatus : std::uint8_t { Complete, Transient, Corrupt };
    // `generation` receives the generation of the entry the fetch worked from.
    [[nodiscard]] FetchStatus fetch_image_chunks(const std::string& id, std::uint64_t& generation);
    // A fetch of `generation` found the image damaged or gone in the bucket: drop it and do not
    // offer it again, unless the entry has been replaced since.
    void settle_failed_fetch(const std::string& id, std::uint64_t generation);
    bool install_chunk(const ChunkRef& chunk, std::span<const std::uint8_t> bytes);

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
    std::uint64_t next_generation_      = 0; // guarded by mutex_
    Stats stats_;

    // Remote worker: one thread runs the queued transfers, one at a time.
    std::mutex remote_mutex_;
    std::condition_variable remote_cv_;
    std::condition_variable remote_idle_cv_;
    std::deque<RemoteTask> remote_queue_;
    bool remote_busy_     = false;
    bool remote_stopping_ = false;
    // Chunk keys last confirmed in the bucket, and when: a lifecycle rule may have expired them
    // since, so the confirmation is trusted for remote_touch_interval only.
    std::unordered_map<std::string, std::int64_t> uploaded_;
    std::unordered_map<std::string, std::int64_t> remote_touched_; // id -> last touch (ms)
    std::unordered_set<std::string> remote_failed_;               // ids not worth importing again
    std::thread remote_thread_;
};

} // namespace ninfer::runtime
