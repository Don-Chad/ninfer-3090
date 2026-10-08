#include "runtime/engine/context_store/context_store.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ninfer::runtime::ContextStore;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL " << message << '\n';
}

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path() /
                ("ninfer_context_store_test_" + std::to_string(counter.fetch_add(1)) + "_" +
                 std::to_string(
                     std::chrono::steady_clock::now().time_since_epoch().count() & 0xffffff));
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

std::vector<std::uint8_t> pseudo_random(std::size_t size, std::uint32_t seed) {
    std::vector<std::uint8_t> out(size);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (auto& byte : out) {
        state = state * 1664525U + 1013904223U;
        byte  = static_cast<std::uint8_t>(state >> 24U);
    }
    return out;
}

std::vector<std::uint8_t> concat(std::initializer_list<std::vector<std::uint8_t>> parts) {
    std::vector<std::uint8_t> out;
    for (const auto& part : parts) { out.insert(out.end(), part.begin(), part.end()); }
    return out;
}

ContextStore::Options options_for(const fs::path& directory) {
    ContextStore::Options options;
    options.directory   = directory;
    options.chunk_bytes = 1024;
    return options;
}

ContextStore::Description describe(const std::string& id, std::uint32_t tokens = 100) {
    ContextStore::Description description;
    description.id      = id;
    description.binding = "model-a";
    description.tokens  = tokens;
    description.checkpoints.push_back(ContextStore::CheckpointKey{
        .frontier = tokens - 1, .digests = {0x1111, 0x2222}, .identity_tag = 7});
    return description;
}

std::vector<fs::path> chunk_files(const fs::path& directory) {
    std::vector<fs::path> files;
    for (const auto& item : fs::recursive_directory_iterator(directory / "chunks")) {
        if (item.is_regular_file()) { files.push_back(item.path()); }
    }
    return files;
}

void test_hash() {
    // XXH64 of the empty input with seed 0 is a published constant.
    check(ContextStore::hash({})[0] == 0xef46db3751d8e999ULL, "xxh64 of empty input");
    const auto a = pseudo_random(5000, 1);
    auto b       = a;
    check(ContextStore::hash(a) == ContextStore::hash(a), "hash is not deterministic");
    b[4999] ^= 1;
    check(ContextStore::hash(a) != ContextStore::hash(b), "a one-bit change did not change the hash");
    const auto short_input = pseudo_random(3, 2);
    check(ContextStore::hash(short_input) != ContextStore::hash({}), "short input hashes alike");
}

void test_round_trip() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto head   = pseudo_random(300, 1);
    const auto state  = pseudo_random(2500, 2);
    const auto kv     = pseudo_random(4000, 3);
    const auto tail   = pseudo_random(8, 4);
    const auto image  = concat({head, state, kv, tail});
    const std::array regions{ContextStore::Region{300, 2500}, ContextStore::Region{2800, 4000}};
    const auto put = store.put(describe("abc123"), image, regions);
    check(put.image_bytes == image.size() && put.chunk_bytes_written == 6500 &&
              put.chunk_bytes_reused == 0,
          "first put did not write every region chunk");
    const auto loaded = store.load("abc123");
    check(loaded && *loaded == image, "round trip changed the image");
    const auto info = store.find("abc123");
    check(info && info->tokens == 100 && info->binding == "model-a" &&
              info->checkpoints.size() == 1 && info->checkpoints[0].digests[1] == 0x2222,
          "stored description was not preserved");
    const auto used_before = store.find("abc123")->last_used_ms;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    (void)store.load("abc123", false);
    check(store.find("abc123")->last_used_ms == used_before,
          "an untouched read refreshed the image's last use");
    (void)store.load("abc123");
    check(store.find("abc123")->last_used_ms > used_before, "a read did not refresh last use");
    check(!store.load("deadbeef").has_value(), "an unknown id loaded");
    check(!store.load("../escape").has_value(), "an unsafe id loaded");
}

void test_incremental_put_writes_only_new_chunks() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto head1 = pseudo_random(200, 1);
    const auto kv1   = pseudo_random(5000, 2);
    const auto image1 = concat({head1, kv1});
    (void)store.put(describe("a1"), image1, std::array{ContextStore::Region{200, 5000}});

    // The conversation grew: a different header, the same older pages, 3000 new bytes of pages.
    const auto head2  = pseudo_random(260, 9);
    const auto kv2    = concat({kv1, pseudo_random(3000, 3)});
    const auto image2 = concat({head2, kv2});
    const auto put    = store.put(describe("a2", 140), image2,
                                  std::array{ContextStore::Region{260, 8000}});
    // 4 whole 1024-byte chunks of the old pages are unchanged; the partial fifth chunk and the new
    // pages are written.
    check(put.chunk_bytes_reused == 4096 && put.chunk_bytes_written == 8000 - 4096,
          "a grown image rewrote chunks it already held");
    const auto loaded = store.load("a2");
    check(loaded && *loaded == image2, "grown image did not round trip");
    check(store.load("a1").value_or(std::vector<std::uint8_t>{}) == image1,
          "the earlier image no longer loads");
}

void test_restart_recovers_images() {
    TempDirectory directory;
    const auto image = concat({pseudo_random(100, 1), pseudo_random(3000, 2), pseudo_random(8, 3)});
    {
        ContextStore store(options_for(directory.path()));
        (void)store.put(describe("f00d"), image, std::array{ContextStore::Region{100, 3000}});
    }
    ContextStore reopened(options_for(directory.path()));
    const auto listed = reopened.list();
    check(listed.size() == 1 && listed[0].id == "f00d" && listed[0].checkpoints.size() == 1,
          "restart did not recover the stored image");
    const auto loaded = reopened.load("f00d");
    check(loaded && *loaded == image, "recovered image differs");
}

void test_corruption_is_a_miss_and_removed() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto image = concat({pseudo_random(64, 1), pseudo_random(3000, 2)});
    (void)store.put(describe("c0ffee"), image, std::array{ContextStore::Region{64, 3000}});
    const auto files = chunk_files(directory.path());
    check(!files.empty(), "no chunk files were written");
    {
        std::fstream file(files.front(), std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(0);
        char byte = 0;
        file.read(&byte, 1);
        file.seekp(0);
        byte ^= 0x55;
        file.write(&byte, 1);
    }
    check(!store.load("c0ffee").has_value(), "a damaged chunk was served");
    check(!store.find("c0ffee").has_value(), "a damaged image stayed listed");
    check(store.stats().corrupt_removed == 1, "a damaged image was not counted");

    // A truncated manifest is dropped at start-up and its chunks are swept.
    (void)store.put(describe("beef"), image, std::array{ContextStore::Region{64, 3000}});
    const fs::path manifest = directory.path() / "manifests" / "beef.manifest";
    fs::resize_file(manifest, fs::file_size(manifest) / 2);
    ContextStore reopened(options_for(directory.path()));
    check(reopened.list().empty() && chunk_files(directory.path()).empty(),
          "a truncated manifest was kept or left chunks behind");
}

void test_space_limit_evicts_least_recently_used() {
    TempDirectory directory;
    ContextStore::Options options = options_for(directory.path());
    std::int64_t now_ms            = 1'000'000;
    options.clock                  = [&] { return now_ms; };
    const auto base                = pseudo_random(6000, 1);
    {
        // Unbounded first, to learn the size of two overlapping images.
        ContextStore probe(options_for(directory.path() / "probe"));
        (void)probe.put(describe("a1"), concat({pseudo_random(50, 1), base}),
                        std::array{ContextStore::Region{50, 6000}});
        (void)probe.put(describe("a2"), concat({pseudo_random(60, 2), base, pseudo_random(2000, 3)}),
                        std::array{ContextStore::Region{60, 8000}});
        options.max_bytes = probe.stats().used_bytes - 100;
    }
    ContextStore store(options);
    (void)store.put(describe("a1"), concat({pseudo_random(50, 1), base}),
                    std::array{ContextStore::Region{50, 6000}});
    now_ms += 1000;
    const auto put = store.put(describe("a2"), concat({pseudo_random(60, 2), base, pseudo_random(2000, 3)}),
                               std::array{ContextStore::Region{60, 8000}});
    check(put.evicted_images == 1 && !store.find("a1") && store.find("a2"),
          "the least recently used image was not the one evicted");
    check(store.load("a2").has_value(), "chunks the survivor shares with the victim were removed");
    check(store.stats().used_bytes <= options.max_bytes, "the size limit was exceeded");
}

void test_ttl_and_use_extend_life() {
    TempDirectory directory;
    ContextStore::Options options = options_for(directory.path());
    std::int64_t now_ms            = 5'000'000;
    options.clock                  = [&] { return now_ms; };
    options.ttl                    = std::chrono::seconds(100);
    ContextStore store(options);
    const auto image = concat({pseudo_random(40, 1), pseudo_random(2000, 2)});
    (void)store.put(describe("aa"), image, std::array{ContextStore::Region{40, 2000}});
    (void)store.put(describe("bb"), concat({pseudo_random(40, 7), pseudo_random(2000, 8)}),
                    std::array{ContextStore::Region{40, 2000}});
    now_ms += 60'000;
    check(store.load("aa").has_value(), "image should still load inside its lifetime");
    now_ms += 60'000; // bb is 120 s old and unused; aa was used 60 s ago
    store.maintain();
    check(!store.find("bb") && store.find("aa"), "expiry removed the wrong image");
    check(store.stats().expired == 1, "expiry was not counted");

    // The use time survives a restart.
    ContextStore reopened(options);
    check(reopened.find("aa").has_value(), "an image used recently expired after restart");
    now_ms += 120'000;
    reopened.maintain();
    check(reopened.list().empty() && chunk_files(directory.path()).empty(),
          "expired images left chunks behind");
}

void test_supersede_and_replace() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto kv = pseudo_random(3000, 1);
    // Image 01 ends at frontier 99, with the key {0x1111, 0x2222} that describe() gives it.
    (void)store.put(describe("01"), concat({pseudo_random(30, 1), kv}),
                    std::array{ContextStore::Region{30, 3000}});
    // An unrelated image whose prefix digests do not reproduce that key supersedes nothing.
    std::vector<std::array<std::uint64_t, 2>> unrelated(130, {7, 7});
    auto stranger           = describe("03", 120);
    stranger.prefix_digests = unrelated;
    (void)store.put(stranger, concat({pseudo_random(40, 5), pseudo_random(500, 6)}),
                    std::array{ContextStore::Region{40, 500}});
    check(store.find("01") && store.find("03"), "an unrelated image superseded another");

    // The next state of the same conversation reproduces 01's endpoint key at frontier 99.
    std::vector<std::array<std::uint64_t, 2>> digests(130, {5, 5});
    digests[99]             = {0x1111, 0x2222};
    auto next               = describe("02", 120);
    next.prefix_digests     = digests;
    (void)store.put(next, concat({pseudo_random(40, 2), kv, pseudo_random(1000, 3)}),
                    std::array{ContextStore::Region{40, 4000}});
    check(!store.find("01") && store.find("02"), "a superseded image was kept");
    check(store.stats().superseded == 1, "supersession was not counted");
    check(store.load("02").has_value(), "the superseding image lost chunks it shared");
    (void)store.erase("03");

    // Replacing an id keeps its shared chunks and drops the ones only the old image used.
    (void)store.put(describe("02", 130), concat({pseudo_random(40, 9), pseudo_random(500, 5)}),
                    std::array{ContextStore::Region{40, 500}});
    check(store.load("02").has_value() && store.list().size() == 1, "replacing an id broke it");
    check(chunk_files(directory.path()).size() == 1, "a replaced image left unreferenced chunks");
}

void test_invalid_input_and_failed_put() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto image = pseudo_random(2000, 1);
    bool threw       = false;
    try {
        (void)store.put(describe("ab"), image,
                        std::array{ContextStore::Region{100, 500}, ContextStore::Region{400, 500}});
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "overlapping regions were accepted");
    threw = false;
    try {
        (void)store.put(describe("ab"), image, std::array{ContextStore::Region{1500, 1000}});
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a region past the end of the image was accepted");
    threw = false;
    try {
        (void)store.put(describe("not-hex!"), image, {});
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "an unsafe id was accepted");

    // The manifest cannot be published: nothing of the image may remain.
    fs::remove_all(directory.path() / "manifests");
    { std::ofstream blocker(directory.path() / "manifests"); }
    threw = false;
    try {
        (void)store.put(describe("cd"), image, std::array{ContextStore::Region{0, 2000}});
    } catch (const std::runtime_error&) { threw = true; }
    check(threw, "a put that could not publish its manifest reported success");
    check(chunk_files(directory.path()).empty() && store.list().empty() &&
              store.stats().put_failures == 1,
          "a failed put left chunks or an entry behind");
}

// An in-memory ObjectStore with switches for the failures a bucket has.
class MemoryObjectStore final : public ninfer::ObjectStore {
public:
    void put(const std::string& key, std::span<const std::uint8_t> bytes) override {
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        objects[key] = {std::vector<std::uint8_t>(bytes.begin(), bytes.end()), ++clock};
        ++puts;
    }
    std::optional<std::vector<std::uint8_t>> get(const std::string& key) override {
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        const auto found = objects.find(key);
        if (found == objects.end()) { return std::nullopt; }
        return found->second.first;
    }
    bool exists(const std::string& key) override {
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        return objects.find(key) != objects.end();
    }
    std::vector<ninfer::ObjectInfo> list(const std::string& prefix) override {
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        std::vector<ninfer::ObjectInfo> out;
        for (const auto& [key, value] : objects) {
            if (key.rfind(prefix, 0) == 0) {
                out.push_back({key, value.first.size(), value.second});
            }
        }
        return out;
    }
    bool touch(const std::string& key) override {
        std::scoped_lock lock(mutex);
        const auto found = objects.find(key);
        if (found == objects.end()) { return false; }
        found->second.second = ++clock;
        ++touches;
        return true;
    }
    void remove(const std::string& key) override {
        std::scoped_lock lock(mutex);
        objects.erase(key);
    }

    std::mutex mutex;
    std::map<std::string, std::pair<std::vector<std::uint8_t>, std::int64_t>> objects;
    std::int64_t clock = 1'000'000'000'000;
    bool down          = false;
    int puts           = 0;
    int touches        = 0;
};

ContextStore::Options remote_options(const fs::path& directory,
                                     const std::shared_ptr<MemoryObjectStore>& remote) {
    ContextStore::Options options = options_for(directory);
    options.remote                = remote;
    options.remote_prefix         = "ninfer/";
    options.remote_refresh        = std::chrono::seconds(3600); // tests refresh explicitly
    options.clock                 = [remote] { return remote->clock; };
    return options;
}

bool drained(ContextStore& store) {
    return store.flush_remote(std::chrono::steady_clock::now() + std::chrono::seconds(10));
}

void test_remote_upload_and_second_engine() {
    TempDirectory first_directory;
    TempDirectory second_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    const auto kv    = pseudo_random(5000, 4);
    const auto image = concat({pseudo_random(30, 1), kv});
    {
        ContextStore store(remote_options(first_directory.path(), remote));
        (void)store.put(describe("0a"), image, std::array{ContextStore::Region{30, 5000}});
        check(drained(store), "the upload queue did not drain");
        const auto stats = store.stats();
        check(stats.remote_uploads == 6 && stats.remote_upload_failures == 0,
              "a put did not upload its chunks and manifest");
        // Writing the next state of the conversation uploads only what is new.
        const int before = remote->puts;
        (void)store.put(describe("0a", 120), concat({pseudo_random(30, 1), kv, pseudo_random(500, 8)}),
                        std::array{ContextStore::Region{30, 5500}});
        check(drained(store), "the second upload did not drain");
        check(remote->puts - before == 3, "a rewritten image uploaded chunks the bucket already had");
    }
    // A second engine with an empty directory sees the image and fetches it when asked.
    ContextStore other(remote_options(second_directory.path(), remote));
    check(other.refresh_remote() == 1 && other.list().size() == 1, "the remote image was not listed");
    check(!other.list()[0].local, "an image not fetched yet reports itself local");
    check(other.stats().remote_images == 1, "the remote-only image was not counted");
    check(!other.list()[0].checkpoints.empty() && other.list()[0].checkpoints[0].digests[0] == 0x1111,
          "the remote image lost its checkpoint keys");
    const auto loaded = other.load("0a");
    check(loaded && *loaded == concat({pseudo_random(30, 1), kv, pseudo_random(500, 8)}),
          "an image fetched from the remote differs from what was stored");
    check(other.list()[0].local && other.stats().remote_images == 0, "a fetched image is still remote");
    check(other.stats().remote_downloads == 6, "the fetch did not count its chunks");
}

void test_remote_survives_local_eviction() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore::Options options = remote_options(directory.path(), remote);
    options.max_bytes             = 8000; // room for one image
    ContextStore store(options);
    const auto a = concat({pseudo_random(30, 1), pseudo_random(5000, 2)});
    const auto b = concat({pseudo_random(30, 3), pseudo_random(5000, 4)});
    (void)store.put(describe("0a"), a, std::array{ContextStore::Region{30, 5000}});
    check(drained(store), "upload a");
    remote->clock += 10;
    (void)store.put(describe("0b"), b, std::array{ContextStore::Region{30, 5000}});
    check(drained(store), "upload b");
    check(!store.list().empty() && store.stats().evicted_for_space >= 1, "the local limit evicted nothing");
    // 0a was evicted locally; the remote still holds it and a refresh brings it back.
    (void)store.refresh_remote();
    bool found = false;
    for (const auto& info : store.list()) {
        if (info.id == "0a") { found = true; }
    }
    check(found, "an image evicted locally was not found again through the remote");
    const auto back = store.load("0a");
    check(back && *back == a, "an image evicted locally could not be fetched back");
    check(store.stats().remote_downloads > 0, "the evicted image was not fetched from the remote");
}

void test_remote_damage_and_outage_are_misses() {
    TempDirectory directory;
    TempDirectory other_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    {
        ContextStore store(remote_options(directory.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                        std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload before damage");
    }
    // A chunk in the bucket is damaged: the load is a miss and the image is not offered again.
    for (auto& [key, value] : remote->objects) {
        if (key.find("/chunks/") != std::string::npos) {
            value.first[0] ^= 0xff;
            break;
        }
    }
    ContextStore other(remote_options(other_directory.path(), remote));
    (void)other.refresh_remote();
    check(other.list().size() == 1, "the image was not listed before the damage was found");
    check(!other.load("0a").has_value(), "a damaged remote chunk produced an image");
    check(other.list().empty(), "a damaged remote image is still offered");
    check(other.refresh_remote() == 0, "a damaged remote image was imported again");

    // An unreachable bucket costs the upload, never the local write.
    TempDirectory outage_directory;
    auto down = std::make_shared<MemoryObjectStore>();
    down->down = true;
    ContextStore offline(remote_options(outage_directory.path(), down));
    (void)offline.put(describe("0c"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                      std::array{ContextStore::Region{30, 3000}});
    check(drained(offline), "the upload queue did not drain during an outage");
    check(offline.load("0c").has_value(), "an outage lost a locally stored image");
    check(offline.stats().remote_upload_failures >= 1 && offline.stats().remote_uploads == 0,
          "an outage was not counted");
    // The bucket comes back: the next write of the image uploads it all.
    down->down = false;
    (void)offline.put(describe("0c"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                      std::array{ContextStore::Region{30, 3000}});
    check(drained(offline) && offline.stats().remote_uploads == 4, "the recovered upload was incomplete");
}

void test_remote_use_restarts_the_remote_age() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore store(remote_options(directory.path(), remote));
    (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                    std::array{ContextStore::Region{30, 3000}});
    check(drained(store), "upload");
    check(remote->touches == 0, "a fresh upload was touched");
    remote->clock += 2 * 24 * 3600 * 1000LL;
    check(store.load("0a").has_value(), "load after a day");
    check(drained(store), "touch drained");
    check(remote->touches == 4, "using an image did not restart the age of its remote objects");
}

} // namespace

int main() {
    test_hash();
    test_round_trip();
    test_incremental_put_writes_only_new_chunks();
    test_restart_recovers_images();
    test_corruption_is_a_miss_and_removed();
    test_space_limit_evicts_least_recently_used();
    test_ttl_and_use_extend_life();
    test_supersede_and_replace();
    test_invalid_input_and_failed_put();
    test_remote_upload_and_second_engine();
    test_remote_survives_local_eviction();
    test_remote_damage_and_outage_are_misses();
    test_remote_use_restarts_the_remote_age();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
