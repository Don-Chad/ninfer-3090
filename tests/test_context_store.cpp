#include "runtime/engine/context_store/context_store.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
