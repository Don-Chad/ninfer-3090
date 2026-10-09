// The S3 object store client. The timestamp parser is checked always; the round trip, listing across
// pages, special characters, touch and remove run against the S3-compatible server named by
// NINFER_TEST_S3_ENDPOINT and NINFER_TEST_S3_BUCKET (credentials from AWS_ACCESS_KEY_ID and
// AWS_SECRET_ACCESS_KEY, "test" if unset), for example a local moto or MinIO, and are skipped
// without them.
#include "product/object_store/s3_object_store.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using ninfer::product::S3Config;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << '\n';
    }
}

int main(int, char**) {
    check(ninfer::product::parse_s3_timestamp_ms("2026-10-08T19:01:45.000Z") == 1791486105000LL,
          "S3 timestamp with fraction");
    check(ninfer::product::parse_s3_timestamp_ms("1970-01-01T00:00:01Z") == 1000, "S3 timestamp epoch");
    check(ninfer::product::parse_s3_timestamp_ms("not a time") == -1, "S3 timestamp garbage");
    const char* endpoint = std::getenv("NINFER_TEST_S3_ENDPOINT");
    const char* bucket   = std::getenv("NINFER_TEST_S3_BUCKET");
    if (failures != 0) { return 1; }
    if (endpoint == nullptr || bucket == nullptr) {
        std::cout << "skipping the server round trip: NINFER_TEST_S3_ENDPOINT / BUCKET not set\n";
        return 77;
    }
    S3Config config;
    config.endpoint   = endpoint;
    config.bucket     = bucket;
    config.access_key = std::getenv("AWS_ACCESS_KEY_ID") ? std::getenv("AWS_ACCESS_KEY_ID") : "test";
    config.secret_key = std::getenv("AWS_SECRET_ACCESS_KEY") ? std::getenv("AWS_SECRET_ACCESS_KEY") : "test";
    auto store = ninfer::product::make_s3_object_store(config);

    std::vector<std::uint8_t> big(5 * 1024 * 1024 + 17);
    for (std::size_t i = 0; i < big.size(); ++i) { big[i] = static_cast<std::uint8_t>(i * 31 + 7); }
    const std::string key = "probe/manifests/ab.manifest";
    try {
        check(!store->exists(key), "object exists before put");
        check(!store->get(key).has_value(), "get of an absent object");
        store->put(key, big);
        check(store->exists(key), "object missing after put");
        const auto back = store->get(key);
        check(back && *back == big, "round trip differs");
        store->put("probe/chunks/ab/with space & plus+.chunk", std::vector<std::uint8_t>{1, 2, 3});
        for (int i = 0; i < 1100; ++i) {
            store->put("probe/many/" + std::to_string(i), std::vector<std::uint8_t>{1});
        }
        const auto listed = store->list("probe/");
        check(listed.size() == 1102, "listing across pages lost objects");
        bool found = false;
        for (const auto& item : listed) {
            if (item.key == "probe/chunks/ab/with space & plus+.chunk") { found = item.size == 3; }
            check(item.modified_ms > 1'600'000'000'000LL, "modified time not parsed");
        }
        check(found, "a key with special characters did not round trip through listing");
        check(store->touch(key), "touch of an existing object");
        check(!store->touch("probe/nothing-here"), "touch of an absent object reported success");
        const auto after_touch = store->get(key);
        check(after_touch && *after_touch == big, "touch changed the bytes");
        store->remove(key);
        check(!store->exists(key), "object exists after remove");
        store->remove(key); // absent: not an error
        std::cout << "listed " << listed.size() << " objects\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "EXCEPTION " << error.what() << '\n';
    }
    std::cout << (failures == 0 ? "ok" : "FAILED") << '\n';
    return failures == 0 ? 0 : 1;
}
