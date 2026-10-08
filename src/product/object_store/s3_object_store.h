#pragma once

#include "ninfer/object_store.h"

#include <chrono>
#include <memory>
#include <string>

namespace ninfer::product {

struct S3Config {
    // "https://s3.eu-west-2.amazonaws.com", or the address of any S3-compatible server (MinIO,
    // Cloudflare R2, Backblaze B2, a local test server). Path-style addressing: the bucket is the
    // first path segment, so no wildcard DNS or per-bucket certificate is needed.
    std::string endpoint;
    std::string bucket;
    std::string region = "us-east-1";
    std::string access_key;
    std::string secret_key;
    std::string session_token; // optional, for temporary credentials
    std::chrono::seconds connect_timeout{10};
    // A transfer that moves less than 1 KiB/s for this long is abandoned; there is no total timeout,
    // so a large chunk on a slow link still finishes.
    std::chrono::seconds stall_timeout{60};
    int attempts = 3; // per call, for connection errors and 5xx responses
};

// An ObjectStore over the S3 REST API, signed with AWS Signature V4 by libcurl. Each call uses its
// own connection handle, so the object is safe to share between threads.
[[nodiscard]] std::shared_ptr<ObjectStore> make_s3_object_store(S3Config config);

// Milliseconds since the Unix epoch of an S3 timestamp such as "2026-10-08T19:01:45.000Z"; -1 when
// the text is not one. Exposed for tests.
[[nodiscard]] std::int64_t parse_s3_timestamp_ms(const std::string& text);

} // namespace ninfer::product
