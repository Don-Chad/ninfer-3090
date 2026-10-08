#include "product/object_store/s3_object_store.h"

#include <curl/curl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninfer::product {
namespace {

void ensure_curl_initialized() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

struct CurlHandle {
    CURL* handle = nullptr;
    curl_slist* headers = nullptr;
    CurlHandle() : handle(curl_easy_init()) {
        if (handle == nullptr) { throw std::runtime_error("cannot create a curl handle"); }
    }
    ~CurlHandle() {
        if (headers != nullptr) { curl_slist_free_all(headers); }
        curl_easy_cleanup(handle);
    }
    CurlHandle(const CurlHandle&)            = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
    void add_header(const std::string& line) { headers = curl_slist_append(headers, line.c_str()); }
};

std::size_t write_to_buffer(char* data, std::size_t size, std::size_t count, void* user) {
    auto* buffer = static_cast<std::vector<std::uint8_t>*>(user);
    const std::size_t bytes = size * count;
    buffer->insert(buffer->end(), reinterpret_cast<std::uint8_t*>(data),
                   reinterpret_cast<std::uint8_t*>(data) + bytes);
    return bytes;
}

struct UploadSource {
    std::span<const std::uint8_t> bytes;
    std::size_t position = 0;
};

std::size_t read_from_span(char* out, std::size_t size, std::size_t count, void* user) {
    auto* source = static_cast<UploadSource*>(user);
    const std::size_t want = std::min(size * count, source->bytes.size() - source->position);
    if (want != 0) { std::memcpy(out, source->bytes.data() + source->position, want); }
    source->position += want;
    return want;
}

// RFC 3986 unreserved characters pass; everything else, '/' only when `keep_slash` is false, is
// percent-encoded, as SigV4 canonical URIs require.
std::string url_encode(const std::string& text, bool keep_slash) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(text.size());
    for (const unsigned char c : text) {
        const bool plain = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                           (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~' ||
                           (keep_slash && c == '/');
        if (plain) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4U]);
            out.push_back(hex[c & 15U]);
        }
    }
    return out;
}

std::string xml_decode(std::string text) {
    const std::pair<const char*, const char*> entities[] = {
        {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}, {"&amp;", "&"}};
    for (const auto& [from, to] : entities) {
        for (std::size_t at = text.find(from); at != std::string::npos; at = text.find(from, at)) {
            text.replace(at, std::strlen(from), to);
            at += std::strlen(to);
        }
    }
    return text;
}

// The text between <tag> and </tag> at or after `from`, advancing `from` past it.
bool xml_field(const std::string& xml, const char* tag, std::size_t& from, std::string& out,
               std::size_t limit = std::string::npos) {
    const std::string open  = std::string("<") + tag + ">";
    const std::string close = std::string("</") + tag + ">";
    const std::size_t begin = xml.find(open, from);
    if (begin == std::string::npos || begin >= limit) { return false; }
    const std::size_t end = xml.find(close, begin + open.size());
    if (end == std::string::npos) { return false; }
    out  = xml.substr(begin + open.size(), end - begin - open.size());
    from = end + close.size();
    return true;
}

class S3ObjectStore final : public ObjectStore {
public:
    explicit S3ObjectStore(S3Config config) : config_(std::move(config)) {
        ensure_curl_initialized();
        if (config_.endpoint.empty() || config_.bucket.empty() || config_.access_key.empty() ||
            config_.secret_key.empty()) {
            throw std::invalid_argument("S3 needs an endpoint, a bucket and credentials");
        }
        while (config_.endpoint.size() > 1 && config_.endpoint.back() == '/') {
            config_.endpoint.pop_back();
        }
        if (config_.attempts < 1) { config_.attempts = 1; }
    }

    void put(const std::string& key, std::span<const std::uint8_t> bytes) override {
        Response response = send("PUT", object_url(key), {}, bytes, true);
        require_success(response, "put " + key);
    }

    std::optional<std::vector<std::uint8_t>> get(const std::string& key) override {
        Response response = send("GET", object_url(key), {}, {}, false);
        if (response.status == 404) { return std::nullopt; }
        require_success(response, "get " + key);
        return std::move(response.body);
    }

    bool exists(const std::string& key) override {
        Response response = send("HEAD", object_url(key), {}, {}, false);
        if (response.status == 404) { return false; }
        require_success(response, "head " + key);
        return true;
    }

    std::vector<ObjectInfo> list(const std::string& prefix) override {
        std::vector<ObjectInfo> out;
        std::string token;
        for (;;) {
            // Query parameters in canonical (sorted) order.
            std::string query = "?";
            if (!token.empty()) { query += "continuation-token=" + url_encode(token, false) + "&"; }
            query += "list-type=2&prefix=" + url_encode(prefix, false);
            Response response = send("GET", bucket_url() + query, {}, {}, false);
            require_success(response, "list " + prefix);
            const std::string xml(response.body.begin(), response.body.end());
            std::size_t cursor = 0;
            for (;;) {
                const std::size_t contents = xml.find("<Contents>", cursor);
                if (contents == std::string::npos) { break; }
                const std::size_t contents_end = xml.find("</Contents>", contents);
                if (contents_end == std::string::npos) { break; }
                ObjectInfo info;
                std::size_t at = contents;
                std::string text;
                if (xml_field(xml, "Key", at, text, contents_end)) { info.key = xml_decode(text); }
                at = contents;
                if (xml_field(xml, "Size", at, text, contents_end)) {
                    info.size = std::strtoull(text.c_str(), nullptr, 10);
                }
                at = contents;
                if (xml_field(xml, "LastModified", at, text, contents_end)) {
                    info.modified_ms = parse_s3_timestamp_ms(text);
                }
                if (!info.key.empty()) { out.push_back(std::move(info)); }
                cursor = contents_end + 1;
            }
            std::size_t at = 0;
            std::string truncated;
            std::string next;
            const bool more = xml_field(xml, "IsTruncated", at, truncated) && truncated == "true";
            at              = 0;
            if (!more || !xml_field(xml, "NextContinuationToken", at, next) || next.empty()) {
                break;
            }
            token = xml_decode(next);
        }
        return out;
    }

    bool touch(const std::string& key) override {
        // A copy of the object onto itself, which S3 accepts only when something about it changes:
        // the metadata is replaced. The bytes do not travel, and the object's age starts again.
        std::vector<std::string> headers{
            "x-amz-copy-source: /" + url_encode(config_.bucket, false) + "/" + url_encode(key, true),
            "x-amz-metadata-directive: REPLACE",
            "x-amz-meta-touched: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                        std::chrono::system_clock::now().time_since_epoch())
                                                        .count())};
        Response response = send("PUT", object_url(key), headers, {}, true);
        if (response.status == 404) { return false; }
        require_success(response, "touch " + key);
        const std::string body(response.body.begin(), response.body.end());
        // S3 can answer 200 and carry the failure in the body.
        if (body.find("<Error>") != std::string::npos) {
            if (body.find("NoSuchKey") != std::string::npos) { return false; }
            throw std::runtime_error("S3 touch " + key + " failed: " + body.substr(0, 200));
        }
        return true;
    }

    void remove(const std::string& key) override {
        Response response = send("DELETE", object_url(key), {}, {}, false);
        if (response.status == 404) { return; }
        require_success(response, "remove " + key);
    }

private:
    struct Response {
        long status = 0;
        std::vector<std::uint8_t> body;
    };

    [[nodiscard]] std::string bucket_url() const {
        return config_.endpoint + "/" + url_encode(config_.bucket, false);
    }
    [[nodiscard]] std::string object_url(const std::string& key) const {
        return bucket_url() + "/" + url_encode(key, true);
    }

    static void require_success(const Response& response, const std::string& what) {
        if (response.status >= 200 && response.status < 300) { return; }
        std::string detail(response.body.begin(),
                           response.body.begin() + static_cast<std::ptrdiff_t>(
                                                       std::min<std::size_t>(response.body.size(), 200)));
        throw std::runtime_error("S3 " + what + " failed with HTTP " + std::to_string(response.status) +
                                 (detail.empty() ? std::string() : ": " + detail));
    }

    Response send(const char* method, const std::string& url,
                  const std::vector<std::string>& extra_headers,
                  std::span<const std::uint8_t> upload, bool is_upload) {
        std::string last_error;
        for (int attempt = 0; attempt < config_.attempts; ++attempt) {
            if (attempt != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250 << attempt));
            }
            CurlHandle curl;
            Response response;
            UploadSource source{upload, 0};
            CURL* h = curl.handle;
            curl_easy_setopt(h, CURLOPT_URL, url.c_str());
            curl_easy_setopt(h, CURLOPT_PROTOCOLS_STR, "http,https");
            curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 0L);
            curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, static_cast<long>(config_.connect_timeout.count()));
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1024L);
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, static_cast<long>(config_.stall_timeout.count()));
            curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 1L);
            curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 2L);
            const std::string signature = "aws:amz:" + config_.region + ":s3";
            const std::string credentials = config_.access_key + ":" + config_.secret_key;
            curl_easy_setopt(h, CURLOPT_AWS_SIGV4, signature.c_str());
            curl_easy_setopt(h, CURLOPT_USERPWD, credentials.c_str());
            curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_buffer);
            curl_easy_setopt(h, CURLOPT_WRITEDATA, &response.body);
            const std::string verb = method;
            if (verb == "HEAD") {
                curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
            } else if (verb == "DELETE") {
                curl_easy_setopt(h, CURLOPT_CUSTOMREQUEST, "DELETE");
            } else if (is_upload) {
                curl_easy_setopt(h, CURLOPT_UPLOAD, 1L);
                curl_easy_setopt(h, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(upload.size()));
                curl_easy_setopt(h, CURLOPT_READFUNCTION, read_from_span);
                curl_easy_setopt(h, CURLOPT_READDATA, &source);
            }
            // The payload is not part of the signature (it travels inside TLS when the endpoint is
            // https), and no 100-continue round trip precedes an upload.
            curl.add_header("x-amz-content-sha256: UNSIGNED-PAYLOAD");
            curl.add_header("Expect:");
            if (!config_.session_token.empty()) {
                curl.add_header("x-amz-security-token: " + config_.session_token);
            }
            for (const std::string& line : extra_headers) { curl.add_header(line); }
            curl_easy_setopt(h, CURLOPT_HTTPHEADER, curl.headers);

            const CURLcode code = curl_easy_perform(h);
            if (code != CURLE_OK) {
                last_error = curl_easy_strerror(code);
                continue;
            }
            curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &response.status);
            if (response.status >= 500) {
                last_error = "HTTP " + std::to_string(response.status);
                continue;
            }
            return response;
        }
        throw std::runtime_error("S3 request failed: " + last_error);
    }

    S3Config config_;
};

} // namespace

std::shared_ptr<ObjectStore> make_s3_object_store(S3Config config) {
    return std::make_shared<S3ObjectStore>(std::move(config));
}

std::int64_t parse_s3_timestamp_ms(const std::string& text) {
    int year = 0, month = 0, day = 0, hour = 0, minute = 0;
    double seconds = 0.0;
    if (std::sscanf(text.c_str(), "%d-%d-%dT%d:%d:%lf", &year, &month, &day, &hour, &minute,
                    &seconds) != 6 ||
        month < 1 || month > 12 || day < 1 || day > 31) {
        return -1;
    }
    // Days from 1970-01-01 (Howard Hinnant's civil-date algorithm).
    year -= month <= 2 ? 1 : 0;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(year - era * 400);
    const unsigned doy = (153U * static_cast<unsigned>(month + (month > 2 ? -3 : 9)) + 2U) / 5U +
                         static_cast<unsigned>(day) - 1U;
    const unsigned doe = yoe * 365U + yoe / 4U - yoe / 100U + doy;
    const std::int64_t days = static_cast<std::int64_t>(era) * 146097 + static_cast<std::int64_t>(doe) - 719468;
    return ((days * 24 + hour) * 60 + minute) * 60000 + static_cast<std::int64_t>(seconds * 1000.0 + 0.5);
}

} // namespace ninfer::product
