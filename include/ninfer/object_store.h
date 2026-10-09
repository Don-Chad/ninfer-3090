#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

struct ObjectInfo {
    std::string key;
    std::uint64_t size   = 0;
    std::int64_t modified_ms = 0; // Unix milliseconds
};

// A flat key/value object store: an S3-compatible bucket in production, a map in tests. Every call
// may block on the network. An absent object is nullopt/false, never an error; transport and server
// failures throw std::runtime_error. Implementations are safe for concurrent calls.
class ObjectStore {
public:
    virtual ~ObjectStore() = default;

    virtual void put(const std::string& key, std::span<const std::uint8_t> bytes) = 0;
    [[nodiscard]] virtual std::optional<std::vector<std::uint8_t>> get(const std::string& key) = 0;
    [[nodiscard]] virtual bool exists(const std::string& key) = 0;
    // Every object whose key starts with `prefix`, in no particular order.
    [[nodiscard]] virtual std::vector<ObjectInfo> list(const std::string& prefix) = 0;
    // Restarts the object's age for lifecycle expiry without transferring its bytes. False when the
    // object is absent.
    virtual bool touch(const std::string& key) = 0;
    virtual void remove(const std::string& key) = 0;

    // Makes every call in progress, and every later one, fail promptly instead of waiting on the
    // network. For shutdown: the store is not used afterwards. May be called from any thread.
    virtual void interrupt() noexcept {}
};

} // namespace ninfer
