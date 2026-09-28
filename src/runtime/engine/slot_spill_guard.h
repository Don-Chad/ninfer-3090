#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace ninfer {

// Keeps an auto-save spill from rolling a slot file back to an older, shallower state of the same
// session. The catalog has no per-session identity, so a conversation's live continuation and a
// stale earlier copy left in another cell by a retaining restore look identical to the Engine;
// whichever is evicted spills to the file it is bound to.
//
// The guard is a per-path high-water mark in process memory. Every explicit save and restore is
// authoritative and sets the mark, since a client may legitimately save a rewound, shallower
// session. A spill proceeds only when it carries at least as many tokens as the mark; equal depth
// is allowed, because a restored-then-evicted session spills itself back unchanged. Every bound
// path was bound by a save or restore in this process, so the mark is primed before any spill can
// target it.
class SlotSpillGuard {
public:
    void note_authoritative(const std::string& path, std::uint32_t tokens) {
        std::scoped_lock lock(mutex_);
        depth_[path] = tokens;
    }

    // nullopt when the spill may be written; otherwise the deeper token count on record that the
    // spill would overwrite.
    [[nodiscard]] std::optional<std::uint32_t> blocks(const std::string& path,
                                                      std::uint32_t tokens) const {
        std::scoped_lock lock(mutex_);
        const auto it = depth_.find(path);
        if (it == depth_.end() || tokens >= it->second) { return std::nullopt; }
        return it->second;
    }

    void note_spilled(const std::string& path, std::uint32_t tokens) {
        std::scoped_lock lock(mutex_);
        std::uint32_t& depth = depth_[path];
        if (tokens > depth) { depth = tokens; }
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::uint32_t> depth_;
};

} // namespace ninfer
