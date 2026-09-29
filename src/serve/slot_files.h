#pragma once

// Filename policy for /slots session persistence. Clients name snapshot files and the server
// confines them to the --slot-save-path directory, so a name is one conservative path component.

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

inline constexpr std::size_t kSlotFilenameMaxBytes = 128;

// Returns the canonical filename, or nullopt when the name is empty, too long, starts or ends with
// a dot, holds anything outside [A-Za-z0-9._-], or names a Windows device (CON, NUL, COM1, ...)
// whatever its extension. A leading-dot ban removes ".", ".." and hidden files in one rule; the
// allowlist keeps every separator out.
//
// The canonical name is lowercase and the trailing-dot ban is absolute, so two accepted names
// reach the same file exactly when they are the same string, on case-insensitive filesystems and
// under Windows' trailing-dot stripping alike. The Engine keys the slot-file binding and the spill
// high-water marks on the path string, and relies on that.
[[nodiscard]] inline std::optional<std::string> sanitize_slot_filename(std::string_view name) {
    if (name.empty() || name.size() > kSlotFilenameMaxBytes || name.front() == '.' ||
        name.back() == '.') {
        return std::nullopt;
    }
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) { return std::nullopt; }
    }
    std::string stem(name.substr(0, name.find('.')));
    for (char& c : stem) {
        if (c >= 'a' && c <= 'z') { c = static_cast<char>(c - 'a' + 'A'); }
    }
    static constexpr std::array<std::string_view, 4> kDevices{"CON", "PRN", "AUX", "NUL"};
    for (const std::string_view device : kDevices) {
        if (stem == device) { return std::nullopt; }
    }
    if (stem.size() == 4 && (stem.starts_with("COM") || stem.starts_with("LPT")) &&
        stem[3] >= '0' && stem[3] <= '9') {
        return std::nullopt;
    }
    std::string canonical(name);
    for (char& c : canonical) {
        if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c - 'A' + 'a'); }
    }
    return canonical;
}

} // namespace ninfer::serve
