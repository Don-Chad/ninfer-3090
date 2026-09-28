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

// Returns the validated filename, or nullopt when the name is empty, too long, dot-leading, holds
// anything outside [A-Za-z0-9._-], or names a Windows device (CON, NUL, COM1, ...) whatever its
// extension. A leading-dot ban removes ".", ".." and hidden files in one rule; the allowlist keeps
// every separator out.
[[nodiscard]] inline std::optional<std::string> sanitize_slot_filename(std::string_view name) {
    if (name.empty() || name.size() > kSlotFilenameMaxBytes || name.front() == '.') {
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
    return std::string(name);
}

} // namespace ninfer::serve
