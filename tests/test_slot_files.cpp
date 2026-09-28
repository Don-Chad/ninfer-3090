#include "serve/slot_files.h"

#include <iostream>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninfer::serve::kSlotFilenameMaxBytes;
    using ninfer::serve::sanitize_slot_filename;

    int failures = 0;

    failures += check(sanitize_slot_filename("session.bin") == "session.bin",
                      "plain filename was rejected");
    failures += check(sanitize_slot_filename("A-1_b.2") == "A-1_b.2",
                      "allowlisted punctuation was rejected");
    failures += check(sanitize_slot_filename(std::string(kSlotFilenameMaxBytes, 'a')).has_value(),
                      "maximum-length filename was rejected");

    failures += check(!sanitize_slot_filename("").has_value(), "empty filename was accepted");
    failures += check(!sanitize_slot_filename(std::string(kSlotFilenameMaxBytes + 1, 'a')),
                      "oversized filename was accepted");
    failures += check(!sanitize_slot_filename(".."), "dot-dot filename was accepted");
    failures += check(!sanitize_slot_filename(".hidden"), "dot-leading filename was accepted");
    failures += check(!sanitize_slot_filename("a/b"), "path separator was accepted");
    failures += check(!sanitize_slot_filename("a\\b"), "backslash was accepted");
    failures += check(!sanitize_slot_filename("a b"), "space was accepted");
    failures += check(!sanitize_slot_filename(std::string("a\0b", 3)), "NUL byte was accepted");
    failures += check(!sanitize_slot_filename("s\xc3\xa9ssion"), "non-ASCII byte was accepted");

    // Windows opens a device for these names whatever their extension or case.
    for (const char* device : {"CON", "nul", "Aux.bin", "prn.snapshot", "COM1", "lpt9.bin"}) {
        failures += check(!sanitize_slot_filename(device), "a Windows device name was accepted");
    }
    failures += check(sanitize_slot_filename("console.bin").has_value() &&
                          sanitize_slot_filename("COM10").has_value() &&
                          sanitize_slot_filename("nullable").has_value(),
                      "a name that merely starts like a device was rejected");

    if (failures != 0) {
        std::cerr << failures << " slot filename checks failed\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
