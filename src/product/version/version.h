#pragma once

#include <string_view>

namespace ninfer::product {

// The release this binary was built from: the `VERSION` file at build time, e.g.
// "0.14.3-rtx3090". A tree that is not exactly the tagged release appends "+<short commit>" (and
// "-dirty" with uncommitted tracked changes); a build without git metadata, such as a
// `git archive` snapshot, reports `VERSION` alone.
std::string_view build_version();

} // namespace ninfer::product
