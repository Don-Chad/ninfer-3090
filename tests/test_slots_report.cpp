#include "serve/slots_report.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // Cell 0 is empty, cell 1 holds a bound conversation that has been continued twice, cell 2 a
    // session nothing has saved, cell 3 is being published into by a request.
    std::vector<ninfer::SlotState> states(4);
    states[1].retained           = true;
    states[1].prompt_tokens      = 94934;
    states[1].cached_tokens      = 94934;
    states[1].session_digest     = "649c7ea309307025";
    states[1].checkpoints        = {{.frontier = 94933, .session_digest = "aa"}};
    states[1].last_used_unix_ms  = 1791292927534;
    states[1].reuse_count        = 2;
    states[1].reused_tokens      = std::numeric_limits<std::uint64_t>::max() - 1;
    states[2].retained           = true;
    states[2].prompt_tokens      = 100;
    states[2].cached_tokens      = 100;
    states[2].session_digest     = "bb";
    states[3].processing         = true;
    states[3].prompt_tokens      = 5000;
    states[3].cached_tokens      = 4000;

    const Json slots = Json::parse(make_slots_report(states, 262144, true));
    failures += check(slots.is_array() && slots.size() == 4, "one entry per cell");
    failures += check(slots.at(1).at("id") == 1, "id is the cell index");

    // What was reported before is unchanged.
    failures += check(slots.at(1).at("retained") == true && slots.at(1).at("is_processing") == false,
                      "retained, not processing");
    failures += check(slots.at(1).at("session_digest") == "649c7ea309307025", "session digest");
    failures += check(slots.at(1).at("n_prompt_tokens") == 94934 &&
                          slots.at(1).at("n_prompt_tokens_cache") == 94934,
                      "retained depth is both token counts");
    failures += check(slots.at(1).at("checkpoints").at(0).at("frontier") == 94933, "checkpoints");
    failures += check(slots.at(1).at("n_ctx") == 262144 && slots.at(1).at("speculative") == true,
                      "context and speculative flags");
    failures += check(slots.at(3).at("is_processing") == true &&
                          slots.at(3).at("n_prompt_tokens") == 5000 &&
                          slots.at(3).at("n_prompt_tokens_cache") == 4000,
                      "a cell being published into reports its request");

    // The usage of a retained session.
    failures += check(slots.at(1).at("last_used_unix_ms") == 1791292927534ULL, "last used");
    failures += check(slots.at(1).at("reuse_count") == 2, "reuse count");
    failures += check(slots.at(1).at("reused_tokens") == std::numeric_limits<std::uint64_t>::max() - 1,
                      "reused tokens keep their full 64-bit value");

    failures += check(slots.at(2).at("reuse_count") == 0 && slots.at(2).at("last_used_unix_ms") == 0,
                      "a session nothing has used yet reports zeros");

    // Empty and in-flight cells have no session to describe.
    for (const std::size_t cell : {std::size_t{0}, std::size_t{3}}) {
        failures += check(slots.at(cell).at("last_used_unix_ms").is_null() &&
                              slots.at(cell).at("reuse_count").is_null() &&
                              slots.at(cell).at("reused_tokens").is_null(),
                          "a cell without a retained session reports null usage");
    }

    failures += check(Json::parse(make_slots_report({}, 8, false)).empty(), "no cells, no entries");
    failures += check(Json::parse(make_slots_report(states, 8, false)).at(0).at("speculative") == false,
                      "speculative flag follows the server");

    return failures;
}
