#pragma once

// GET /slots: llama.cpp-shaped listing of the private context-cache cells, with the session
// persistence and usage facts a client needs to decide which sessions are worth keeping.

#include "ninfer/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

// Renders the /slots JSON body: one entry per cell, `id` being the cell's index.
[[nodiscard]] std::string make_slots_report(const std::vector<ninfer::SlotState>& states,
                                            std::uint32_t max_context, bool speculative);

} // namespace ninfer::serve
