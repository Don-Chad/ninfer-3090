#pragma once

#include "qus/model/config.h"

#include <cstdint>
#include <string>

namespace qus::serve {

// Upper bound for the auto-derived default output length. Requests that omit
// max_tokens get min(max_context/2, this); half the window is reserved for the
// prompt, and this ceiling keeps a no-limit client from generating for minutes.
inline constexpr int kDefaultMaxTokensCeiling = 8192;

struct ServeOptions {
    bool help_requested = false;
    std::string weights_path;
    std::string tokenizer_path;
    std::string host          = "127.0.0.1";
    int port                  = 8080;
    std::string api_key;                       // empty => no auth
    std::string model_id      = "qwen3.6-27b"; // reported by /v1/models
    std::uint32_t max_context = 8192;
    std::uint32_t prefill_chunk = model::kDefaultPrefillChunk;
    int device                = 0;
    int mtp_draft_tokens      = 0;
    bool use_cuda_graph       = true;
    bool use_lm_head_draft    = false;
    bool enable_thinking      = false;  // default thinking mode for the generation prompt
    // Used when a request omits max_tokens. 0 => derive from max_context in
    // parse_serve_options; --default-max-tokens overrides with an explicit value.
    int default_max_tokens    = 0;
    bool enable_cors          = false;  // send permissive CORS headers for browser UIs
};

// The default output length derived from the context window when the operator
// does not pass --default-max-tokens: min(max_context/2, kDefaultMaxTokensCeiling),
// floored at 1 so it is always a positive cap.
int derive_default_max_tokens(std::uint32_t max_context);

ServeOptions parse_serve_options(int argc, char** argv);
std::string serve_usage_text(const char* argv0);

} // namespace qus::serve
