#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "ninfer/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <functional>

namespace ninfer::serve {

// Media acquisition is a product-layer concern. Translation preserves part order
// and asks the caller to turn each wire source into owning bytes before the
// target frontend sees it.
using MediaAcquirer = std::function<ninfer::OwnedMedia(const ContentPart&)>;

struct ResolvedPromptSemantics {
    std::optional<bool> enable_thinking;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    // The client's own effort choice (top-level field or chat_template_kwargs), or unset when the
    // server default filled it in. Distinct from reasoning_effort, which also carries the resolved
    // server-default value so the template still has one place to read the effective effort from.
    std::optional<RequestedReasoningEffort> requested_reasoning_effort;
    std::optional<bool> preserve_thinking;
    // The client's own preserve_thinking choice (top-level field or chat_template_kwargs), or
    // unset when the server --preserve-thinking default filled it in. Distinct from
    // preserve_thinking, which also carries the resolved server-default value for execution.
    std::optional<bool> requested_preserve_thinking;
    std::string chat_template_kwargs_json;
    std::string graft;
};

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& req,
                                                 const ServeOptions& server);

ninfer::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media);

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy.
ninfer::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse);

} // namespace ninfer::serve
