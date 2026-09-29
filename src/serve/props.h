#pragma once

// Read-only llama.cpp-style `GET /props` document for clients that discover a server through it.

#include "ninfer/types.h"
#include "serve/openai_common.h"
#include "serve/serve_options.h"

#include <string>

namespace ninfer::serve {

// Fills only the llama.cpp /props fields NInfer can state truthfully: the context ceiling, the
// generation defaults a request inherits when it sends none, the lane count, the model identity
// and the input modalities. `sampling_defaults` are the loaded model's registered presets; the
// reported sampler is the one the server's default thinking mode resolves to.
std::string make_props(const ServeOptions& options, const ModelDescription& model,
                       const ninfer::ModelSamplingDefaults& sampling_defaults);

} // namespace ninfer::serve
