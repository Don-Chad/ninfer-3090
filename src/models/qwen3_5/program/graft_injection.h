#pragma once

#include "models/qwen3_5/frontend/graft.h"

#include <cuda_runtime.h>

namespace ninfer::models::qwen3_5 {

namespace detail {
class ProgramImpl;
}

// Inject a direct_kv or softprompt_kv graft into a pinned shared-prefix entry.
// Called once at startup, before any request is admitted. The graft's K/V are written into the
// text KV cache via kv_cache_append and its GDN conv/recurrent state is uploaded to a frozen
// StateImage. The synthesized shared prefix is Catalogued so admission can reference it.
// Throws on failure (no free shared-prefix slot, allocation errors, CUDA errors).
void inject_direct_graft(detail::ProgramImpl& program, const PromptGraft& graft,
                         cudaStream_t stream);

} // namespace ninfer::models::qwen3_5
