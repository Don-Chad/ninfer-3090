#pragma once
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/execution_context.h"

#include "core/nvtx.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        const cudaStream_t entry = state.execution.device.stream;
        // A split model's graph also runs the later stages' layers, but only rank 0's stream
        // orders it: fence it against the other ranks' streams on both sides.
        if (state.execution.stages != nullptr) {
            state.execution.stages->join_into(state.execution.device, entry);
        }
        executable->launch(entry);
        if (state.execution.stages != nullptr) {
            state.execution.stages->release_from(state.execution.device, entry);
        }
    } else {
        nvtx::ScopedRange eager_range(nvtx::Name::DecodeEager, nvtx::Category::Decode);
        body();
    }
}

template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    definition.capture(state.execution.device.stream, body);
}

} // namespace ninfer::models::qwen3_5::execution
