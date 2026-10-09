#pragma once
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/program/program.h"

namespace ninfer::models::qwen3_5 {
struct RuntimeTypes {
    using Frontend          = qwen3_5::Frontend;
    using PreparedPrompt    = qwen3_5::PreparedPrompt;
    using OutputSession     = qwen3_5::OutputSession;
    using PublishedOutput   = qwen3_5::PublishedOutput;
    using SequencePlanner   = qwen3_5::SequencePlanner;
    using SequencePlan      = qwen3_5::SequencePlan;
    using RequestBasePlan   = qwen3_5::RequestBasePlan;
    using SequenceHandle    = qwen3_5::SequenceHandle;
    using CheckpointHandle  = qwen3_5::CheckpointHandle;
    using CheckpointSummary = qwen3_5::CheckpointSummary;
    using SourceCandidate   = qwen3_5::SourceCandidate;
    using ResumeState       = qwen3_5::ResumeState;
    using ExecutionUnit     = qwen3_5::ExecutionUnit;
    using ExecutionUnitKind = qwen3_5::ExecutionUnitKind;
    using ContextProgress   = qwen3_5::ContextProgress;
    using PendingBatch      = qwen3_5::PendingBatch;
    using PrefillProgress   = qwen3_5::PrefillProgress;
    using ReplayProgress    = qwen3_5::ReplayProgress;
    using CommitResult      = qwen3_5::CommitResult;
    using DiscardResult     = qwen3_5::DiscardResult;
    using FinishResult      = qwen3_5::FinishResult;
    using AbortResult       = qwen3_5::AbortResult;
    using Program           = qwen3_5::Program;
    using CacheSessionKey   = qwen3_5::PreparedSessionKey;
};
} // namespace ninfer::models::qwen3_5
