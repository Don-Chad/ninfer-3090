#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/frontend.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_6_27b_rtx5090 {

struct Package;

namespace detail {

class ActivationDumpAccess;

using Frontend       = qwen3_6::Frontend;
using PreparedPrompt = qwen3_6::PreparedPrompt;
using OutputSession  = qwen3_6::OutputSession;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b_rtx5090::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b_rtx5090::Package;
    friend class Program;
};

class SequencePlan {
public:
    struct Impl;

    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

private:
    explicit SequencePlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b_rtx5090::Package;
    friend class Program;
};

class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

private:
    struct Impl;
    explicit RequestPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Program;
};

class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    [[nodiscard]] RequestPlan plan_request(const PreparedPrompt& prompt,
                                           const ExecutionOptions& options) const;
    [[nodiscard]] runtime::BeginResult begin(PreparedPrompt&& prompt, RequestPlan&& plan,
                                             runtime::TransientRegion transient);
    [[nodiscard]] runtime::GeneratedRound decode_round(runtime::RoundBudget budget);

    void resolve_pending(std::uint32_t accepted_tokens, bool terminal);
    void finish_active();
    void abort_request() noexcept;

    [[nodiscard]] std::uint32_t materialized_tokens() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats() const;
    [[nodiscard]] GenerationTimings generation_timings() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    class Impl;
    explicit Program(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b_rtx5090::Package;
    friend class ActivationDumpAccess;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id = "qwen3.6-27b";

    using LoadPlan       = detail::LoadPlan;
    using LoadedModel    = detail::LoadedModel;
    using Frontend       = detail::Frontend;
    using PreparedPrompt = detail::PreparedPrompt;
    using OutputSession  = detail::OutputSession;
    using SequencePlan   = detail::SequencePlan;
    using RequestPlan    = detail::RequestPlan;
    using Program        = detail::Program;

    // Cheap target-owned option/device validation. The registry calls this after matching
    // model_id and before any weight materialization.
    static void preflight(DeviceContext& device, const EngineOptions& options);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model);
    [[nodiscard]] static SequencePlan plan_sequence(const LoadedModel& model, DeviceContext& device,
                                                    const EngineOptions& options);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::qwen3_6_27b_rtx5090
} // namespace ninfer
