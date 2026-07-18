#include <ninfer/targets/qwen3_6_35b_a3b_rtx5090/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6_35b_a3b_rtx5090/impl/load/bindings.h"
#include "targets/qwen3_6_35b_a3b_rtx5090/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_35b_a3b_rtx5090::detail {

class LoadPlan::Impl {
public:
    explicit Impl(ArtifactLoadPlan target_plan) : plan(std::move(target_plan)) {}

    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_35b_a3b_rtx5090::detail

namespace ninfer::targets::qwen3_6_35b_a3b_rtx5090 {

void Package::preflight(DeviceContext& device, const EngineOptions& options) {
    detail::Variant::preflight(device, options);
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(detail::bind_artifact(binder)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(std::move(plan.impl_->plan.bindings),
                                                    std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend);
}

Package::SequencePlan Package::plan_sequence(const LoadedModel& model, DeviceContext& device,
                                             const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::plan_sequence<detail::Variant>(model.impl_->data.runtime, device, options);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(model.impl_->data.runtime, std::move(plan),
                                                    device);
}

} // namespace ninfer::targets::qwen3_6_35b_a3b_rtx5090
