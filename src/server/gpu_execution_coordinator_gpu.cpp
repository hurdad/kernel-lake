// Provides GpuExecutionCoordinator for KERNELLAKE_WITH_CUDA=ON builds.
// Mutually exclusive with gpu_execution_coordinator_stub.cpp -- see that
// file's comment.
#include "kernellake/server/gpu_execution_coordinator.hpp"

#include <mutex>

#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {

struct GpuExecutionCoordinator::Impl {
  explicit Impl(const EngineConfig& config) : rmm_environment(config) {}

  RmmEnvironment rmm_environment;
  std::mutex execute_mutex;
};

GpuExecutionCoordinator::GpuExecutionCoordinator(const EngineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

GpuExecutionCoordinator::~GpuExecutionCoordinator() = default;

QueryResult GpuExecutionCoordinator::execute(const QueryEngine& engine, const PhysicalPlanPtr& physical) {
  const std::lock_guard<std::mutex> lock(impl_->execute_mutex);
  return engine.execute(physical, impl_->rmm_environment);
}

}  // namespace kernellake
