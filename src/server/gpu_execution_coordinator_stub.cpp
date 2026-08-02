// Provides GpuExecutionCoordinator for KERNELLAKE_WITH_CUDA=OFF builds.
// Mutually exclusive with gpu_execution_coordinator_gpu.cpp -- see that
// file's comment. Exists so kernellake-server can be built and tested
// entirely without CUDA/RMM (the server-dev preset); requesting the "gpu"
// backend against such a build fails fast at server startup here, rather
// than at first-query time.
#include "kernellake/server/gpu_execution_coordinator.hpp"

#include "kernellake/common/errors.hpp"

namespace kernellake {

struct GpuExecutionCoordinator::Impl {};

GpuExecutionCoordinator::GpuExecutionCoordinator(const EngineConfig& /*config*/) {
  throw ConfigurationError(
      "server.engine.backend 'gpu' requires GPU operators (libcudf/RMM), which are not part of "
      "this build; rebuild with -DKERNELLAKE_WITH_CUDA=ON once libcudf/RMM are installed, or set "
      "engine.backend: cpu to run kernellake-server on the Acero CPU execution backend instead");
}

GpuExecutionCoordinator::~GpuExecutionCoordinator() = default;

QueryResult GpuExecutionCoordinator::execute(const QueryEngine& /*engine*/,
                                             const PhysicalPlanPtr& /*physical*/) {
  throw ConfigurationError("unreachable: GpuExecutionCoordinator cannot be constructed in this build");
}

}  // namespace kernellake
