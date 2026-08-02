// Provides QueryEngine::execute() for CPU-only (KERNELLAKE_WITH_CUDA=OFF)
// builds. Mutually exclusive with query_engine_execute_gpu.cpp -- exactly
// one of the two is compiled into kernellake_api, selected by
// src/api/CMakeLists.txt based on KERNELLAKE_WITH_CUDA, so there is never an
// ODR conflict between them.
//
// This file's execute(sql) is not simply "always throws" any more: when
// config_.engine.backend == "cpu", it runs the query for real on the
// always-available Acero CPU backend (query_engine_execute_cpu.cpp), which
// needs no CUDA/RMM at all. Only a request for the "gpu" backend still
// throws here.
#include <chrono>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/planner/physical_plan.hpp"

namespace kernellake {

namespace {
[[noreturn]] void throw_no_gpu_build() {
  throw ExecutionError(
      "query execution requires GPU operators (libcudf/RMM), which are not part of this build; "
      "use `kernellake explain --sql ...` to see the plan KernelLake would run, build with "
      "-DKERNELLAKE_WITH_CUDA=ON once libcudf/RMM are installed, or pass --backend cpu to run on "
      "the Acero CPU execution backend instead");
}
}  // namespace

QueryResult QueryEngine::execute(std::string_view sql) const {
  if (config_.engine.backend != "cpu") throw_no_gpu_build();

  const auto wall_start = std::chrono::steady_clock::now();

  double metadata_inspection_seconds = 0.0;
  const LogicalPlanPtr logical = plan_logical(sql, &metadata_inspection_seconds);
  const PhysicalPlanPtr physical = build_physical_plan(logical, store_);

  QueryResult result = execute_cpu(physical);
  result.metadata_inspection_seconds = metadata_inspection_seconds;
  result.elapsed_wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  return result;
}

QueryResult QueryEngine::execute(const PhysicalPlanPtr& /*physical*/,
                                 RmmEnvironment& /*rmm_environment*/) const {
  throw_no_gpu_build();
}

}  // namespace kernellake
