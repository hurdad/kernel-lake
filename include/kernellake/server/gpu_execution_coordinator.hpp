#pragma once

#include <memory>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/config.hpp"
#include "kernellake/planner/physical_plan.hpp"

namespace kernellake {

// Owns the single long-lived RmmEnvironment a Flight SQL server needs to
// safely call QueryEngine::execute(physical, RmmEnvironment&) across
// concurrent gRPC handler threads for the "gpu" backend, instead of the
// one-shot QueryEngine::execute(sql) convenience overload's per-call
// RmmEnvironment (see query_engine.hpp's own doc comment on that overload,
// and docs/ARCHITECTURE.md's Concurrency notes -- rebuilding the RMM pool
// per request is both wrong under concurrency and wasteful even
// single-threaded).
//
// This split into its own translation-unit pair
// (gpu_execution_coordinator_gpu.cpp / _stub.cpp, selected by
// KERNELLAKE_WITH_CUDA exactly like query_engine_execute_gpu.cpp /
// _stub.cpp) keeps every other server source CUDA-agnostic: the "gpu"
// backend case is the only place a Flight SQL server needs to touch
// RmmEnvironment at all, since the "cpu" backend calls
// QueryEngine::execute_cpu() directly and needs no external resource
// (Acero owns its own thread pool).
//
// Constructing this in a KERNELLAKE_WITH_CUDA=OFF build throws
// ConfigurationError immediately -- callers must not construct it for
// backend == "cpu" in the first place.
class GpuExecutionCoordinator {
 public:
  explicit GpuExecutionCoordinator(const EngineConfig& config);
  ~GpuExecutionCoordinator();

  GpuExecutionCoordinator(const GpuExecutionCoordinator&) = delete;
  GpuExecutionCoordinator& operator=(const GpuExecutionCoordinator&) = delete;

  // Runs concurrent calls against the one shared RmmEnvironment, bounded
  // to at most EngineSection::max_concurrent_gpu_queries at a time (a
  // semaphore, not the single-flight mutex this used to be -- see that
  // config field's own comment for why bounded rather than unbounded, and
  // RmmEnvironment::make_query_tracker() for how per-query GPU memory
  // reporting stays correctly isolated once more than one call can be
  // in-flight here at once).
  [[nodiscard]] QueryResult execute(const QueryEngine& engine, const PhysicalPlanPtr& physical);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernellake
