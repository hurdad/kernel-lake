#pragma once

#include <memory>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/config.hpp"
#include "kernellake/planner/physical_plan.hpp"

namespace kernellake {

// Owns one long-lived RmmEnvironment per GPU this server is configured to
// use (see docs/MULTI_GPU_SCALING.md's Tier 1) that a Flight SQL server
// needs to safely call QueryEngine::execute(physical, RmmEnvironment&)
// across concurrent gRPC handler threads for the "gpu" backend, instead of
// the one-shot QueryEngine::execute(sql) convenience overload's per-call
// RmmEnvironment (see query_engine.hpp's own doc comment on that overload,
// and docs/ARCHITECTURE.md's Concurrency notes -- rebuilding the RMM pool
// per request is both wrong under concurrency and wasteful even
// single-threaded). Which GPUs "this server is configured to use" means is
// ServerConfig::gpu_device_ids -- every visible device
// (cudaGetDeviceCount()) if that list is empty (Tier 1's original
// behavior), or exactly the listed ordinals otherwise, e.g. to leave some
// GPUs on a shared box for other workloads.
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
// backend == "cpu" in the first place. Also throws ConfigurationError if
// ServerConfig::gpu_device_ids names an ordinal >= the real
// cudaGetDeviceCount(), or if cudaGetDeviceCount() itself reports zero
// visible devices.
class GpuExecutionCoordinator {
 public:
  explicit GpuExecutionCoordinator(const ServerConfig& config);
  ~GpuExecutionCoordinator();

  GpuExecutionCoordinator(const GpuExecutionCoordinator&) = delete;
  GpuExecutionCoordinator& operator=(const GpuExecutionCoordinator&) = delete;

  // Round-robins each call across one RmmEnvironment per visible CUDA
  // device (docs/MULTI_GPU_SCALING.md's Tier 1) -- not the single
  // process-wide RmmEnvironment pinned to config.engine.device_id this used
  // to be. Each device's own share of concurrent queries is bounded to at
  // most EngineSection::max_concurrent_gpu_queries at a time (a semaphore,
  // not the single-flight mutex this used to be before opt #2 -- see that
  // config field's own comment for why bounded rather than unbounded, and
  // RmmEnvironment::make_query_tracker() for how per-query GPU memory
  // reporting stays correctly isolated once more than one call can be
  // in-flight against the same device at once). A single query still runs
  // entirely on whichever one device it's dispatched to -- this does not
  // split one query's work across multiple GPUs.
  [[nodiscard]] QueryResult execute(const QueryEngine& engine, const PhysicalPlanPtr& physical);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernellake
