#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "kernellake/common/config.hpp"

namespace kernellake {

// Query-scoped GPU memory usage: current and peak bytes allocated through
// the tracked resource since the tracker was created (see
// RmmEnvironment::track_query()).
struct MemoryUsage {
  std::int64_t current_bytes = 0;
  std::int64_t peak_bytes = 0;
};

// Owns KernelLake's RMM device memory resource stack for the process, built
// from EngineConfig's memory/engine sections, and installs it as the
// current CUDA device's default resource for the lifetime of this object
// (restoring whatever was previously installed on destruction).
//
// Resource stack (outermost first, i.e. what allocations actually go
// through): limiting_resource_adaptor (engine.query_memory_limit_bytes) ->
// statistics_resource_adaptor (current/peak/allocation tracking) ->
// either cuda_async_memory_resource (memory.use_async_allocator: true) or
// pool_memory_resource over cuda_memory_resource, sized by
// memory.pool_initial_bytes/pool_max_bytes.
//
// Avoids global mutable execution state beyond what CUDA/RMM themselves
// require (a single current-device-resource slot per process) -- see
// docs/architecture.md's "Concurrency" notes for why a single shared
// statistics/limiting stack is an acceptable MVP simplification given
// KernelLake executes one query at a time.
class RmmEnvironment {
public:
  explicit RmmEnvironment(const EngineConfig& config);
  ~RmmEnvironment();

  RmmEnvironment(const RmmEnvironment&) = delete;
  RmmEnvironment& operator=(const RmmEnvironment&) = delete;

  // Pushes a fresh set of statistics counters, runs `query`, pops them, and
  // returns the usage attributable to just that call -- this is how
  // per-query current/peak GPU memory (QueryResult::peak_gpu_memory_bytes)
  // is measured without needing a separate resource per query.
  MemoryUsage track_query(const std::function<void()>& query);

  [[nodiscard]] MemoryUsage current_usage() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernellake
