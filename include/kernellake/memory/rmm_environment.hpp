#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "kernellake/common/config.hpp"

namespace kernellake {

// Resolves EngineSection::query_memory_limit_bytes's "0 means auto-detect"
// convention (see that field's own comment) into a concrete byte count:
// the configured value as-is if non-zero (an explicit override always
// wins), otherwise 75% of config.engine.device_id's currently *free* VRAM
// (not total capacity -- a real device can already have a meaningful chunk
// permanently held by something else entirely, e.g. a desktop compositor
// on a GPU shared with a display; sizing off total capacity there would
// set a limit the allocator could never actually satisfy), queried fresh
// via cudaMemGetInfo() every call rather than cached -- cheap (a single
// driver call), and correct if a caller ever changes device_id between
// calls. 75%, not e.g. 90%, leaves real headroom for the CUDA context
// itself, driver overhead, and any further external usage that arises
// after this snapshot was taken -- deliberately not tuned to exactly
// saturate whatever was free at that instant. Every caller that
// needs the actually-enforced limit (RmmEnvironment's own
// limiting_resource_adaptor, and query_engine_execute_gpu.cpp's
// pass_read_limit_bytes sizing) must call this rather than reading
// config.engine.query_memory_limit_bytes directly, or the two would
// disagree whenever auto-detection is in effect.
[[nodiscard]] std::uint64_t resolve_query_memory_limit_bytes(const EngineConfig& config);

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
// docs/ARCHITECTURE.md's "Concurrency" notes for why a single shared
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
