#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "kernellake/common/config.hpp"
#include "kernellake/memory/query_memory_tracker.hpp"

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

// Owns KernelLake's RMM device memory resource stack for the process, built
// from EngineConfig's memory/engine sections, and installs it as the
// current CUDA device's default resource for the lifetime of this object
// (restoring whatever was previously installed on destruction).
//
// Resource stack (outermost first, i.e. what allocations actually go
// through): TrackingMemoryResource (process-wide OTel-facing counters via
// GpuMemoryMetricsRegistry -- see kernellake/memory/gpu_memory_metrics.hpp;
// deliberately outermost so it also sees limiter rejections, not just
// genuine CUDA OOM) -> limiting_resource_adaptor
// (engine.query_memory_limit_bytes, SHARED across every concurrently
// in-flight query -- see make_query_tracker() below for why the ceiling
// itself stays global while only reporting is per-query) -> either
// cuda_async_memory_resource (memory.use_async_allocator: true) or
// pool_memory_resource over cuda_memory_resource, sized by
// memory.pool_initial_bytes/pool_max_bytes. Every layer here is already
// documented thread-safe by RMM itself (limiting_resource_adaptor: atomics;
// pool_memory_resource/cuda_async_memory_resource: safe for concurrent
// allocate/deallocate by design) -- see make_query_tracker()'s own comment
// for the one layer that wasn't safe to share (per-query statistics) and
// how that's handled instead.
//
// Avoids global mutable execution state beyond what CUDA/RMM themselves
// require (a single current-device-resource slot per process) -- see
// docs/ARCHITECTURE.md's "Concurrency" notes.
class RmmEnvironment {
 public:
  explicit RmmEnvironment(const EngineConfig& config);
  ~RmmEnvironment();

  RmmEnvironment(const RmmEnvironment&) = delete;
  RmmEnvironment& operator=(const RmmEnvironment&) = delete;

  // Returns a fresh, independent QueryMemoryTracker for one query, wrapping
  // this instance's shared limiting_resource_adaptor as upstream -- safe to
  // call concurrently from multiple threads (each call constructs its own
  // statistics_resource_adaptor instance; nothing here is shared between
  // them beyond the already-thread-safe limiter/pool underneath). Replaces
  // the old track_query(std::function)/current_usage() pair, which pushed/
  // popped a *single shared* statistics_resource_adaptor stack -- safe for
  // one query at a time, but not for concurrent callers: RMM's own stack is
  // one structure, not thread-local, so two threads' push/pop calls could
  // race and pop each other's frame. A fresh instance per query sidesteps
  // that class of bug entirely rather than trying to synchronize around it.
  [[nodiscard]] QueryMemoryTracker make_query_tracker();

  // The byte count actually enforced by this instance's
  // limiting_resource_adaptor -- resolved once, at construction, via
  // resolve_query_memory_limit_bytes(). Callers needing to size anything
  // relative to the real ceiling (e.g. query_engine_execute_gpu.cpp's
  // pass_read_limit_bytes) must read it from here, not by calling
  // resolve_query_memory_limit_bytes() again themselves: for a
  // long-lived RmmEnvironment (kernellake-server keeps one for the whole
  // process, reused across every query -- see
  // GpuExecutionCoordinator), free VRAM at query time can differ from
  // free VRAM when this instance was constructed, and a fresh call would
  // silently drift from the ceiling this instance actually enforces.
  [[nodiscard]] std::uint64_t query_memory_limit_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernellake
