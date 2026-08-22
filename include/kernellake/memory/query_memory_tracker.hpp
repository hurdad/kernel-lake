#pragma once

#include <cuda/memory_resource>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

namespace kernellake {

// Query-scoped GPU memory usage: current and peak bytes allocated through
// this tracker's own resource_ref() since construction. Distinct from
// RmmEnvironment's process-wide GpuMemoryMetricsRegistry counters (see
// gpu_memory_metrics.hpp) -- this is per-query, that's process-wide.
struct MemoryUsage {
  std::int64_t current_bytes = 0;
  std::int64_t peak_bytes = 0;
};

// One query's own memory-usage tracking, isolated from every other
// concurrently-running query.
//
// Owns a fresh rmm::mr::statistics_resource_adaptor wrapping whatever
// upstream resource is passed in (RmmEnvironment::make_query_tracker()
// passes its shared, process-wide limiting_resource_adaptor -- see that
// file for why the memory *ceiling* stays global/shared while only
// *reporting* is per-instance here). Deliberately NOT built on
// statistics_resource_adaptor's own push_counters()/pop_counters()
// stack -- that stack is a single, shared, non-thread-local structure
// (confirmed from RMM's own header: one counter stack, lock-protected but
// not per-caller), so two QueryMemoryTracker instances concurrently
// pushing/popping the *same* adaptor would race and could pop the wrong
// query's frame. Constructing a genuinely separate adaptor instance per
// query sidesteps that class of bug entirely: each instance's counters
// are its own, not a frame on a structure anything else touches.
class QueryMemoryTracker {
 public:
  explicit QueryMemoryTracker(cuda::mr::any_resource<cuda::mr::device_accessible> upstream)
      : stats_(std::move(upstream)) {}

  QueryMemoryTracker(const QueryMemoryTracker&) = delete;
  QueryMemoryTracker& operator=(const QueryMemoryTracker&) = delete;
  // Move-only: statistics_resource_adaptor is copyable (it shares ownership
  // of its internal state via cuda::mr::shared_resource), but a copy here
  // would defeat the whole point -- two QueryMemoryTracker instances
  // sharing one adaptor's counters is exactly the aliasing this type
  // exists to avoid. Move is fine (single owner, just relocated).
  QueryMemoryTracker(QueryMemoryTracker&&) = default;
  QueryMemoryTracker& operator=(QueryMemoryTracker&&) = default;

  // Usable as ExecutionContext::memory_resource for the query this tracker
  // belongs to -- every allocation issued through it counts against this
  // instance's own counters and, one layer further upstream, the shared
  // process-wide limiter.
  [[nodiscard]] rmm::device_async_resource_ref resource_ref() {
    return rmm::device_async_resource_ref{stats_};
  }

  // Usage so far -- safe to call mid-query, not just after close().
  [[nodiscard]] MemoryUsage current_usage() const {
    const rmm::mr::statistics_resource_adaptor::counter bytes = stats_.get_bytes_counter();
    return MemoryUsage{bytes.value, bytes.peak};
  }

 private:
  rmm::mr::statistics_resource_adaptor stats_;
};

}  // namespace kernellake
