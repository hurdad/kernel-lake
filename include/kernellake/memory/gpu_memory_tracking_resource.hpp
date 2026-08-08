#pragma once

// Split into its own header from gpu_memory_metrics.hpp deliberately:
// GpuMemoryMetricsRegistry itself is plain atomics, no CUDA/RMM type in
// sight (so it stays trivially unit-testable, and RMM could be swapped for
// something else later without touching it -- see its own header comment).
// This file is the one place that couples the registry to RMM/CCCL's
// resource-concept system.
#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <utility>

#include "kernellake/memory/gpu_memory_metrics.hpp"

namespace kernellake {

// Thin pass-through wrapper satisfying CCCL's cuda::mr::resource concept
// (the same structural-typing scheme rmm::mr::cuda_memory_resource/
// statistics_resource_adaptor/etc. satisfy in this RMM version -- see
// rmm_environment.cpp's own comment on the 26.06 switch away from a
// virtual device_memory_resource base class): every allocate/deallocate
// call is forwarded to `upstream` unchanged, with a
// GpuMemoryMetricsRegistry::record_*() call bracketing it. Intended as the
// *outermost* layer of RmmEnvironment's resource stack -- see that file --
// so it sees every allocation attempt actually issued through
// set_current_device_resource(), including ones the limiting_resource_
// adaptor beneath it rejects for exceeding query_memory_limit_bytes (a
// real, actionable "allocation failure" from an operator's point of view,
// not just genuine CUDA OOM).
class TrackingMemoryResource {
 public:
  TrackingMemoryResource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream, int device_id)
      : upstream_(std::move(upstream)), device_id_(device_id) {}

  // Alignment-taking overloads only -- cuda::mr::any_resource also exposes
  // deprecated no-alignment overloads (kept for source compatibility with
  // pre-alignment-argument callers, per their own "will be removed in a
  // future release" deprecation note), but nothing in this codebase calls
  // them and the resource concept doesn't require a wrapper to provide
  // them itself -- confirmed by the static_assert below still passing
  // without them.
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    return allocate_impl(bytes, [&] { return upstream_.allocate(stream, bytes, alignment); });
  }

  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    GpuMemoryMetricsRegistry::record_deallocation(device_id_, bytes);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return allocate_impl(bytes, [&] { return upstream_.allocate_sync(bytes, alignment); });
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept {
    upstream_.deallocate_sync(ptr, bytes, alignment);
    GpuMemoryMetricsRegistry::record_deallocation(device_id_, bytes);
  }

  [[nodiscard]] bool operator==(const TrackingMemoryResource& other) const noexcept {
    return upstream_ == other.upstream_;
  }
  [[nodiscard]] bool operator!=(const TrackingMemoryResource& other) const noexcept {
    return !(*this == other);
  }

  friend constexpr void get_property(const TrackingMemoryResource&, cuda::mr::device_accessible) noexcept {}

 private:
  // Records a success/failure against GpuMemoryMetricsRegistry around
  // whatever `upstream_` call `fn` makes, without altering the exception
  // `fn` throws on failure -- see this class's own doc comment on why
  // failures (including limiter rejections) are worth tracking, and the
  // registry header's "preserve existing error semantics" note.
  template <class Fn>
  void* allocate_impl(std::size_t bytes, Fn&& fn) {
    try {
      void* ptr = fn();
      GpuMemoryMetricsRegistry::record_allocation(device_id_, bytes);
      return ptr;
    } catch (...) {
      GpuMemoryMetricsRegistry::record_allocation_failure(device_id_);
      throw;
    }
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  int device_id_;
};

static_assert(cuda::mr::resource_with<TrackingMemoryResource, cuda::mr::device_accessible>,
              "TrackingMemoryResource does not satisfy the cuda::mr::resource concept");

}  // namespace kernellake
