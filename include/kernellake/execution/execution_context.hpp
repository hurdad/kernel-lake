#pragma once

#include <cuda_runtime.h>
#include <rmm/resource_ref.hpp>

#include "kernellake/common/identifiers.hpp"

namespace kernellake {

class CancellationToken;
class MetricsRegistry;
class QueryMemoryTracker;

// Explicit execution context, passed by reference through every operator
// rather than relying on any process-wide "current query" state (see
// docs/ARCHITECTURE.md's Concurrency notes).
//
// `memory_resource` is an `rmm::device_async_resource_ref` (a small,
// type-erased reference value) rather than the raw
// `rmm::mr::device_memory_resource*` named in the original design sketch:
// this RMM version (26.06) replaced that abstract base class with CCCL's
// `cuda::mr` resource-concept system, and resource_ref is its idiomatic
// non-owning reference type -- see RmmEnvironment (kernellake/memory) for
// how the underlying resource stack is actually constructed.
struct ExecutionContext {
  QueryId query_id;
  int cuda_device_id = 0;
  cudaStream_t stream = nullptr;
  rmm::device_async_resource_ref memory_resource;
  CancellationToken* cancellation = nullptr;
  MetricsRegistry* metrics = nullptr;
  QueryMemoryTracker* memory_tracker = nullptr;
};

}  // namespace kernellake
