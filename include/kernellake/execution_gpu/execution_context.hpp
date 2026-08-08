#pragma once

#include <cuda_runtime.h>
#include <rmm/resource_ref.hpp>

#include "kernellake/common/identifiers.hpp"
#include "kernellake/execution_gpu/metrics_registry.hpp"
#include "kernellake/observability/query_tracing.hpp"

namespace kernellake {

class CancellationToken;
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

  // The span the *currently opening* operator should parent its own child
  // span to (see InstrumentedOperator, operator_builder.cpp), giving Jaeger
  // a real span tree shaped like the physical plan rather than one flat
  // query span. nullptr at the top of the tree (the outermost operator
  // parents to the whole-query span instead -- see start_query_span()).
  // Non-owning; only ever points at a ClientSpan that outlives the
  // recursive open() call it's set for -- see InstrumentedOperator::open()
  // for why this is threaded explicitly through here rather than via
  // OTel's own thread-local "current span" mechanism (query_tracing.hpp's
  // ClientSpan/start_client_span already use that mechanism for their one
  // existing caller, an outbound gRPC call -- it doesn't generalize to an
  // operator with more than one child opened sequentially, e.g.
  // HashJoinOperator's left then right).
  const observability::ClientSpan* current_span = nullptr;
};

}  // namespace kernellake
