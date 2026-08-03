#pragma once

#include <cstddef>
#include <memory>

#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Converts an optimized PhysicalPlan into a tree of concrete
// PhysicalOperator instances, ready for open()/next()/close().
//
// `store` resolves each ParquetScanNode's fragments to bytes (local, S3,
// GCS, or Azure -- see ParquetScanOperator's own docs); it must outlive the
// returned operator tree.
//
// `pass_read_limit_bytes` (0 = unlimited) is forwarded to
// ParquetScanOperator to bound its per-pass decompression memory; see that
// operator's own docs for why this is a byte budget rather than an exact
// row count.
//
// Every node in the returned tree is wrapped (see operator_builder.cpp's
// InstrumentedOperator) to record its own wall-clock next() time into
// ExecutionContext::metrics when non-null, and to emit an NVTX range per
// next() call when `nvtx_enabled` is true (see EngineConfig's
// ProfilingSection::nvtx) -- this is generic instrumentation, not something
// any individual operator implements itself.
[[nodiscard]] std::unique_ptr<PhysicalOperator> build_operator_tree(const PhysicalPlanPtr& plan,
                                                                    ObjectStore& store,
                                                                    std::size_t pass_read_limit_bytes,
                                                                    bool nvtx_enabled = false);

}  // namespace kernellake
