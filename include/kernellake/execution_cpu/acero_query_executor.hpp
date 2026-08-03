#pragma once

#include <arrow/api.h>

#include <memory>

#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

struct CpuQueryExecutionResult {
  std::shared_ptr<arrow::Table> table;
  double execution_seconds = 0.0;
};

// Translates `physical` into an arrow::acero::Declaration tree and runs it
// to completion via arrow::acero::DeclarationToTable() -- the CPU execution
// backend (see docs/ARCHITECTURE.md's CPU backend section). Deliberately
// free of any dependency on kernellake_api/QueryEngine (this module sits
// *below* kernellake_api in the dependency graph, mirroring how
// kernellake_execution/operator_builder.hpp relates to the GPU path) --
// callers are responsible for turning the returned arrow::Table into
// whatever result shape they need.
//
// Scope for this phase (matching the GPU engine's own original MVP build
// order): ParquetScan, Filter, Projection (arithmetic/comparisons/BETWEEN/
// numeric CAST only), ScalarAggregate, HashAggregate (grouping only by a
// plain column, not a computed alias), Sort (by a plain column only, not a
// computed expression), Limit. LIKE/IN/CASE/CAST-to-DECIMAL-or-STRING/
// HashJoin/GROUP-BY-or-ORDER-BY-alias are not yet supported here -- throws
// PlanningError/ExecutionError naming the unsupported node rather than
// silently miscompiling. See docs/ARCHITECTURE.md for the full scope and
// the reasoning behind it.
// `store` resolves each ParquetScanNode's fragments to bytes (local, S3,
// GCS, or Azure -- see the GPU path's ParquetScanOperator for the same
// split); must outlive this call.
[[nodiscard]] CpuQueryExecutionResult execute_physical_plan_cpu(const PhysicalPlanPtr& physical,
                                                                ObjectStore& store);

}  // namespace kernellake
