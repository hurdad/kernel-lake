#pragma once

#include "kernellake/planner/logical_plan.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Converts an optimized LogicalPlan into a PhysicalPlan: resolves the
// LogicalScan's source paths into concrete files via `store`, inspects each
// file's Parquet metadata, validates cross-file schema compatibility,
// evaluates row-group pruning against LogicalScan::pushable_predicates(),
// and maps the remaining logical nodes onto their physical equivalents
// (LogicalAggregate becomes HashAggregate when it has a GROUP BY, or
// ScalarAggregate when it does not). Always wraps the result in an
// ArrowResultNode.
//
// Throws PlanningError for constructs with no physical implementation yet
// (currently: a LogicalSort anywhere in the plan, since no physical Sort
// operator exists -- see docs/architecture.md's future-operator list)
// rather than silently dropping it and returning unsorted results.
[[nodiscard]] PhysicalPlanPtr build_physical_plan(const LogicalPlanPtr& logical_plan,
                                                   ObjectStore& store);

}  // namespace kernellake
