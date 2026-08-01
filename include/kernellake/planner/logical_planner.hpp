#pragma once

#include "kernellake/planner/binder.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// Builds the initial (unoptimized) logical plan from a bound query plus the
// schema of its FROM source. Column indices inside `query`'s expressions
// were assigned by the binder against `source_schema`, so this function must
// be called with the same schema that was passed to bind_query().
//
// Known MVP limitation: ORDER BY after GROUP BY is rejected with
// PlanningError rather than silently sorting the wrong rows, since it would
// require binding ORDER BY against the post-aggregation output schema
// (aliases included), which is not yet implemented. ORDER BY on a
// non-aggregate query works normally.
[[nodiscard]] LogicalPlanPtr build_logical_plan(const BoundQuery& query,
                                                 const Schema& source_schema);

}  // namespace kernellake
