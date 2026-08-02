#pragma once

#include "kernellake/planner/binder.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// Builds the initial (unoptimized) logical plan from a bound query plus the
// schema(s) of its FROM source(s). Column indices inside `query`'s
// expressions were assigned by the binder against these same schemas (in
// the JOIN case, against their concatenation -- see LogicalJoin), so this
// function must be called with the same schema(s) that were passed to the
// matching bind_query() overload.
//
// `right_schema` must be non-null exactly when `query.join.has_value()` --
// builds a LogicalJoin(LogicalScan(left), LogicalScan(right)) as the plan's
// source instead of a single LogicalScan.
[[nodiscard]] LogicalPlanPtr build_logical_plan(const BoundQuery& query, const Schema& source_schema,
                                                const Schema* right_schema = nullptr);

}  // namespace kernellake
