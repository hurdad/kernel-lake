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
// The single-table overload; only valid when `!query.join.has_value()`.
// `partition_columns` (empty by default, for a plain non-partitioned
// source and every existing caller that predates Hive-style partition
// discovery) is threaded straight onto the resulting LogicalScan -- see
// kernellake/io/table_resolution.hpp for how it's discovered.
[[nodiscard]] LogicalPlanPtr build_logical_plan(const BoundQuery& query, const Schema& source_schema,
                                                std::vector<PartitionColumn> partition_columns = {});

// The JOIN-chain overload; only valid when `query.join.has_value()`.
// `join_schemas` must have exactly `query.join->steps.size() + 1` entries,
// one per FROM-clause source in left-to-right order (matching the schemas
// passed to bind_query()'s own JOIN overload) -- builds a left-deep chain
// of LogicalJoin(..., LogicalScan(...)) nodes, one join per step.
// `partition_columns_per_source`, if non-empty, must have the same length
// as `join_schemas` and is threaded onto each corresponding LogicalScan.
[[nodiscard]] LogicalPlanPtr build_logical_plan(
    const BoundQuery& query, const std::vector<Schema>& join_schemas,
    std::vector<std::vector<PartitionColumn>> partition_columns_per_source = {});

}  // namespace kernellake
