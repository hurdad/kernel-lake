#pragma once

#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// Applies KernelLake's rule-based logical optimizer:
//   - constant folding and boolean simplification
//   - BETWEEN -> (>= AND <=) simplification
//   - combining adjacent filters / removing filters that fold to TRUE
//     (a filter that folds to FALSE is kept but annotated estimated_rows=0,
//     since there is no dedicated empty-result plan node yet)
//   - redundant (identity) projection removal
//   - LIMIT pushdown through pass-through projections
//   - projection pushdown: annotates LogicalScan::required_columns() with
//     the minimal column set the plan actually references
//   - predicate pushdown: annotates LogicalScan::pushable_predicates() with
//     `column OP literal` conjuncts extracted from the WHERE clause, for
//     Parquet file/row-group pruning to consume later
//
// All rules operate on the structured plan/expression trees -- never on SQL
// text -- and return a new, potentially smaller/simpler plan; correctness is
// never traded for a more aggressive rewrite.
[[nodiscard]] LogicalPlanPtr optimize(LogicalPlanPtr plan);

}  // namespace kernellake
