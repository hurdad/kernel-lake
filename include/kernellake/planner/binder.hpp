#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kernellake/expression/expression.hpp"
#include "kernellake/sql/ast.hpp"
#include "kernellake/types/join_type.hpp"
#include "kernellake/types/schema.hpp"

namespace kernellake {

struct BoundSelectItem {
  ExpressionPtr expr;
  std::string output_name;
};

struct BoundOrderByItem {
  ExpressionPtr expr;
  bool ascending;
};

// One additional source joined onto the running left-deep chain built so
// far -- see sql::AstJoinStep and BoundJoin below. `combined_key_index` is
// an index into the *accumulated* schema of every source before this step
// (source 0's fields, then source 1's, ...), matching exactly what a
// left-deep chain of LogicalJoin/HashJoinNode nodes produces at each
// level; `source_key_index` is an index into this step's own new source's
// *own* (unpruned) schema, not the combined row -- the physical planner
// needs it to build the HashJoinOperator against the actual pruned
// Parquet scan schema, the same way any other column reference gets
// remapped above a scan.
struct BoundJoinStep {
  std::vector<std::string> source_paths;
  std::size_t combined_key_index;
  std::size_t source_key_index;
  JoinType join_type = JoinType::Inner;
  // Non-null when this step's ON clause had additional AND-conjuncts beyond
  // the required equality key, all referencing only this step's own new
  // source (e.g. TPC-H Q13's `ON c_custkey = o_custkey AND o_comment NOT
  // LIKE '%special%requests%'`) -- already rebased to that source's own
  // 0-based schema (indices into `source_paths`' own schema, not the
  // combined row). logical_planner.cpp applies it as a LogicalFilter
  // directly on that source's scan, before it becomes this step's join
  // child -- see extract_join_step_keys() (binder.cpp) for why this is
  // exact (not an approximation) for both INNER and LEFT OUTER JOIN, and
  // why a left-side-referencing conjunct is rejected at bind time instead
  // of ever reaching here.
  ExpressionPtr right_prefilter;
  // Non-null only for `join_type == LeftSemi/LeftAnti` (TPC-H Q21's
  // shape: `EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey =
  // l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)`) -- an ON-clause
  // conjunct beyond the required equality key that references *both*
  // sides at once (not just the newly-joined source's own columns, the
  // only case `right_prefilter` above handles), so it can't be pushed
  // down as a single-sided prefilter at all; it must be evaluated per
  // candidate (already-joined-row, newly-joined-row) pair, at the join
  // itself. Column indices are unrebased -- exactly the same combined-
  // schema convention `combined_key_index` above already uses (indices
  // `< combined_field_count` are the already-joined side's own original
  // position, `>= combined_field_count` are `combined_field_count` +
  // the newly-joined source's own original position) -- see
  // extract_join_step_keys() (binder.cpp) for why real `JOIN ... ON`
  // (INNER/LEFT OUTER) still rejects this shape outright rather than
  // accepting it here too.
  ExpressionPtr residual_predicate;
};

// The bound form of a `JOIN ... ON` chain (see sql::AstJoinClause):
// `first_source_paths` is the leftmost source, `steps` joins one more
// source at a time onto the running combined result, in left-to-right
// order -- `steps.size() + 1` total sources, matching a left-deep chain of
// binary joins (`(first JOIN steps[0]) JOIN steps[1]) ...`), which is
// exactly what build_logical_plan() constructs from this.
struct BoundJoin {
  std::vector<std::string> first_source_paths;
  std::vector<BoundJoinStep> steps;  // at least one
};

struct BoundQuery {
  std::vector<BoundSelectItem> select_list;
  Schema output_schema{std::vector<Field>{}};
  std::vector<std::string> source_paths;  // single-table case; empty when `join` is set
  std::optional<BoundJoin> join;          // JOIN chain case (see bind_query's join overload)
  ExpressionPtr where;                    // null if no WHERE clause
  std::vector<ExpressionPtr> group_by;
  // `HAVING <bool expr>` -- null if no HAVING clause. Only ever set when
  // `is_aggregate_query` is true (see bind_query_common()); may reference
  // aggregates (unlike `where` above, which never may) via the same
  // ColumnExpression-into-aggregate-output mechanism SELECT-list items
  // use -- see rewrite_aggregate_refs() (logical_planner.cpp).
  ExpressionPtr having;
  std::vector<BoundOrderByItem> order_by;
  std::optional<std::int64_t> limit;
  bool is_aggregate_query = false;
};

// Binds a parsed AST against the schema of its FROM source: resolves column
// names, assigns expression result types, inserts safe implicit numeric
// casts, and validates aggregate/GROUP BY usage.
//
// `input_schema` is supplied by the caller rather than looked up here, so
// the binder stays decoupled from file discovery and Parquet metadata
// inspection (see docs/ARCHITECTURE.md); in the full engine it comes from
// inspecting the FROM read_parquet(...) source's Parquet schema.
//
// Throws BindingError, identifying the problem, on: unknown or ambiguous
// columns, invalid aggregate usage (nested aggregates, aggregates in
// WHERE/GROUP BY, ungrouped columns in an aggregate query), incompatible
// comparisons, unsupported '*' usage, and duplicate output names.
[[nodiscard]] BoundQuery bind_query(const sql::AstSelectStatement& stmt, const Schema& input_schema);

// The `FROM ... JOIN ... ON ... [JOIN ... ON ...]` overload -- only called
// when `stmt.join.has_value()`. `join_schemas` must have exactly
// `stmt.join->steps.size() + 1` entries, one per source in left-to-right
// FROM-clause order (matching `stmt.join->first`/`stmt.join->steps`).
// Every resolved column's index (in the SELECT list, WHERE, GROUP BY, ...)
// is into the *combined* row (source 0's fields, then source 1's, ...),
// matching what a left-deep chain of HashJoinOperators actually produces;
// each JOIN step's own condition is required to be a single equality
// between one plain column already in the running combined schema and one
// plain column from the newly-joined source (of identical type -- mixing
// e.g. INT32 and INT64 join keys is not supported, see
// docs/ARCHITECTURE.md), which `BoundQuery::join` records by each side's
// *own* (pre-join) column index for the physical planner.
[[nodiscard]] BoundQuery bind_query(const sql::AstSelectStatement& stmt,
                                    const std::vector<Schema>& join_schemas);

}  // namespace kernellake
