#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kernellake/expression/expression.hpp"
#include "kernellake/sql/ast.hpp"
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

// The bound form of a two-table `JOIN ... ON` clause (see
// sql::AstJoinClause). `left_key_index`/`right_key_index` are indices into
// each side's *own* (unpruned) schema, not the combined post-join row --
// the physical planner needs each side's key column to build the
// HashJoinOperator against the actual pruned Parquet scan schema, the same
// way any other column reference gets remapped above a scan.
struct BoundJoin {
  std::vector<std::string> left_source_paths;
  std::size_t left_key_index;
  std::vector<std::string> right_source_paths;
  std::size_t right_key_index;
};

struct BoundQuery {
  std::vector<BoundSelectItem> select_list;
  Schema output_schema{std::vector<Field>{}};
  std::vector<std::string> source_paths;  // single-table case; empty when `join` is set
  std::optional<BoundJoin> join;          // two-table case (see bind_query's two-schema overload)
  ExpressionPtr where;                    // null if no WHERE clause
  std::vector<ExpressionPtr> group_by;
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

// The two-table `FROM ... JOIN ... ON ...` overload -- only called when
// `stmt.join.has_value()`. Every resolved column's index (in the SELECT
// list, WHERE, GROUP BY, ...) is into the *combined* [left_schema fields...,
// right_schema fields...] row, matching what HashJoinOperator actually
// produces; only the JOIN condition itself is required to be a single
// equality between one plain column from each side (of identical type --
// mixing e.g. INT32 and INT64 join keys is not supported, see
// docs/ARCHITECTURE.md), which `BoundQuery::join` records by each side's
// *own* (pre-join) column index for the physical planner.
[[nodiscard]] BoundQuery bind_query(const sql::AstSelectStatement& stmt, const Schema& left_schema,
                                    const Schema& right_schema);

}  // namespace kernellake
