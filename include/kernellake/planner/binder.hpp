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

struct BoundQuery {
  std::vector<BoundSelectItem> select_list;
  Schema output_schema{std::vector<Field>{}};
  std::vector<std::string> source_paths;
  ExpressionPtr where;  // null if no WHERE clause
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
// inspection (see docs/architecture.md); in the full engine it comes from
// inspecting the FROM read_parquet(...) source's Parquet schema.
//
// Throws BindingError, identifying the problem, on: unknown or ambiguous
// columns, invalid aggregate usage (nested aggregates, aggregates in
// WHERE/GROUP BY, ungrouped columns in an aggregate query), incompatible
// comparisons, unsupported '*' usage, and duplicate output names.
[[nodiscard]] BoundQuery bind_query(const sql::AstSelectStatement& stmt, const Schema& input_schema);

}  // namespace kernellake
