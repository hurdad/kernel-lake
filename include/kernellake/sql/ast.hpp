#pragma once

// Parser-independent KernelLake AST. Produced by kernellake::sql::parse_sql()
// (see parser.hpp), which internally uses a third-party SQL grammar (hyrise/
// sql-parser) but never leaks its types outside src/sql/parser.cpp. Nothing
// downstream of this header should ever depend on the third-party parser.
//
// This AST is untyped and unbound: column names are plain strings, not yet
// resolved to a schema position or a DataType. The binder (see
// kernellake/planner or kernellake/sql binder, task 7) walks this tree
// against a resolved Schema to produce a typed kernellake::Expression tree.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace kernellake::sql {

struct AstExpr;
using AstExprPtr = std::shared_ptr<AstExpr>;

// `table` is the optional qualifier on a qualified column reference (e.g.
// `a` in `a.key`) -- only meaningful for a two-table JOIN query, where it
// disambiguates which side's schema to resolve `name` against. Null for an
// unqualified reference.
struct AstColumnRef {
  std::string name;
  std::optional<std::string> table;
};

// SELECT * — expanded against the FROM schema during binding.
struct AstStar {};

enum class AstLiteralKind : std::uint8_t {
  Integer,
  Float,
  String,
  Boolean,
  Date,
  Null,
};

struct AstLiteral {
  AstLiteralKind kind;
  std::int64_t int_value = 0;  // Integer, and Date (days since 1970-01-01)
  double float_value = 0.0;    // Float
  std::string string_value;    // String
  bool bool_value = false;     // Boolean
};

enum class AstBinaryOp : std::uint8_t {
  Add,
  Subtract,
  Multiply,
  Divide,
  Eq,
  NotEq,
  Lt,
  LtEq,
  Gt,
  GtEq,
  And,
  Or,
};

struct AstBinary {
  AstBinaryOp op;
  AstExprPtr left;
  AstExprPtr right;
};

enum class AstUnaryOp : std::uint8_t {
  Not,
  Negate,
  IsNull,
  IsNotNull,
};

struct AstUnary {
  AstUnaryOp op;
  AstExprPtr operand;
};

struct AstBetween {
  AstExprPtr value;
  AstExprPtr lower;
  AstExprPtr upper;
};

enum class AstAggregateFunc : std::uint8_t {
  Sum,
  Count,
  CountStar,
  Min,
  Max,
  Avg,
};

struct AstAggregate {
  AstAggregateFunc function;
  AstExprPtr argument;  // null only for CountStar
};

// `value LIKE 'pattern'` ('%'/'_' wildcards) or `value NOT LIKE 'pattern'`.
// `pattern` must bind to a string literal (see binder.cpp) -- a per-row
// pattern column is not supported.
struct AstLike {
  AstExprPtr value;
  AstExprPtr pattern;
  bool negated = false;
};

// `value IN (list...)` or `value NOT IN (list...)`. Desugars at bind time
// into a chain of `=`/`<>` comparisons combined with OR/AND -- see
// binder.cpp -- rather than needing its own Expression/GPU-operator
// support.
struct AstIn {
  AstExprPtr value;
  std::vector<AstExprPtr> list;
  bool negated = false;
};

// `CASE WHEN c1 THEN r1 [WHEN c2 THEN r2 ...] [ELSE re] END`.
// `else_branch` is null for a CASE with no ELSE (result is NULL when no
// WHEN condition matches).
struct AstCase {
  std::vector<std::pair<AstExprPtr, AstExprPtr>> when_then;
  AstExprPtr else_branch;
};

// `CAST(operand AS type_name)`. `type_name` is resolved against
// KernelLake's DataType names by the binder, not here.
struct AstCast {
  AstExprPtr operand;
  std::string type_name;
  // Only meaningful for type_name == "DECIMAL" (hsql's own ColumnType
  // fields; see parser.cpp's kExprCast handling and binder.cpp's
  // resolve_cast_type_name()). Both are 0 for a bare `CAST(x AS DECIMAL)`
  // with no explicit precision/scale, which the binder rejects.
  std::int64_t decimal_precision = 0;
  std::int64_t decimal_scale = 0;
};

struct AstExpr {
  std::variant<AstColumnRef, AstStar, AstLiteral, AstBinary, AstUnary, AstBetween, AstAggregate, AstLike,
               AstIn, AstCase, AstCast>
      node;
  std::optional<std::string> alias;
};

struct AstOrderByItem {
  AstExprPtr expr;
  bool ascending = true;
};

// A single `read_parquet('path' [, 'path2', ...])` source, optionally
// aliased (`AS a`). An alias is required when this source participates in
// a JOIN (it's how a qualified column reference like `a.key` picks a side)
// but optional for the single-table MVP shape.
struct AstParquetSource {
  std::vector<std::string> paths;
  std::optional<std::string> alias;
};

// `FROM read_parquet(...) AS a JOIN read_parquet(...) AS b ON <condition>`.
// Only a two-table INNER JOIN is supported -- see docs/ARCHITECTURE.md for
// the full scope (single equality key, both sides must be aliased
// `read_parquet(...)` sources, no comma-style joins or 3+ tables).
struct AstJoinClause {
  AstParquetSource left;
  AstParquetSource right;
  AstExprPtr condition;
};

struct AstSelectStatement {
  std::vector<AstExprPtr> select_list;
  AstParquetSource from;              // single-table FROM; unused when `join` is set
  std::optional<AstJoinClause> join;  // two-table FROM ... JOIN ... ON ...
  AstExprPtr where;                   // null if no WHERE clause
  std::vector<AstExprPtr> group_by;
  std::vector<AstOrderByItem> order_by;
  std::optional<std::int64_t> limit;
};

}  // namespace kernellake::sql
