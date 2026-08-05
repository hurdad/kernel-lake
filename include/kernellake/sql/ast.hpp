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

// One additional source joined onto the running left-deep chain so far,
// e.g. in `A JOIN B ON c1 JOIN C ON c2`, the chain is `first = A`,
// `steps = [{B, c1}, {C, c2}]` -- matching exactly how the underlying
// hsql SQL parser itself left-associates a multi-way JOIN into nested
// `(A JOIN B) JOIN C` TableRefs (see parser.cpp's `convert_join_chain()`),
// so no parser-side re-association is needed, just flattening. `condition`
// is bound against the *combined* schema of every source before this one
// plus this step's own `source` -- see binder.cpp.
struct AstJoinStep {
  AstParquetSource source;
  AstExprPtr condition;
};

// `FROM read_parquet(...) AS a JOIN read_parquet(...) AS b ON <c1> [JOIN
// read_parquet(...) AS c ON <c2> ...]`. Each step is a two-table INNER
// JOIN with a single equality key against the running combined schema so
// far -- see docs/ARCHITECTURE.md for the full scope (both sides of every
// step must be aliased `read_parquet(...)` sources, no comma-style joins).
struct AstJoinClause {
  AstParquetSource first;
  std::vector<AstJoinStep> steps;  // at least one
};

struct AstSelectStatement {
  std::vector<AstExprPtr> select_list;
  AstParquetSource from;              // single-table FROM; unused when `join` is set
  std::optional<AstJoinClause> join;  // FROM ... JOIN ... ON ... [JOIN ... ON ...]
  AstExprPtr where;                   // null if no WHERE clause
  std::vector<AstExprPtr> group_by;
  std::vector<AstOrderByItem> order_by;
  std::optional<std::int64_t> limit;
};

}  // namespace kernellake::sql
