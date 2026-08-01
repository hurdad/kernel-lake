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

struct AstColumnRef {
  std::string name;
};

// SELECT * — expanded against the FROM schema during binding.
struct AstStar {};

enum class AstLiteralKind {
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

enum class AstBinaryOp {
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

enum class AstUnaryOp {
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

enum class AstAggregateFunc {
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

struct AstExpr {
  std::variant<AstColumnRef, AstStar, AstLiteral, AstBinary, AstUnary, AstBetween, AstAggregate> node;
  std::optional<std::string> alias;
};

struct AstOrderByItem {
  AstExprPtr expr;
  bool ascending = true;
};

// The MVP's only supported data source: FROM read_parquet('path' [, 'path2', ...]).
struct AstParquetSource {
  std::vector<std::string> paths;
};

struct AstSelectStatement {
  std::vector<AstExprPtr> select_list;
  AstParquetSource from;
  AstExprPtr where;  // null if no WHERE clause
  std::vector<AstExprPtr> group_by;
  std::vector<AstOrderByItem> order_by;
  std::optional<std::int64_t> limit;
};

}  // namespace kernellake::sql
