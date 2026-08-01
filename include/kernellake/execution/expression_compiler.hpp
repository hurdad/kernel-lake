#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <vector>

#include "kernellake/expression/expression.hpp"

namespace kernellake {

// Compiles a kernellake::Expression tree into a cudf::ast expression tree
// for row-wise GPU evaluation via cudf::compute_column() /
// cudf::apply_boolean_mask(). Owns the underlying cudf::ast::tree and any
// literal scalar objects it allocates; the returned references are valid
// only as long as this compiler instance stays alive (construct one per
// operator per open(), not per batch -- column indices are stable across
// batches from the same scan).
//
// AggregateExpression cannot be compiled here: aggregates are evaluated by
// ScalarAggregateOperator/HashAggregateOperator via cudf::reduce/
// cudf::groupby, not as a row-wise AST expression. CastExpression is
// supported only for the three widening targets cudf::ast itself supports
// (to INT64/UINT64/FLOAT64); anything else throws ExecutionError naming the
// unsupported target rather than silently miscompiling.
class ExpressionCompiler {
 public:
  [[nodiscard]] const cudf::ast::expression& compile(const Expression& expr);

 private:
  const cudf::ast::expression& compile_column(const ColumnExpression& expr);
  const cudf::ast::expression& compile_literal(const LiteralExpression& expr);
  const cudf::ast::expression& compile_binary(const BinaryExpression& expr);
  const cudf::ast::expression& compile_unary(const UnaryExpression& expr);
  const cudf::ast::expression& compile_between(const BetweenExpression& expr);
  const cudf::ast::expression& compile_cast(const CastExpression& expr);

  const cudf::ast::expression& make_literal(const DataType& type, const LiteralStorage& value, bool is_valid);

  cudf::ast::tree tree_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
};

}  // namespace kernellake
