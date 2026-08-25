#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution_gpu/execution_context.hpp"
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
// (to INT64/UINT64/FLOAT64) plus casting *from* DECIMAL to any of those
// (cudf::ast's CAST_TO_* operators accept a DECIMAL operand natively);
// casting *to* DECIMAL has no cudf::ast operator at all and must be
// materialized outside the AST tree (see ProjectionOperator/
// HashAggregateOperator's decimal-cast fast path) -- anything else throws
// ExecutionError naming the unsupported target rather than silently
// miscompiling.
class ExpressionCompiler {
 public:
  // `context` is threaded down to every literal-scalar construction
  // (make_literal() and its callers below) so those cudf::scalar
  // allocations use this query's own stream/memory_resource rather than
  // cudf's process-wide ambient defaults -- see cudf_adapter.hpp's
  // literal_to_scalar() for the same rationale, which applies identically
  // here.
  [[nodiscard]] const cudf::ast::expression& compile(const Expression& expr, ExecutionContext& context);

  // Compiles a two-sided predicate (currently only
  // SemiAntiJoinOperator's own residual_predicate_, TPC-H Q21's `l2.l_
  // suppkey <> l1.l_suppkey` shape) for cudf::mixed_left_semi_join()/
  // mixed_left_anti_join(), which evaluate an ast::expression against a
  // *pair* of tables rather than one. `left_field_count` is the same
  // threshold HashJoinNode::residual_predicate()'s own column-index
  // convention already uses: a ColumnExpression index below it compiles
  // to `cudf::ast::table_reference::LEFT` at that index; at or above it,
  // to `table_reference::RIGHT` at `index - left_field_count`. Every
  // other Expression node type compiles identically to the plain
  // single-table compile() above (BinaryExpression/UnaryExpression/etc.
  // don't care which table a nested ColumnExpression's reference resolves
  // against, only compile_column() does) -- implemented by setting
  // `two_table_left_field_count_` for the duration of this call and
  // delegating straight to compile(), rather than a separate parallel
  // recursive dispatcher.
  [[nodiscard]] const cudf::ast::expression& compile_two_table(const Expression& expr,
                                                               std::size_t left_field_count,
                                                               ExecutionContext& context);

 private:
  const cudf::ast::expression& compile_column(const ColumnExpression& expr);
  const cudf::ast::expression& compile_literal(const LiteralExpression& expr, ExecutionContext& context);
  const cudf::ast::expression& compile_binary(const BinaryExpression& expr, ExecutionContext& context);
  const cudf::ast::expression& compile_unary(const UnaryExpression& expr, ExecutionContext& context);
  const cudf::ast::expression& compile_between(const BetweenExpression& expr, ExecutionContext& context);
  const cudf::ast::expression& compile_cast(const CastExpression& expr, ExecutionContext& context);

  const cudf::ast::expression& make_literal(const DataType& type, const LiteralStorage& value, bool is_valid,
                                            ExecutionContext& context);

  cudf::ast::tree tree_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  // Set only for the duration of a compile_two_table() call -- nullopt
  // (the default, normal single-table mode) for every other caller, so
  // compile_column() below behaves exactly as it always has for them.
  std::optional<std::size_t> two_table_left_field_count_;
};

}  // namespace kernellake
