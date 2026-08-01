#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kernellake/execution/expression_compiler.hpp"
#include "kernellake/execution/operator.hpp"
#include "kernellake/expression/expression.hpp"

namespace kernellake {

// Filters each batch from `child` by `predicate`, evaluated via
// cudf::compute_column + cudf::apply_boolean_mask. Batches that filter down
// to zero rows are skipped rather than returned, so downstream operators
// only ever see non-empty batches.
//
// cudf::ast has no LIKE operator, so a `LikeExpression` (or its negation)
// appearing as one of the predicate's top-level AND-connected conjuncts
// (e.g. `WHERE region LIKE 'A%' AND amount > 10`) is evaluated separately
// via cudf::strings::like and combined with the AST-evaluated remainder via
// a column-level AND -- see docs/ARCHITECTURE.md. A LIKE anywhere else
// (inside OR, arithmetic, or the SELECT list) is not supported; the
// ExpressionCompiler's "unrecognized expression type" error is the natural
// failure point for that.
class FilterOperator final : public PhysicalOperator {
 public:
  FilterOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child, ExpressionPtr predicate);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "Filter"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  struct CompiledLikeConjunct {
    std::optional<cudf::size_type> value_column_index;
    const cudf::ast::expression* value_expr = nullptr;  // set iff value_column_index is not
    std::string pattern;
    bool negated;
  };

  [[nodiscard]] std::unique_ptr<cudf::column> evaluate_like(const CompiledLikeConjunct& like,
                                                            const cudf::table_view& batch,
                                                            ExecutionContext& context);

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  ExpressionPtr predicate_;
  ExpressionCompiler compiler_;
  const cudf::ast::expression* compiled_ast_conjuncts_ = nullptr;  // null if every conjunct is a LIKE
  std::vector<CompiledLikeConjunct> compiled_like_conjuncts_;
};

}  // namespace kernellake
