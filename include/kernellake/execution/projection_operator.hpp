#pragma once

#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution/expression_compiler.hpp"
#include "kernellake/execution/operator.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// Evaluates each of `items` against every batch from `child`, assembling
// the results into a new batch whose schema is derived from `items`' names
// and result types.
//
// A plain column reference (e.g. `SELECT region ...`, no computation) is
// copied directly rather than routed through cudf::ast::compute_column:
// cudf's AST evaluator can only materialize fixed-width output columns, so
// running a STRING (or other variable-width) column reference through it
// aborts with "Invalid, non-fixed-width type" even though no actual
// computation was requested. Genuinely computed expressions still go
// through compute_column as before.
//
// A CASE expression is evaluated via a chain of cudf::copy_if_else calls
// (cudf::ast has no ternary/branching operator) -- see
// docs/ARCHITECTURE.md. Each branch's own condition/result must itself be
// a plain column or AST-compilable expression; a LIKE or nested CASE inside
// a CASE branch is not supported in this version (the ExpressionCompiler's
// "unrecognized expression type" error is the natural failure point for
// that).
class ProjectionOperator final : public PhysicalOperator {
 public:
  ProjectionOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                     std::vector<NamedExpression> items);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "Projection"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  // A plain column index to copy, a plain literal to broadcast, or a
  // compiled AST expression to evaluate -- used both for ordinary
  // projection items and for a CASE branch's own condition/result.
  //
  // The literal case matters beyond CASE branches: cudf::ast can only
  // produce a fixed-width *output* column, so even `SELECT 'foo' FROM ...`
  // (a STRING literal as a whole projection item, not just a CASE branch)
  // would hit "Invalid, non-fixed-width type" if routed through
  // compute_column -- a string literal is only valid as an *intermediate*
  // AST node (e.g. one side of `region = 'A'`), never as the compiled
  // tree's root when its result type isn't fixed-width.
  struct CompiledValue {
    std::optional<cudf::size_type> source_column_index;
    std::shared_ptr<cudf::scalar> literal_scalar;
    const cudf::ast::expression* expr = nullptr;
  };

  struct CompiledCaseBranch {
    CompiledValue condition;
    CompiledValue result;
  };

  struct CompiledCase {
    std::vector<CompiledCaseBranch> branches;
    std::optional<CompiledValue> else_value;  // nullopt: NULL when no branch matches
    DataType result_type{TypeId::Boolean};
  };

  // Exactly one of `value`/`case_expr` is engaged per item.
  struct CompiledItem {
    CompiledValue value;
    std::unique_ptr<CompiledCase> case_expr;
  };

  [[nodiscard]] CompiledValue compile_value(const Expression& expr);
  [[nodiscard]] CompiledItem compile_item(const Expression& expr);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_value(const CompiledValue& value,
                                                                const cudf::table_view& batch,
                                                                ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_case(const CompiledCase& case_expr,
                                                               const cudf::table_view& batch,
                                                               ExecutionContext& context);

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::vector<NamedExpression> items_;
  std::shared_ptr<const Schema> output_schema_;
  ExpressionCompiler compiler_;
  std::vector<CompiledItem> compiled_items_;
};

}  // namespace kernellake
