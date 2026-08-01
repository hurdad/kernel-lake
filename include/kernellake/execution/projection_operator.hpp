#pragma once

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
  // One entry per output item: either a source column index to copy
  // directly (plain column reference) or a compiled AST expression to
  // evaluate (anything else).
  struct CompiledItem {
    std::optional<std::size_t> source_column_index;
    const cudf::ast::expression* expr = nullptr;
  };

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::vector<NamedExpression> items_;
  std::shared_ptr<const Schema> output_schema_;
  ExpressionCompiler compiler_;
  std::vector<CompiledItem> compiled_items_;
};

}  // namespace kernellake
