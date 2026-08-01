#pragma once

#include <memory>
#include <vector>

#include "kernellake/execution/expression_compiler.hpp"
#include "kernellake/execution/operator.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// Evaluates each of `items` against every batch from `child` via
// cudf::compute_column, assembling the results into a new batch whose
// schema is derived from `items`' names and result types.
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
  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::vector<NamedExpression> items_;
  std::shared_ptr<const Schema> output_schema_;
  ExpressionCompiler compiler_;
  std::vector<const cudf::ast::expression*> compiled_items_;
};

}  // namespace kernellake
