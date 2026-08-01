#pragma once

#include <memory>

#include "kernellake/execution/expression_compiler.hpp"
#include "kernellake/execution/operator.hpp"
#include "kernellake/expression/expression.hpp"

namespace kernellake {

// Filters each batch from `child` by a compiled boolean predicate,
// evaluated via cudf::compute_column + cudf::apply_boolean_mask. Batches
// that filter down to zero rows are skipped rather than returned, so
// downstream operators only ever see non-empty batches.
class FilterOperator final : public PhysicalOperator {
 public:
  FilterOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child, ExpressionPtr predicate);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "Filter"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  ExpressionPtr predicate_;
  ExpressionCompiler compiler_;
  const cudf::ast::expression* compiled_predicate_ = nullptr;
};

}  // namespace kernellake
