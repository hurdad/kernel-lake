#pragma once

#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution/expression_compiler.hpp"
#include "kernellake/execution/operator.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// The no-GROUP-BY aggregate case: consumes every batch from `child`, then
// produces exactly one output batch (a single row) on the following next()
// call, then reports exhausted. Each item in `aggregates` must be an
// AggregateExpression.
//
// Partial results are accumulated across batches without retaining input
// batches (per the spec's "accumulate partial reductions" requirement):
// SUM/MIN/MAX use cudf::reduce's `init` parameter to fold each new batch
// directly into the running scalar; COUNT/AVG's denominator is accumulated
// as a plain host-side counter (cudf::reduce has no init-based COUNT/MEAN).
// A COUNT(*)/SUM/etc. over zero input rows produces NULL, not zero.
class ScalarAggregateOperator final : public PhysicalOperator {
public:
  ScalarAggregateOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                           std::vector<NamedExpression> aggregates);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "ScalarAggregate"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

private:
  struct Accumulator {
    AggregateFunction function;
    ExpressionPtr argument;  // null only for CountStar
    DataType result_type;
    // A plain column reference (e.g. `MIN(region)`) is copied directly
    // instead of routed through cudf::ast::compute_column: cudf's AST
    // evaluator can only materialize fixed-width output columns, so a
    // STRING argument column would abort with "Invalid, non-fixed-width
    // type" even though no actual computation was requested.
    std::optional<cudf::size_type> argument_column_index;
    const cudf::ast::expression* compiled_argument = nullptr;
    std::unique_ptr<cudf::scalar> running_value;  // Sum/Min/Max/Avg's running sum
    std::int64_t running_count = 0;               // Count/CountStar/Avg's denominator
  };

  [[nodiscard]] std::unique_ptr<cudf::column> materialize_argument(Accumulator& state, const DeviceBatch& batch,
                                                                     ExecutionContext& context);
  void process_batch(Accumulator& state, const DeviceBatch& batch, ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> finalize(Accumulator& state, ExecutionContext& context);

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::vector<NamedExpression> aggregates_;
  std::shared_ptr<const Schema> output_schema_;
  ExpressionCompiler compiler_;
  std::vector<Accumulator> states_;
  bool produced_ = false;
};

}  // namespace kernellake
