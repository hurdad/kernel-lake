#pragma once

#include <cudf/groupby.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution/expression_compiler.hpp"
#include "kernellake/execution/operator.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// The GROUP BY case: consumes every batch from `child`, feeding each one
// into a cudf::groupby::streaming_groupby (bounded-memory partial
// aggregation across batches -- peak memory scales with distinct keys seen,
// not cumulative input rows), then produces exactly one output batch (the
// final grouped result) on the following next() call.
//
// `max_distinct_keys` bounds the number of distinct group-by key
// combinations streaming_groupby will track; exceeding it throws. There is
// no cardinality estimation yet (that is cost-based-optimization territory,
// explicitly out of MVP scope), so this defaults to a generous fixed value
// -- see kDefaultMaxDistinctKeys.
class HashAggregateOperator final : public PhysicalOperator {
public:
  static constexpr cudf::size_type kDefaultMaxDistinctKeys = 10'000'000;

  HashAggregateOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                         std::vector<NamedExpression> group_by, std::vector<NamedExpression> aggregates,
                         cudf::size_type max_distinct_keys = kDefaultMaxDistinctKeys);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "HashAggregate"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

private:
  // A plain column reference (e.g. `GROUP BY region`) is copied directly
  // rather than routed through cudf::ast::compute_column: cudf's AST
  // evaluator can only materialize fixed-width output columns, so a STRING
  // (or other variable-width) key column would abort with "Invalid,
  // non-fixed-width type" even though no actual computation was requested.
  struct CompiledExpr {
    std::optional<cudf::size_type> source_column_index;
    const cudf::ast::expression* expr = nullptr;
  };

  [[nodiscard]] CompiledExpr compile_expr(const Expression& expr);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize(const CompiledExpr& compiled, const DeviceBatch& batch,
                                                           ExecutionContext& context);
  void process_batch(const DeviceBatch& batch, ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::table> build_combined_columns(const DeviceBatch& batch,
                                                                     ExecutionContext& context);

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::vector<NamedExpression> group_by_;
  std::vector<NamedExpression> aggregates_;
  cudf::size_type max_distinct_keys_;
  std::shared_ptr<const Schema> output_schema_;

  ExpressionCompiler compiler_;
  std::vector<CompiledExpr> compiled_group_by_;
  // CountStar entries are a copy of compiled_group_by_.front() (reused, not
  // aliased) so every aggregate still gets its own materialized column at
  // its own table slot -- see the comment in open().
  std::vector<CompiledExpr> compiled_aggregate_args_;

  // cudf::groupby's COUNT aggregations always produce cudf::size_type
  // (INT32) output regardless of requested type, but KernelLake's binder
  // declares COUNT/COUNT(*) results as INT64 -- entries here mark which
  // result columns need an explicit int32->int64 cast after finalize() so
  // the output DeviceBatch's actual column types match output_schema_.
  std::vector<bool> result_is_count_;

  std::unique_ptr<cudf::groupby::streaming_groupby> streaming_;
  bool any_batch_seen_ = false;
  bool produced_ = false;
};

}  // namespace kernellake
