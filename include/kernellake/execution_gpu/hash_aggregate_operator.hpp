#pragma once

#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution_gpu/expression_compiler.hpp"
#include "kernellake/execution_gpu/operator.hpp"
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
  struct CompiledCase;         // defined below; forward-declared so CompiledExpr can hold a shared_ptr to it.
  struct CompiledDecimalCast;  // ditto.

  // A plain column reference (e.g. `GROUP BY region`) is copied directly
  // rather than routed through cudf::ast::compute_column: cudf's AST
  // evaluator can only materialize fixed-width output columns, so a STRING
  // (or other variable-width) key column would abort with "Invalid,
  // non-fixed-width type" even though no actual computation was requested.
  //
  // `case_expr` is engaged instead of the above for a CASE expression --
  // reachable as a group-by key via `GROUP BY <alias>` resolving to a CASE
  // in the SELECT list (see binder.cpp), since a CASE has no column name of
  // its own to write directly in GROUP BY. `case_expr` is a shared_ptr, not
  // unique_ptr, so CompiledExpr stays copyable -- Avg's sum and count slots
  // below share one copy of the same compiled argument.
  //
  // `decimal_cast` is the same idea for `GROUP BY <alias>` resolving to a
  // `CAST(... AS DECIMAL(p,s))` in the SELECT list: cudf::ast has no
  // CAST_TO_DECIMAL* operator, so it's materialized directly via
  // cudf::cast() instead of through `expr`. See ProjectionOperator's
  // identical fast path and docs/ARCHITECTURE.md.
  struct CompiledExpr {
    std::optional<cudf::size_type> source_column_index;
    std::shared_ptr<cudf::scalar> literal_scalar;  // plain literal (see cudf_adapter.hpp's literal_to_scalar)
    const cudf::ast::expression* expr = nullptr;
    std::shared_ptr<CompiledCase> case_expr;
    std::shared_ptr<CompiledDecimalCast> decimal_cast;
  };

  struct CompiledCaseBranch {
    CompiledExpr condition;
    CompiledExpr result;
  };

  struct CompiledCase {
    std::vector<CompiledCaseBranch> branches;
    std::optional<CompiledExpr> else_value;  // nullopt: NULL when no branch matches
    DataType result_type{TypeId::Boolean};
  };

  struct CompiledDecimalCast {
    CompiledExpr operand;
    DataType target_type;
  };

  // How to materialize a physical aggregate-request's value column (one
  // entry per `requests`/`compiled_aggregate_args_` slot -- Avg contributes
  // two of these per logical aggregate, see AggregateOutputKind below).
  //   Expression: the aggregate's own argument, as compiled by compile_expr
  //     (used for SUM/MIN/MAX, and for Avg's sum-side slot).
  //   CountStarOnes: an all-valid INT64 column of 1s, sized to the batch --
  //     COUNT(*) counts every row regardless of any column's nulls.
  //   CountColumnOnes: an INT64 column of 1s carrying the *argument*
  //     column's null mask -- summing it gives COUNT(argument)'s
  //     null-excluding semantics. Also used for Avg's count-side slot.
  enum class ValueColumnKind { Expression, CountStarOnes, CountColumnOnes };

  // How finalize() turns a logical aggregate's physical result column(s)
  // back into exactly one output column.
  //   Direct: SUM/MIN/MAX, or the COUNT-like SUM-of-ones trick above --
  //     either way, a single physical result column, used as-is (already
  //     the right type -- no int32->int64 cast needed, since ValueColumnKind
  //     above already picked an INT64 input column for COUNT-like cases).
  //   Average: two physical result columns (sum, then count -- both from
  //     the SUM-of-ones trick, not cudf's native MEAN aggregation) that
  //     finalize() divides itself. See the comment on AggregateFunction::Avg
  //     in open() for why: cudf's own grouped/streaming MEAN divides by an
  //     internal COUNT_VALID that is *also* a 32-bit cudf::size_type
  //     accumulator, so it silently wraps around under the exact same
  //     large-single-group-cardinality condition as COUNT -- confirmed by a
  //     real SF1000 TPC-H Q1 run producing a negative avg_disc alongside
  //     count_order's own wraparound (see docs/ROADMAP.md).
  enum class AggregateOutputKind { Direct, Average };

  [[nodiscard]] CompiledExpr compile_expr(const Expression& expr);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize(const CompiledExpr& compiled,
                                                          const DeviceBatch& batch,
                                                          ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_case(const CompiledCase& case_expr,
                                                               const DeviceBatch& batch,
                                                               ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_value_column(ValueColumnKind kind,
                                                                       const CompiledExpr& compiled,
                                                                       const DeviceBatch& batch,
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
  // One entry per physical aggregate-request slot (see ValueColumnKind) --
  // Avg contributes two consecutive entries, everything else contributes
  // one, so this can be longer than `aggregates_`.
  std::vector<CompiledExpr> compiled_aggregate_args_;
  std::vector<ValueColumnKind> value_column_kind_;

  // One entry per logical aggregate (parallel to `aggregates_`), telling
  // finalize() how many consecutive physical result columns that aggregate
  // consumed and how to combine them into its single output column.
  std::vector<AggregateOutputKind> aggregate_output_kind_;

  std::unique_ptr<cudf::groupby::streaming_groupby> streaming_;
  bool any_batch_seen_ = false;
  bool produced_ = false;
};

}  // namespace kernellake
