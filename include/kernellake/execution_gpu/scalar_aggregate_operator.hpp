#pragma once

#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution_gpu/expression_compiler.hpp"
#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// The no-GROUP-BY aggregate case: consumes every batch from `child`, then
// produces exactly one output batch (a single row) on the following next()
// call, then reports exhausted. Each item in `aggregates` must be an
// AggregateExpression.
//
// Partial results are accumulated across batches without retaining input
// batches (per the spec's "accumulate partial reductions" requirement) and
// entirely device-side, never reading a value back to host per batch:
// SUM/MIN/MAX use cudf::reduce's `init` parameter to fold each new batch
// directly into the running scalar; COUNT(x)/AVG(x)'s denominator uses the
// same mechanism over a synthesized ones-column (cudf::reduce has no
// init-based COUNT/MEAN of its own -- see materialize_count_ones(), the
// same SUM-of-ones trick HashAggregateOperator::materialize_value_column's
// CountColumnOnes already uses). CountStar is the one exception: it stays
// plain host bookkeeping, since batch.row_count() is host-side metadata
// already and involves no GPU work to begin with. SUM/MIN/MAX/AVG over
// zero input rows produce NULL, not zero; COUNT(*) and COUNT(x) produce 0
// -- see CountStarCountsRowsAcrossBatchesIncludingZero and
// SumOfEmptyInputIsNullNotZero in the corresponding test file.
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
  struct CompiledCase;         // defined below; forward-declared so CompiledExpr can hold a shared_ptr to it.
  struct CompiledDecimalCast;  // ditto.
  struct CompiledLike;         // ditto.
  struct CompiledExtract;      // ditto.

  // Mirrors HashAggregateOperator::CompiledExpr exactly (same rationale for
  // each fast path -- see that class's own comments): a plain column
  // reference or literal is materialized directly rather than routed
  // through cudf::ast::compute_column (which can only produce fixed-width
  // output), `case_expr`/`decimal_cast` handle a CASE or CAST-to-DECIMAL
  // aggregate argument (e.g. `SUM(CASE WHEN ... THEN ... ELSE ... END)`,
  // TPC-H Q14's shape) the same way ProjectionOperator and
  // HashAggregateOperator already do, and `expr` is the ordinary
  // cudf::ast-compiled path for everything else (e.g. `SUM(price *
  // discount)`).
  struct CompiledExpr {
    std::optional<cudf::size_type> source_column_index;
    std::shared_ptr<cudf::scalar> literal_scalar;
    const cudf::ast::expression* expr = nullptr;
    std::shared_ptr<CompiledCase> case_expr;
    std::shared_ptr<CompiledDecimalCast> decimal_cast;
    std::shared_ptr<CompiledLike> like_expr;
    std::shared_ptr<CompiledExtract> extract_expr;
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

  // Mirrors FilterOperator::evaluate_like()'s exact algorithm (cudf::ast has
  // no LIKE-equivalent operator) -- see HashAggregateOperator's identical
  // fast path.
  struct CompiledLike {
    CompiledExpr value;
    std::string pattern;
    bool negated;
  };

  // Mirrors HashAggregateOperator::CompiledExtract exactly -- see that
  // struct's own comment.
  struct CompiledExtract {
    CompiledExpr operand;
    DatePart part;
  };

  struct Accumulator {
    AggregateFunction function;
    DataType result_type;
    CompiledExpr compiled_argument;               // unused for CountStar (no argument column)
    std::unique_ptr<cudf::scalar> running_value;  // Sum/Min/Max/Avg's running sum
    // Count/Avg's denominator, accumulated entirely device-side via the same
    // SUM-of-a-synthesized-ones-column trick HashAggregateOperator's
    // CountColumnOnes already uses -- see materialize_count_ones() -- so
    // per-batch accumulation never reads a value back to host (a
    // cudf::reduce init-overload fold, exactly like running_value above).
    // nullptr means "no batch has contributed a valid row yet", not zero --
    // see finalize()'s own handling.
    std::unique_ptr<cudf::scalar> running_count_scalar;
    // CountStar only: batch.row_count() is host-side metadata already (no
    // GPU work, no sync at all), so this stays plain host bookkeeping --
    // unlike Count(x)/Avg(x) above, there is nothing to move device-side
    // here.
    std::int64_t running_count = 0;
  };

  [[nodiscard]] CompiledExpr compile_expr(const Expression& expr);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize(const CompiledExpr& compiled,
                                                          const DeviceBatch& batch,
                                                          ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_case(const CompiledCase& case_expr,
                                                               const DeviceBatch& batch,
                                                               ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_like(const CompiledLike& like_expr,
                                                               const DeviceBatch& batch,
                                                               ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_extract(const CompiledExtract& extract_expr,
                                                                  const DeviceBatch& batch,
                                                                  ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_argument(Accumulator& state,
                                                                   const DeviceBatch& batch,
                                                                   ExecutionContext& context);
  // An all-valid INT64 ones column (batch.row_count() rows), carrying
  // `argument`'s own null mask -- summing it gives argument's non-null row
  // count for this batch, the same CountColumnOnes trick
  // HashAggregateOperator::materialize_value_column uses, duplicated here
  // rather than shared since ScalarAggregateOperator has no
  // ValueColumnKind dispatch to hang a shared helper off of.
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_count_ones(const cudf::column& argument,
                                                                     const DeviceBatch& batch,
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
