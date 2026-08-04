#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution_gpu/expression_compiler.hpp"
#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// The GROUP BY case: consumes every batch from `child`, incrementally
// folding each one into a running partial-aggregate result (`accumulated_`),
// then produces exactly one output batch (the final grouped result) on the
// following next() call. Peak memory scales with distinct keys seen, not
// cumulative input rows -- same intent as before, but no longer implemented
// via cudf::groupby::streaming_groupby (see below for why).
//
// Each incoming batch is aggregated on its own with a plain, one-shot
// `cudf::groupby::groupby` (cost scales with that batch's actual row count
// and actual distinct-key count), then folded into `accumulated_` by
// concatenating [accumulated_, this batch's partial result] and running
// another plain groupby over that (cost scales with accumulated_'s
// actual size so far, which stays small for the common low-cardinality
// case -- TPC-H's GROUP BYs included). This replaces an earlier design
// built on cudf::groupby::streaming_groupby: profiling a real SF1000 TPC-H
// Q1 run found that design spending 67.5 of 106.9 total seconds inside
// HashAggregate despite the query having only 3 real distinct groups --
// root-caused to `max_distinct_keys` doubly serving as both
// streaming_groupby's persistent hash-table capacity *and* (an unrelated,
// cudf-internal encoding-scheme detail) the max row count a single
// aggregate() call accepts, so every ~33M-row scan pass had to be sliced
// into ~4 separate calls, each apparently carrying cost proportional to
// that *capacity* rather than to the query's real cardinality (raising
// `max_distinct_keys` to need fewer/larger calls made each call
// proportionally more expensive too -- net wash -- and OOM'd well before
// reaching the pass size outright; see docs/ROADMAP.md's "Not yet
// started" for the full investigation). SUM/MIN/MAX are all
// associative/self-combinable, so re-aggregating already-partially-
// aggregated columns with the same aggregation is correct -- this is why
// the earlier COUNT/AVG-via-SUM-of-ones fix (see ValueColumnKind /
// AggregateOutputKind below) is also what makes this design apply
// uniformly with no COUNT/AVG special-casing: every physical value column
// this operator ever aggregates is a SUM, MIN, or MAX by the time it gets
// here.
//
// `max_distinct_keys` now means exactly what its name says: `accumulated_`
// exceeding it throws (checked after every merge) -- unlike the old
// per-call row-count coupling above, this bounds real result cardinality
// directly, with no cardinality estimation needed (that stays
// cost-based-optimization territory, explicitly out of MVP scope) --
// hence still a generous fixed default, see kDefaultMaxDistinctKeys.
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
  struct CompiledLike;         // ditto.

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
  // `like_expr` handles a LIKE/NOT LIKE expression appearing inside a CASE
  // branch or as a plain (non-CASE) group-by key/aggregate argument --
  // cudf::ast has no LIKE-equivalent operator at all (same reason
  // FilterOperator special-cases top-level WHERE LIKE conjuncts instead of
  // routing them through its own AST compiler), so it's evaluated directly
  // via cudf::strings::like(), mirroring FilterOperator::evaluate_like()'s
  // exact algorithm.
  struct CompiledExpr {
    std::optional<cudf::size_type> source_column_index;
    std::shared_ptr<cudf::scalar> literal_scalar;  // plain literal (see cudf_adapter.hpp's literal_to_scalar)
    const cudf::ast::expression* expr = nullptr;
    std::shared_ptr<CompiledCase> case_expr;
    std::shared_ptr<CompiledDecimalCast> decimal_cast;
    std::shared_ptr<CompiledLike> like_expr;
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

  struct CompiledLike {
    CompiledExpr value;
    std::string pattern;
    bool negated;
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

  // Every physical value column this operator ever aggregates reduces to
  // one of these three, by the time ValueColumnKind above has materialized
  // it -- see the class comment for why this makes the plain-groupby
  // partial-then-merge design apply uniformly with no COUNT/AVG
  // special-casing.
  enum class PhysicalAggKind { Sum, Min, Max };

  [[nodiscard]] static std::unique_ptr<cudf::groupby_aggregation> make_physical_aggregation(
      PhysicalAggKind kind);
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
  [[nodiscard]] std::unique_ptr<cudf::column> materialize_value_column(ValueColumnKind kind,
                                                                       const CompiledExpr& compiled,
                                                                       const DeviceBatch& batch,
                                                                       ExecutionContext& context);
  void process_batch(const DeviceBatch& batch, ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::table> build_combined_columns(const DeviceBatch& batch,
                                                                    ExecutionContext& context);
  // Runs one plain (non-streaming) cudf::groupby::groupby over `key_view`/
  // `value_view` (physical_agg_kind_-driven aggregations, one per value
  // column) and reassembles the result into a single table (keys, then
  // aggregate results, same column order as `key_view`+`value_view` --
  // i.e. the same layout process_batch()'s caller expects, whether this is
  // a fresh batch's partial result or accumulated_'s post-merge
  // replacement).
  [[nodiscard]] std::unique_ptr<cudf::table> run_groupby_and_assemble(const cudf::table_view& key_view,
                                                                      const cudf::table_view& value_view,
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
  // Parallel to compiled_aggregate_args_/value_column_kind_ -- which
  // aggregation run_groupby_and_assemble() requests for that physical value
  // column.
  std::vector<PhysicalAggKind> physical_agg_kind_;

  // One entry per logical aggregate (parallel to `aggregates_`), telling
  // finalize() how many consecutive physical result columns that aggregate
  // consumed and how to combine them into its single output column.
  std::vector<AggregateOutputKind> aggregate_output_kind_;

  // The running partial-aggregate result, folded in one batch at a time by
  // process_batch() -- null until the first batch arrives (see the class
  // comment). Column layout: group_by_.size() key columns, then one column
  // per physical_agg_kind_ entry, same order throughout.
  std::unique_ptr<cudf::table> accumulated_;
  bool any_batch_seen_ = false;
  bool produced_ = false;
};

}  // namespace kernellake
