#include "kernellake/execution_gpu/hash_aggregate_operator.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <numeric>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cudf_adapter.hpp"
#include "kernellake/expression/expression.hpp"

namespace kernellake {

namespace {

std::shared_ptr<const Schema> build_output_schema(const std::vector<NamedExpression>& group_by,
                                                  const std::vector<NamedExpression>& aggregates) {
  std::vector<Field> fields;
  fields.reserve(group_by.size() + aggregates.size());
  for (const NamedExpression& item : group_by) fields.push_back(Field{item.name, item.expr->result_type()});
  for (const NamedExpression& item : aggregates) fields.push_back(Field{item.name, item.expr->result_type()});
  return std::make_shared<const Schema>(Schema(std::move(fields)));
}

std::unique_ptr<cudf::groupby_aggregation> to_streaming_aggregation(AggregateFunction function) {
  switch (function) {
    case AggregateFunction::Sum:
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case AggregateFunction::Min:
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case AggregateFunction::Max:
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case AggregateFunction::Count:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
    case AggregateFunction::CountStar:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
    case AggregateFunction::Avg:
      return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  }
  throw ExecutionError("unreachable: unknown AggregateFunction");
}

}  // namespace

HashAggregateOperator::HashAggregateOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                                             std::vector<NamedExpression> group_by,
                                             std::vector<NamedExpression> aggregates,
                                             cudf::size_type max_distinct_keys)
    : id_(id),
      child_(std::move(child)),
      group_by_(std::move(group_by)),
      aggregates_(std::move(aggregates)),
      max_distinct_keys_(max_distinct_keys),
      output_schema_(build_output_schema(group_by_, aggregates_)) {
  if (group_by_.empty()) {
    throw PlanningError("HashAggregateOperator requires at least one GROUP BY column");
  }
}

HashAggregateOperator::CompiledExpr HashAggregateOperator::compile_expr(const Expression& expr) {
  if (const auto* column_ref = dynamic_cast<const ColumnExpression*>(&expr)) {
    CompiledExpr compiled;
    compiled.source_column_index = static_cast<cudf::size_type>(column_ref->column_index());
    return compiled;
  }
  if (const auto* literal = dynamic_cast<const LiteralExpression*>(&expr)) {
    CompiledExpr compiled;
    compiled.literal_scalar = literal_to_scalar(*literal);
    return compiled;
  }
  if (const auto* case_expr = dynamic_cast<const CaseExpression*>(&expr)) {
    auto compiled_case = std::make_shared<CompiledCase>();
    compiled_case->result_type = case_expr->result_type();
    compiled_case->branches.reserve(case_expr->when_then().size());
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      compiled_case->branches.push_back(
          CompiledCaseBranch{compile_expr(*branch.condition), compile_expr(*branch.result)});
    }
    if (case_expr->else_branch() != nullptr) {
      compiled_case->else_value = compile_expr(*case_expr->else_branch());
    }
    CompiledExpr compiled;
    compiled.case_expr = std::move(compiled_case);
    return compiled;
  }
  if (const auto* cast_expr = dynamic_cast<const CastExpression*>(&expr);
      cast_expr != nullptr && cast_expr->result_type().id == TypeId::Decimal) {
    auto decimal_cast = std::make_shared<CompiledDecimalCast>();
    decimal_cast->operand = compile_expr(*cast_expr->operand());
    decimal_cast->target_type = cast_expr->result_type();
    CompiledExpr compiled;
    compiled.decimal_cast = std::move(decimal_cast);
    return compiled;
  }
  CompiledExpr compiled;
  compiled.expr = &compiler_.compile(expr);
  return compiled;
}

std::unique_ptr<cudf::column> HashAggregateOperator::materialize(const CompiledExpr& compiled,
                                                                 const DeviceBatch& batch,
                                                                 ExecutionContext& context) {
  if (compiled.case_expr != nullptr) return materialize_case(*compiled.case_expr, batch, context);
  if (compiled.decimal_cast != nullptr) {
    const std::unique_ptr<cudf::column> operand = materialize(compiled.decimal_cast->operand, batch, context);
    return cudf::cast(operand->view(), to_cudf_type(compiled.decimal_cast->target_type), context.stream,
                      context.memory_resource);
  }
  if (compiled.source_column_index.has_value()) {
    return std::make_unique<cudf::column>(batch.view().column(*compiled.source_column_index), context.stream,
                                          context.memory_resource);
  }
  if (compiled.literal_scalar != nullptr) {
    return cudf::make_column_from_scalar(*compiled.literal_scalar,
                                         static_cast<cudf::size_type>(batch.row_count()), context.stream,
                                         context.memory_resource);
  }
  return cudf::compute_column(batch.view(), *compiled.expr, context.stream, context.memory_resource);
}

std::unique_ptr<cudf::column> HashAggregateOperator::materialize_case(const CompiledCase& case_expr,
                                                                      const DeviceBatch& batch,
                                                                      ExecutionContext& context) {
  // Folds from the last branch backward -- see the identical algorithm and
  // comment in ProjectionOperator::materialize_case.
  std::unique_ptr<cudf::column> result;
  if (case_expr.else_value.has_value()) {
    result = materialize(*case_expr.else_value, batch, context);
  } else {
    const std::unique_ptr<cudf::scalar> null_scalar = cudf::make_default_constructed_scalar(
        to_cudf_type(case_expr.result_type), context.stream, context.memory_resource);
    result = cudf::make_column_from_scalar(*null_scalar, static_cast<cudf::size_type>(batch.row_count()),
                                           context.stream, context.memory_resource);
  }
  for (auto it = case_expr.branches.rbegin(); it != case_expr.branches.rend(); ++it) {
    std::unique_ptr<cudf::column> condition = materialize(it->condition, batch, context);
    std::unique_ptr<cudf::column> branch_result = materialize(it->result, batch, context);
    result = cudf::copy_if_else(branch_result->view(), result->view(), condition->view(), context.stream,
                                context.memory_resource);
  }
  return result;
}

void HashAggregateOperator::open(ExecutionContext& context) {
  child_->open(context);

  compiled_group_by_.reserve(group_by_.size());
  for (const NamedExpression& item : group_by_) compiled_group_by_.push_back(compile_expr(*item.expr));

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.reserve(aggregates_.size());
  compiled_aggregate_args_.reserve(aggregates_.size());
  cudf::size_type next_index = static_cast<cudf::size_type>(group_by_.size());

  for (const NamedExpression& item : aggregates_) {
    const auto* aggregate = dynamic_cast<const AggregateExpression*>(item.expr.get());
    if (aggregate == nullptr) {
      throw ExecutionError(
          fmt::format("HashAggregateOperator item '{}' is not an AggregateExpression", item.name));
    }
    // COUNT(*) has no natural argument column; reuse group_by[0]'s compiled
    // form (materialized at its own dedicated table slot below, via
    // null_policy::INCLUDE so nulls in that column don't suppress the
    // count). Note: referencing an *existing* key column's index directly,
    // instead of materializing a fresh copy at its own index, silently
    // produced all-zero counts in testing -- streaming_groupby apparently
    // doesn't support a value column index that aliases a key index.
    const CompiledExpr compiled_argument = aggregate->function() == AggregateFunction::CountStar
                                               ? compiled_group_by_.front()
                                               : compile_expr(*aggregate->argument());
    compiled_aggregate_args_.push_back(compiled_argument);
    result_is_count_.push_back(aggregate->function() == AggregateFunction::Count ||
                               aggregate->function() == AggregateFunction::CountStar);
    requests.push_back(cudf::groupby::streaming_aggregation_request{
        next_index, to_streaming_aggregation(aggregate->function())});
    ++next_index;
  }

  std::vector<cudf::size_type> key_indices(group_by_.size());
  std::iota(key_indices.begin(), key_indices.end(), 0);
  streaming_ = std::make_unique<cudf::groupby::streaming_groupby>(key_indices, requests, max_distinct_keys_);
}

std::unique_ptr<cudf::table> HashAggregateOperator::build_combined_columns(const DeviceBatch& batch,
                                                                           ExecutionContext& context) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(compiled_group_by_.size() + compiled_aggregate_args_.size());
  for (const CompiledExpr& compiled : compiled_group_by_)
    columns.push_back(materialize(compiled, batch, context));
  for (const CompiledExpr& compiled : compiled_aggregate_args_) {
    columns.push_back(materialize(compiled, batch, context));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

void HashAggregateOperator::process_batch(const DeviceBatch& batch, ExecutionContext& context) {
  any_batch_seen_ = true;
  const std::unique_ptr<cudf::table> combined = build_combined_columns(batch, context);
  const cudf::table_view combined_view = combined->view();
  const cudf::size_type total_rows = combined_view.num_rows();

  // cudf::groupby::streaming_groupby::aggregate() requires a single call's
  // row count to not exceed max_distinct_keys_ -- its own doc comment
  // explains why: each in-flight batch row is encoded as
  // `max_distinct_keys + row_idx` inside the hash set, an implementation
  // detail of the encoding scheme, unrelated to this query's actual GROUP
  // BY cardinality (confirmed by a real failure: TPC-H Q1's own GROUP BY
  // has ~6 distinct values, but a real SF10 run still hit "Batch size
  // (59619013) exceeds max_distinct_keys (10000000)" because
  // ParquetScanOperator's own pass splitting is purely memory-based
  // (pass_read_limit_bytes), with no awareness of this separate,
  // row-count-based constraint). Re-slicing an oversized batch here, on
  // our side, is the fix -- not raising the global max_distinct_keys_
  // default, which would inflate the persistent hash table/
  // aggregation-results memory for *every* hash-aggregate query
  // (proportional to max_distinct_keys_, regardless of that query's real
  // cardinality -- see cudf/groupby.hpp's own streaming_groupby doc
  // comment: "the persistent state is sized to max_distinct_keys").
  if (total_rows <= max_distinct_keys_) {
    streaming_->aggregate(combined_view, context.stream);
    return;
  }
  std::vector<cudf::size_type> slice_indices;
  slice_indices.reserve(2 * ((total_rows / max_distinct_keys_) + 1));
  for (cudf::size_type offset = 0; offset < total_rows; offset += max_distinct_keys_) {
    slice_indices.push_back(offset);
    slice_indices.push_back(std::min(offset + max_distinct_keys_, total_rows));
  }
  const std::vector<cudf::table_view> chunks = cudf::slice(combined_view, slice_indices, context.stream);
  for (const cudf::table_view& chunk : chunks) {
    streaming_->aggregate(chunk, context.stream);
  }
}

std::optional<DeviceBatch> HashAggregateOperator::next(ExecutionContext& context) {
  if (produced_) return std::nullopt;

  while (std::optional<DeviceBatch> batch = child_->next(context)) {
    process_batch(*batch, context);
  }
  produced_ = true;

  if (!any_batch_seen_) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(output_schema_->field_count());
    for (const Field& field : output_schema_->fields()) {
      columns.push_back(cudf::make_empty_column(to_cudf_type(field.type)));
    }
    return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
  }

  auto [keys_table, results] = streaming_->finalize(context.stream, context.memory_resource);
  std::vector<std::unique_ptr<cudf::column>> final_columns = keys_table->release();
  final_columns.reserve(final_columns.size() + results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    std::unique_ptr<cudf::column> column = std::move(results[i].results.front());
    if (result_is_count_[i]) {
      column = cudf::cast(column->view(), cudf::data_type{cudf::type_id::INT64}, context.stream,
                          context.memory_resource);
    }
    final_columns.push_back(std::move(column));
  }
  return DeviceBatch(std::make_unique<cudf::table>(std::move(final_columns)), output_schema_);
}

void HashAggregateOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
