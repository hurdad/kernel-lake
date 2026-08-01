#include "kernellake/execution/hash_aggregate_operator.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <numeric>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution/cudf_adapter.hpp"

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

void HashAggregateOperator::open(ExecutionContext& context) {
  child_->open(context);

  compiled_group_by_.reserve(group_by_.size());
  for (const NamedExpression& item : group_by_) compiled_group_by_.push_back(&compiler_.compile(*item.expr));

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.reserve(aggregates_.size());
  compiled_aggregate_args_.reserve(aggregates_.size());
  cudf::size_type next_index = static_cast<cudf::size_type>(group_by_.size());

  for (const NamedExpression& item : aggregates_) {
    const auto* aggregate = dynamic_cast<const AggregateExpression*>(item.expr.get());
    if (aggregate == nullptr) {
      throw ExecutionError("HashAggregateOperator item '" + item.name + "' is not an AggregateExpression");
    }
    // COUNT(*) has no natural argument column; reuse group_by[0]'s compiled
    // expression (materialized at its own dedicated table slot below, via
    // null_policy::INCLUDE so nulls in that column don't suppress the
    // count). Note: referencing an *existing* key column's index directly,
    // instead of materializing a fresh copy at its own index, silently
    // produced all-zero counts in testing -- streaming_groupby apparently
    // doesn't support a value column index that aliases a key index.
    const cudf::ast::expression* compiled_argument = aggregate->function() == AggregateFunction::CountStar
                                                          ? compiled_group_by_.front()
                                                          : &compiler_.compile(*aggregate->argument());
    compiled_aggregate_args_.push_back(compiled_argument);
    result_is_count_.push_back(aggregate->function() == AggregateFunction::Count ||
                                aggregate->function() == AggregateFunction::CountStar);
    requests.push_back(
        cudf::groupby::streaming_aggregation_request{next_index, to_streaming_aggregation(aggregate->function())});
    ++next_index;
  }

  std::vector<cudf::size_type> key_indices(group_by_.size());
  std::iota(key_indices.begin(), key_indices.end(), 0);
  streaming_ =
      std::make_unique<cudf::groupby::streaming_groupby>(key_indices, requests, max_distinct_keys_);
}

std::unique_ptr<cudf::table> HashAggregateOperator::build_combined_columns(const DeviceBatch& batch,
                                                                            ExecutionContext& context) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(compiled_group_by_.size() + compiled_aggregate_args_.size());
  for (const cudf::ast::expression* expr : compiled_group_by_) {
    columns.push_back(cudf::compute_column(batch.view(), *expr, context.stream, context.memory_resource));
  }
  for (const cudf::ast::expression* expr : compiled_aggregate_args_) {
    columns.push_back(cudf::compute_column(batch.view(), *expr, context.stream, context.memory_resource));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

void HashAggregateOperator::process_batch(const DeviceBatch& batch, ExecutionContext& context) {
  any_batch_seen_ = true;
  const std::unique_ptr<cudf::table> combined = build_combined_columns(batch, context);
  streaming_->aggregate(combined->view(), context.stream);
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

void HashAggregateOperator::close(ExecutionContext& context) { child_->close(context); }

}  // namespace kernellake
