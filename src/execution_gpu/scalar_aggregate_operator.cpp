#include "kernellake/execution_gpu/scalar_aggregate_operator.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cudf_adapter.hpp"
#include "kernellake/expression/expression.hpp"

namespace kernellake {

namespace {
std::shared_ptr<const Schema> build_output_schema(const std::vector<NamedExpression>& items) {
  std::vector<Field> fields;
  fields.reserve(items.size());
  for (const NamedExpression& item : items) {
    fields.push_back(Field{item.name, item.expr->result_type()});
  }
  return std::make_shared<const Schema>(Schema(std::move(fields)));
}
}  // namespace

ScalarAggregateOperator::ScalarAggregateOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                                                 std::vector<NamedExpression> aggregates)
    : id_(id),
      child_(std::move(child)),
      aggregates_(std::move(aggregates)),
      output_schema_(build_output_schema(aggregates_)) {}

void ScalarAggregateOperator::open(ExecutionContext& context) {
  child_->open(context);
  states_.reserve(aggregates_.size());
  for (const NamedExpression& item : aggregates_) {
    const auto* aggregate = dynamic_cast<const AggregateExpression*>(item.expr.get());
    if (aggregate == nullptr) {
      throw ExecutionError("ScalarAggregateOperator item '" + item.name + "' is not an AggregateExpression");
    }
    Accumulator state;
    state.function = aggregate->function();
    state.argument = aggregate->argument();
    state.result_type = aggregate->result_type();
    if (state.argument != nullptr) {
      if (const auto* column_ref = dynamic_cast<const ColumnExpression*>(state.argument.get())) {
        state.argument_column_index = static_cast<cudf::size_type>(column_ref->column_index());
      } else {
        state.compiled_argument = &compiler_.compile(*state.argument);
      }
    }
    states_.push_back(std::move(state));
  }
}

std::unique_ptr<cudf::column> ScalarAggregateOperator::materialize_argument(Accumulator& state,
                                                                            const DeviceBatch& batch,
                                                                            ExecutionContext& context) {
  if (state.argument_column_index.has_value()) {
    return std::make_unique<cudf::column>(batch.view().column(*state.argument_column_index), context.stream,
                                          context.memory_resource);
  }
  return cudf::compute_column(batch.view(), *state.compiled_argument, context.stream,
                              context.memory_resource);
}

void ScalarAggregateOperator::process_batch(Accumulator& state, const DeviceBatch& batch,
                                            ExecutionContext& context) {
  switch (state.function) {
    case AggregateFunction::CountStar:
      state.running_count += static_cast<std::int64_t>(batch.row_count());
      return;
    case AggregateFunction::Count: {
      std::unique_ptr<cudf::column> column = materialize_argument(state, batch, context);
      auto agg = cudf::make_count_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE);
      std::unique_ptr<cudf::scalar> count =
          cudf::reduce(column->view(), *agg, cudf::data_type{cudf::type_id::INT64}, context.stream,
                       context.memory_resource);
      state.running_count += static_cast<cudf::numeric_scalar<std::int64_t>&>(*count).value(context.stream);
      return;
    }
    case AggregateFunction::Sum:
    case AggregateFunction::Min:
    case AggregateFunction::Max: {
      std::unique_ptr<cudf::column> column = materialize_argument(state, batch, context);
      std::unique_ptr<cudf::reduce_aggregation> agg =
          state.function == AggregateFunction::Sum   ? cudf::make_sum_aggregation<cudf::reduce_aggregation>()
          : state.function == AggregateFunction::Min ? cudf::make_min_aggregation<cudf::reduce_aggregation>()
                                                     : cudf::make_max_aggregation<cudf::reduce_aggregation>();
      const cudf::data_type output_type = to_cudf_type(state.result_type);
      const std::optional<std::reference_wrapper<cudf::scalar const>> init =
          state.running_value
              ? std::optional<std::reference_wrapper<cudf::scalar const>>(*state.running_value)
              : std::nullopt;
      state.running_value =
          cudf::reduce(column->view(), *agg, output_type, init, context.stream, context.memory_resource);
      return;
    }
    case AggregateFunction::Avg: {
      std::unique_ptr<cudf::column> column = materialize_argument(state, batch, context);
      auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      const std::optional<std::reference_wrapper<cudf::scalar const>> init =
          state.running_value
              ? std::optional<std::reference_wrapper<cudf::scalar const>>(*state.running_value)
              : std::nullopt;
      state.running_value = cudf::reduce(column->view(), *sum_agg, cudf::data_type{cudf::type_id::FLOAT64},
                                         init, context.stream, context.memory_resource);
      auto count_agg = cudf::make_count_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE);
      std::unique_ptr<cudf::scalar> count =
          cudf::reduce(column->view(), *count_agg, cudf::data_type{cudf::type_id::INT64}, context.stream,
                       context.memory_resource);
      state.running_count += static_cast<cudf::numeric_scalar<std::int64_t>&>(*count).value(context.stream);
      return;
    }
  }
  throw ExecutionError("unreachable: unknown AggregateFunction");
}

std::unique_ptr<cudf::column> ScalarAggregateOperator::finalize(Accumulator& state,
                                                                ExecutionContext& context) {
  const cudf::data_type output_type = to_cudf_type(state.result_type);

  switch (state.function) {
    case AggregateFunction::CountStar:
    case AggregateFunction::Count: {
      cudf::numeric_scalar<std::int64_t> scalar(state.running_count, true, context.stream,
                                                context.memory_resource);
      return cudf::make_column_from_scalar(scalar, 1, context.stream, context.memory_resource);
    }
    case AggregateFunction::Sum:
    case AggregateFunction::Min:
    case AggregateFunction::Max: {
      if (!state.running_value) {
        std::unique_ptr<cudf::scalar> null_scalar =
            cudf::make_default_constructed_scalar(output_type, context.stream, context.memory_resource);
        return cudf::make_column_from_scalar(*null_scalar, 1, context.stream, context.memory_resource);
      }
      return cudf::make_column_from_scalar(*state.running_value, 1, context.stream, context.memory_resource);
    }
    case AggregateFunction::Avg: {
      if (!state.running_value || state.running_count == 0) {
        std::unique_ptr<cudf::scalar> null_scalar =
            cudf::make_default_constructed_scalar(output_type, context.stream, context.memory_resource);
        return cudf::make_column_from_scalar(*null_scalar, 1, context.stream, context.memory_resource);
      }
      const double sum_value =
          static_cast<cudf::numeric_scalar<double>&>(*state.running_value).value(context.stream);
      cudf::numeric_scalar<double> avg_scalar(sum_value / static_cast<double>(state.running_count), true,
                                              context.stream, context.memory_resource);
      return cudf::make_column_from_scalar(avg_scalar, 1, context.stream, context.memory_resource);
    }
  }
  throw ExecutionError("unreachable: unknown AggregateFunction");
}

std::optional<DeviceBatch> ScalarAggregateOperator::next(ExecutionContext& context) {
  if (produced_) return std::nullopt;

  while (std::optional<DeviceBatch> batch = child_->next(context)) {
    for (Accumulator& state : states_) process_batch(state, *batch, context);
  }
  produced_ = true;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(states_.size());
  for (Accumulator& state : states_) columns.push_back(finalize(state, context));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
}

void ScalarAggregateOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
