#include "kernellake/execution_gpu/scalar_aggregate_operator.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <fmt/format.h>

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

// Mirrors HashAggregateOperator::compile_expr exactly -- see that
// function's own comments for why each fast path exists.
ScalarAggregateOperator::CompiledExpr ScalarAggregateOperator::compile_expr(const Expression& expr,
                                                                            ExecutionContext& context) {
  if (const auto* column_ref = dynamic_cast<const ColumnExpression*>(&expr)) {
    CompiledExpr compiled;
    compiled.source_column_index = static_cast<cudf::size_type>(column_ref->column_index());
    return compiled;
  }
  if (const auto* literal = dynamic_cast<const LiteralExpression*>(&expr)) {
    CompiledExpr compiled;
    compiled.literal_scalar = literal_to_scalar(*literal, context);
    return compiled;
  }
  if (const auto* case_expr = dynamic_cast<const CaseExpression*>(&expr)) {
    auto compiled_case = std::make_shared<CompiledCase>();
    compiled_case->result_type = case_expr->result_type();
    compiled_case->branches.reserve(case_expr->when_then().size());
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      compiled_case->branches.push_back(CompiledCaseBranch{compile_expr(*branch.condition, context),
                                                           compile_expr(*branch.result, context)});
    }
    if (case_expr->else_branch() != nullptr) {
      compiled_case->else_value = compile_expr(*case_expr->else_branch(), context);
    }
    CompiledExpr compiled;
    compiled.case_expr = std::move(compiled_case);
    return compiled;
  }
  if (const auto* cast_expr = dynamic_cast<const CastExpression*>(&expr);
      cast_expr != nullptr && cast_expr->result_type().id == TypeId::Decimal) {
    auto decimal_cast = std::make_shared<CompiledDecimalCast>();
    decimal_cast->operand = compile_expr(*cast_expr->operand(), context);
    decimal_cast->target_type = cast_expr->result_type();
    CompiledExpr compiled;
    compiled.decimal_cast = std::move(decimal_cast);
    return compiled;
  }
  if (const auto* like_expr = dynamic_cast<const LikeExpression*>(&expr)) {
    auto compiled_like = std::make_shared<CompiledLike>();
    compiled_like->value = compile_expr(*like_expr->value(), context);
    compiled_like->pattern = like_expr->pattern();
    compiled_like->negated = like_expr->negated();
    CompiledExpr compiled;
    compiled.like_expr = std::move(compiled_like);
    return compiled;
  }
  if (const auto* extract_expr = dynamic_cast<const ExtractExpression*>(&expr)) {
    auto compiled_extract = std::make_shared<CompiledExtract>();
    compiled_extract->operand = compile_expr(*extract_expr->operand(), context);
    compiled_extract->part = extract_expr->part();
    CompiledExpr compiled;
    compiled.extract_expr = std::move(compiled_extract);
    return compiled;
  }
  CompiledExpr compiled;
  compiled.expr = &compiler_.compile(expr, context);
  return compiled;
}

std::unique_ptr<cudf::column> ScalarAggregateOperator::materialize(const CompiledExpr& compiled,
                                                                   const DeviceBatch& batch,
                                                                   ExecutionContext& context) {
  if (compiled.case_expr != nullptr) return materialize_case(*compiled.case_expr, batch, context);
  if (compiled.like_expr != nullptr) return materialize_like(*compiled.like_expr, batch, context);
  if (compiled.extract_expr != nullptr) return materialize_extract(*compiled.extract_expr, batch, context);
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

// Mirrors HashAggregateOperator::materialize_case exactly -- see that
// function's own comment for the fold-from-the-last-branch-backward
// algorithm.
std::unique_ptr<cudf::column> ScalarAggregateOperator::materialize_case(const CompiledCase& case_expr,
                                                                        const DeviceBatch& batch,
                                                                        ExecutionContext& context) {
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

// Mirrors FilterOperator::evaluate_like()'s exact algorithm -- see
// HashAggregateOperator::materialize_like's identical comment.
std::unique_ptr<cudf::column> ScalarAggregateOperator::materialize_like(const CompiledLike& like_expr,
                                                                        const DeviceBatch& batch,
                                                                        ExecutionContext& context) {
  const std::unique_ptr<cudf::column> value = materialize(like_expr.value, batch, context);
  std::unique_ptr<cudf::column> mask =
      cudf::strings::like(cudf::strings_column_view(value->view()), like_expr.pattern, "", context.stream,
                          context.memory_resource);
  if (!like_expr.negated) return mask;
  return cudf::unary_operation(mask->view(), cudf::unary_operator::NOT, context.stream,
                               context.memory_resource);
}

// Mirrors HashAggregateOperator::materialize_extract exactly -- see that
// function's own comment for the INT16->INT64 cast.
std::unique_ptr<cudf::column> ScalarAggregateOperator::materialize_extract(
    const CompiledExtract& extract_expr, const DeviceBatch& batch, ExecutionContext& context) {
  const std::unique_ptr<cudf::column> operand = materialize(extract_expr.operand, batch, context);
  std::unique_ptr<cudf::column> extracted = cudf::datetime::extract_datetime_component(
      operand->view(), to_cudf_datetime_component(extract_expr.part), context.stream,
      context.memory_resource);
  return cudf::cast(extracted->view(), cudf::data_type{cudf::type_id::INT64}, context.stream,
                    context.memory_resource);
}

void ScalarAggregateOperator::open(ExecutionContext& context) {
  child_->open(context);
  states_.reserve(aggregates_.size());
  for (const NamedExpression& item : aggregates_) {
    const auto* aggregate = dynamic_cast<const AggregateExpression*>(item.expr.get());
    if (aggregate == nullptr) {
      throw ExecutionError(
          fmt::format("ScalarAggregateOperator item '{}' is not an AggregateExpression", item.name));
    }
    Accumulator state;
    state.function = aggregate->function();
    state.result_type = aggregate->result_type();
    if (aggregate->argument() != nullptr) {
      state.compiled_argument = compile_expr(*aggregate->argument(), context);
    }
    states_.push_back(std::move(state));
  }
}

std::unique_ptr<cudf::column> ScalarAggregateOperator::materialize_argument(Accumulator& state,
                                                                            const DeviceBatch& batch,
                                                                            ExecutionContext& context) {
  return materialize(state.compiled_argument, batch, context);
}

// Mirrors HashAggregateOperator::materialize_value_column's
// ValueColumnKind::CountColumnOnes branch exactly -- see that function's
// own comment.
std::unique_ptr<cudf::column> ScalarAggregateOperator::materialize_count_ones(const cudf::column& argument,
                                                                              const DeviceBatch& batch,
                                                                              ExecutionContext& context) {
  const std::unique_ptr<cudf::scalar> one =
      cudf::make_fixed_width_scalar<std::int64_t>(1, context.stream, context.memory_resource);
  std::unique_ptr<cudf::column> ones = cudf::make_column_from_scalar(
      *one, static_cast<cudf::size_type>(batch.row_count()), context.stream, context.memory_resource);
  if (argument.nullable()) {
    rmm::device_buffer mask = cudf::copy_bitmask(argument.view(), context.stream, context.memory_resource);
    const cudf::size_type null_count = argument.null_count();
    ones->set_null_mask(std::move(mask), null_count);
  }
  return ones;
}

void ScalarAggregateOperator::process_batch(Accumulator& state, const DeviceBatch& batch,
                                            ExecutionContext& context) {
  switch (state.function) {
    case AggregateFunction::CountStar:
      state.running_count += static_cast<std::int64_t>(batch.row_count());
      return;
    case AggregateFunction::Count: {
      std::unique_ptr<cudf::column> column = materialize_argument(state, batch, context);
      // Not cudf's own COUNT reduce aggregation -- accumulates the same way
      // Sum/Min/Max below do (a device-resident running scalar via
      // cudf::reduce's init overload), so this never reads a value back to
      // host per batch. See materialize_count_ones()'s own comment.
      std::unique_ptr<cudf::column> ones = materialize_count_ones(*column, batch, context);
      if (ones->size() == 0 || ones->null_count() == ones->size()) return;

      auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      if (!state.running_count_scalar) {
        state.running_count_scalar =
            cudf::reduce(ones->view(), *sum_agg, cudf::data_type{cudf::type_id::INT64}, context.stream,
                         context.memory_resource);
      } else {
        const std::optional<std::reference_wrapper<cudf::scalar const>> init = *state.running_count_scalar;
        state.running_count_scalar =
            cudf::reduce(ones->view(), *sum_agg, cudf::data_type{cudf::type_id::INT64}, init, context.stream,
                         context.memory_resource);
      }
      return;
    }
    case AggregateFunction::Sum:
    case AggregateFunction::Min:
    case AggregateFunction::Max: {
      std::unique_ptr<cudf::column> column = materialize_argument(state, batch, context);
      // A batch with zero valid values (either genuinely empty, or every row's
      // argument is NULL) must leave the running value untouched: cudf::reduce's
      // init-based overload returns an *invalid* scalar whenever the column itself
      // contributes no valid values, regardless of whether `init` was valid --
      // folding such a batch in via init would silently wipe an already-accumulated
      // running SUM/MIN/MAX back to NULL. See
      // SumAcrossBatchesSurvivesAnEntirelyNullBatch/
      // SumAcrossBatchesWhereLaterBatchIsEntirelyNull in the test file, which
      // reproduce this against a real GPU either way round.
      if (column->size() == 0 || column->null_count() == column->size()) return;

      std::unique_ptr<cudf::reduce_aggregation> agg =
          state.function == AggregateFunction::Sum   ? cudf::make_sum_aggregation<cudf::reduce_aggregation>()
          : state.function == AggregateFunction::Min ? cudf::make_min_aggregation<cudf::reduce_aggregation>()
                                                     : cudf::make_max_aggregation<cudf::reduce_aggregation>();
      const cudf::data_type output_type = to_cudf_type(state.result_type);
      if (!state.running_value) {
        state.running_value =
            cudf::reduce(column->view(), *agg, output_type, context.stream, context.memory_resource);
      } else {
        const std::optional<std::reference_wrapper<cudf::scalar const>> init = *state.running_value;
        state.running_value =
            cudf::reduce(column->view(), *agg, output_type, init, context.stream, context.memory_resource);
      }
      return;
    }
    case AggregateFunction::Avg: {
      std::unique_ptr<cudf::column> column = materialize_argument(state, batch, context);

      // Same guard as Sum/Min/Max above: a batch contributing zero valid values
      // must not fold into running_value via cudf::reduce's init overload, or it
      // silently poisons an already-accumulated sum back to NULL. Also skips
      // the denominator fold below -- running_value and running_count_scalar
      // are always set together by the same set of batches, so finalize()
      // only needs to check one of them.
      if (column->size() == 0 || column->null_count() == column->size()) return;

      auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      if (!state.running_value) {
        state.running_value = cudf::reduce(column->view(), *sum_agg, cudf::data_type{cudf::type_id::FLOAT64},
                                           context.stream, context.memory_resource);
      } else {
        const std::optional<std::reference_wrapper<cudf::scalar const>> init = *state.running_value;
        state.running_value = cudf::reduce(column->view(), *sum_agg, cudf::data_type{cudf::type_id::FLOAT64},
                                           init, context.stream, context.memory_resource);
      }

      // Denominator: same device-resident SUM-of-ones accumulation as
      // AggregateFunction::Count above, instead of cudf's native MEAN
      // (which this operator never requests, same rationale as
      // HashAggregateOperator's own AVG decomposition) or a per-batch host
      // read.
      std::unique_ptr<cudf::column> ones = materialize_count_ones(*column, batch, context);
      auto count_sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      if (!state.running_count_scalar) {
        state.running_count_scalar =
            cudf::reduce(ones->view(), *count_sum_agg, cudf::data_type{cudf::type_id::INT64}, context.stream,
                         context.memory_resource);
      } else {
        const std::optional<std::reference_wrapper<cudf::scalar const>> init = *state.running_count_scalar;
        state.running_count_scalar =
            cudf::reduce(ones->view(), *count_sum_agg, cudf::data_type{cudf::type_id::INT64}, init,
                         context.stream, context.memory_resource);
      }
      return;
    }
  }
  throw ExecutionError("unreachable: unknown AggregateFunction");
}

std::unique_ptr<cudf::column> ScalarAggregateOperator::finalize(Accumulator& state,
                                                                ExecutionContext& context) {
  const cudf::data_type output_type = to_cudf_type(state.result_type);

  switch (state.function) {
    case AggregateFunction::CountStar: {
      cudf::numeric_scalar<std::int64_t> scalar(state.running_count, true, context.stream,
                                                context.memory_resource);
      return cudf::make_column_from_scalar(scalar, 1, context.stream, context.memory_resource);
    }
    case AggregateFunction::Count: {
      // COUNT(x) of zero (or all-NULL) input is 0, not NULL -- matching
      // CountStar's own semantics, unlike Sum/Min/Max/Avg below, which are
      // genuinely NULL over zero valid rows. No host read needed either
      // way: make_column_from_scalar takes running_count_scalar by
      // reference and fills the output column with it entirely
      // device-side.
      if (!state.running_count_scalar) {
        cudf::numeric_scalar<std::int64_t> zero(0, true, context.stream, context.memory_resource);
        return cudf::make_column_from_scalar(zero, 1, context.stream, context.memory_resource);
      }
      return cudf::make_column_from_scalar(*state.running_count_scalar, 1, context.stream,
                                           context.memory_resource);
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
      // running_value/running_count_scalar are always set together (see
      // process_batch's own comment) -- checking one suffices.
      if (!state.running_value) {
        std::unique_ptr<cudf::scalar> null_scalar =
            cudf::make_default_constructed_scalar(output_type, context.stream, context.memory_resource);
        return cudf::make_column_from_scalar(*null_scalar, 1, context.stream, context.memory_resource);
      }
      // Divide device-side (both operands wrapped as 1-row columns) rather
      // than reading sum/count back to host and dividing on the CPU --
      // mirrors HashAggregateOperator's own AVG finalization, just over a
      // single row instead of one row per group.
      const std::unique_ptr<cudf::column> sum_column =
          cudf::make_column_from_scalar(*state.running_value, 1, context.stream, context.memory_resource);
      const std::unique_ptr<cudf::column> count_column_int64 = cudf::make_column_from_scalar(
          *state.running_count_scalar, 1, context.stream, context.memory_resource);
      const std::unique_ptr<cudf::column> count_column =
          cudf::cast(count_column_int64->view(), cudf::data_type{cudf::type_id::FLOAT64}, context.stream,
                     context.memory_resource);
      return cudf::binary_operation(sum_column->view(), count_column->view(), cudf::binary_operator::DIV,
                                    cudf::data_type{cudf::type_id::FLOAT64}, context.stream,
                                    context.memory_resource);
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
