#include "kernellake/execution_gpu/hash_aggregate_operator.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/null_mask.hpp>
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

std::shared_ptr<const Schema> build_output_schema(const std::vector<NamedExpression>& group_by,
                                                  const std::vector<NamedExpression>& aggregates) {
  std::vector<Field> fields;
  fields.reserve(group_by.size() + aggregates.size());
  for (const NamedExpression& item : group_by) fields.push_back(Field{item.name, item.expr->result_type()});
  for (const NamedExpression& item : aggregates) fields.push_back(Field{item.name, item.expr->result_type()});
  return std::make_shared<const Schema>(Schema(std::move(fields)));
}

// A lightweight sub-table view over `source`'s columns [begin, end) --
// table_view owns its own vector of (non-owning) column_views internally,
// so this is safe to return by value.
cudf::table_view column_range(const cudf::table_view& source, cudf::size_type begin, cudf::size_type end) {
  std::vector<cudf::column_view> columns;
  columns.reserve(static_cast<std::size_t>(end - begin));
  for (cudf::size_type i = begin; i < end; ++i) columns.push_back(source.column(i));
  return cudf::table_view(columns);
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

// SUM/MIN/MAX only -- Count/CountStar/Avg are built from a SUM over a
// synthesized INT64 value column instead (see ValueColumnKind and open()),
// never from cudf's own COUNT/MEAN aggregations, both of which accumulate
// through a 32-bit cudf::size_type internally and silently wrap around once
// a single group's row count exceeds INT32_MAX -- confirmed by a real
// SF1000 TPC-H Q1 run (see docs/ROADMAP.md).
std::unique_ptr<cudf::groupby_aggregation> HashAggregateOperator::make_physical_aggregation(
    PhysicalAggKind kind) {
  switch (kind) {
    case PhysicalAggKind::Sum:
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case PhysicalAggKind::Min:
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case PhysicalAggKind::Max:
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
  }
  throw ExecutionError("unreachable: unknown PhysicalAggKind");
}

HashAggregateOperator::CompiledExpr HashAggregateOperator::compile_expr(const Expression& expr,
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

std::unique_ptr<cudf::column> HashAggregateOperator::materialize(const CompiledExpr& compiled,
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

// Mirrors FilterOperator::evaluate_like()'s exact algorithm -- see that
// function's own comment for why cudf::strings::like() rather than
// cudf::ast (which has no LIKE-equivalent operator at all).
std::unique_ptr<cudf::column> HashAggregateOperator::materialize_like(const CompiledLike& like_expr,
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

// Mirrors ProjectionOperator::materialize_extract's exact algorithm -- see
// that function's own comment for why the INT16->INT64 cast is needed.
std::unique_ptr<cudf::column> HashAggregateOperator::materialize_extract(const CompiledExtract& extract_expr,
                                                                         const DeviceBatch& batch,
                                                                         ExecutionContext& context) {
  const std::unique_ptr<cudf::column> operand = materialize(extract_expr.operand, batch, context);
  std::unique_ptr<cudf::column> extracted = cudf::datetime::extract_datetime_component(
      operand->view(), to_cudf_datetime_component(extract_expr.part), context.stream,
      context.memory_resource);
  return cudf::cast(extracted->view(), cudf::data_type{cudf::type_id::INT64}, context.stream,
                    context.memory_resource);
}

std::unique_ptr<cudf::column> HashAggregateOperator::materialize_value_column(ValueColumnKind kind,
                                                                              const CompiledExpr& compiled,
                                                                              const DeviceBatch& batch,
                                                                              ExecutionContext& context) {
  if (kind == ValueColumnKind::Expression) return materialize(compiled, batch, context);

  const std::unique_ptr<cudf::scalar> one =
      cudf::make_fixed_width_scalar<std::int64_t>(1, context.stream, context.memory_resource);
  std::unique_ptr<cudf::column> ones = cudf::make_column_from_scalar(
      *one, static_cast<cudf::size_type>(batch.row_count()), context.stream, context.memory_resource);
  if (kind == ValueColumnKind::CountStarOnes) return ones;

  // CountColumnOnes: COUNT(argument) excludes nulls, so the ones column
  // needs argument's own null mask -- summing it then gives exactly
  // argument's non-null row count per group.
  const std::unique_ptr<cudf::column> argument = materialize(compiled, batch, context);
  if (argument->nullable()) {
    rmm::device_buffer mask = cudf::copy_bitmask(argument->view(), context.stream, context.memory_resource);
    const cudf::size_type null_count = argument->null_count();
    ones->set_null_mask(std::move(mask), null_count);
  }
  return ones;
}

void HashAggregateOperator::open(ExecutionContext& context) {
  child_->open(context);

  compiled_group_by_.reserve(group_by_.size());
  for (const NamedExpression& item : group_by_) {
    compiled_group_by_.push_back(compile_expr(*item.expr, context));
  }

  compiled_aggregate_args_.reserve(aggregates_.size());
  value_column_kind_.reserve(aggregates_.size());
  physical_agg_kind_.reserve(aggregates_.size());
  aggregate_output_kind_.reserve(aggregates_.size());

  for (const NamedExpression& item : aggregates_) {
    const auto* aggregate = dynamic_cast<const AggregateExpression*>(item.expr.get());
    if (aggregate == nullptr) {
      throw ExecutionError(
          fmt::format("HashAggregateOperator item '{}' is not an AggregateExpression", item.name));
    }

    switch (aggregate->function()) {
      case AggregateFunction::Sum:
      case AggregateFunction::Min:
      case AggregateFunction::Max:
        compiled_aggregate_args_.push_back(compile_expr(*aggregate->argument(), context));
        value_column_kind_.push_back(ValueColumnKind::Expression);
        physical_agg_kind_.push_back(aggregate->function() == AggregateFunction::Sum ? PhysicalAggKind::Sum
                                     : aggregate->function() == AggregateFunction::Min
                                         ? PhysicalAggKind::Min
                                         : PhysicalAggKind::Max);
        aggregate_output_kind_.push_back(AggregateOutputKind::Direct);
        break;

      case AggregateFunction::CountStar:
        // No natural argument column -- materialize_value_column synthesizes
        // an all-valid INT64 ones column straight from the batch's row
        // count, no compiled expression needed.
        compiled_aggregate_args_.push_back(CompiledExpr{});
        value_column_kind_.push_back(ValueColumnKind::CountStarOnes);
        physical_agg_kind_.push_back(PhysicalAggKind::Sum);
        aggregate_output_kind_.push_back(AggregateOutputKind::Direct);
        break;

      case AggregateFunction::Count:
        compiled_aggregate_args_.push_back(compile_expr(*aggregate->argument(), context));
        value_column_kind_.push_back(ValueColumnKind::CountColumnOnes);
        physical_agg_kind_.push_back(PhysicalAggKind::Sum);
        aggregate_output_kind_.push_back(AggregateOutputKind::Direct);
        break;

      case AggregateFunction::Avg: {
        // See AggregateOutputKind::Average's comment in the header for why
        // this is decomposed into our own SUM(argument)/COUNT(argument)
        // pair (both accumulated via the SUM-of-ones trick, in genuine
        // INT64/argument-native precision) instead of requesting cudf's
        // native MEAN aggregation.
        const CompiledExpr compiled_argument = compile_expr(*aggregate->argument(), context);

        compiled_aggregate_args_.push_back(compiled_argument);
        value_column_kind_.push_back(ValueColumnKind::Expression);
        physical_agg_kind_.push_back(PhysicalAggKind::Sum);

        compiled_aggregate_args_.push_back(compiled_argument);
        value_column_kind_.push_back(ValueColumnKind::CountColumnOnes);
        physical_agg_kind_.push_back(PhysicalAggKind::Sum);

        aggregate_output_kind_.push_back(AggregateOutputKind::Average);
        break;
      }
    }
  }
}

std::unique_ptr<cudf::table> HashAggregateOperator::build_combined_columns(const DeviceBatch& batch,
                                                                           ExecutionContext& context) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(compiled_group_by_.size() + compiled_aggregate_args_.size());
  for (const CompiledExpr& compiled : compiled_group_by_)
    columns.push_back(materialize(compiled, batch, context));
  for (std::size_t i = 0; i < compiled_aggregate_args_.size(); ++i) {
    columns.push_back(
        materialize_value_column(value_column_kind_[i], compiled_aggregate_args_[i], batch, context));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> HashAggregateOperator::run_groupby_and_assemble(
    const cudf::table_view& key_view, const cudf::table_view& value_view, ExecutionContext& context) {
  cudf::groupby::groupby grouper(key_view);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.reserve(static_cast<std::size_t>(value_view.num_columns()));
  for (cudf::size_type i = 0; i < value_view.num_columns(); ++i) {
    cudf::groupby::aggregation_request request;
    request.values = value_view.column(i);
    request.aggregations.push_back(
        make_physical_aggregation(physical_agg_kind_[static_cast<std::size_t>(i)]));
    requests.push_back(std::move(request));
  }
  auto [keys_table, results] = grouper.aggregate(requests, context.stream, context.memory_resource);
  std::vector<std::unique_ptr<cudf::column>> columns = keys_table->release();
  columns.reserve(columns.size() + results.size());
  for (cudf::groupby::aggregation_result& result : results) {
    columns.push_back(std::move(result.results.front()));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

void HashAggregateOperator::process_batch(const DeviceBatch& batch, ExecutionContext& context) {
  any_batch_seen_ = true;
  const std::unique_ptr<cudf::table> combined = build_combined_columns(batch, context);
  const cudf::table_view combined_view = combined->view();
  const auto group_by_count = static_cast<cudf::size_type>(group_by_.size());

  // Plain, one-shot groupby over this whole batch -- cost scales with the
  // batch's actual row count and actual distinct-key count, unlike the
  // cudf::groupby::streaming_groupby design this replaced (see the header's
  // class comment for the real SF1000 profiling that motivated this).
  std::unique_ptr<cudf::table> partial = run_groupby_and_assemble(
      column_range(combined_view, 0, group_by_count),
      column_range(combined_view, group_by_count, combined_view.num_columns()), context);

  pending_rows_ += partial->num_rows();
  pending_partials_.push_back(std::move(partial));

  // Defer folding into accumulated_ until pending_partials_ has accumulated
  // at least as many rows as accumulated_ already has -- see
  // flush_pending()'s own comment for why.
  if (accumulated_ == nullptr || pending_rows_ >= accumulated_->num_rows()) {
    flush_pending(context);
  }
}

// Folding every batch's partial result into accumulated_ immediately (the
// original design) touches accumulated_'s full row count on every single
// batch: for a low-cardinality GROUP BY, accumulated_ stays small forever,
// so this is cheap and was never worth complicating. For a high-cardinality
// GROUP BY, though, accumulated_ grows toward the query's real distinct-key
// count, and re-scanning/re-concatenating all of it on every one of
// (potentially many thousands of) incoming batches makes total merge cost
// scale with batches * distinct_keys rather than with the input size.
//
// Deferring the fold until pending_partials_ has accumulated at least as
// many rows as accumulated_ currently holds is a size-adaptive doubling
// strategy (the same shape as std::vector's amortized-growth rule, just
// applied to merge frequency instead of allocation): once accumulated_'s
// size plateaus near the query's real distinct-key count D, this triggers
// roughly every D/batch_size batches, each flush costing O(D), for a total
// merge cost of O(D) per D rows of input -- i.e. O(total_rows) overall
// instead of O(batches * D). Low-cardinality GROUP BYs still flush almost
// every batch (accumulated_ stays tiny, so the threshold is trivially
// reached), matching the original per-batch behavior for that case.
void HashAggregateOperator::flush_pending(ExecutionContext& context) {
  if (pending_partials_.empty()) {
    return;
  }

  if (accumulated_ == nullptr && pending_partials_.size() == 1) {
    // First flush of a single partial: no accumulated_ to merge against and
    // nothing else pending, so the partial's own groupby result already is
    // the correct accumulated_ -- no redundant re-aggregation needed.
    accumulated_ = std::move(pending_partials_.front());
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(pending_partials_.size() + 1);
    if (accumulated_ != nullptr) {
      views.push_back(accumulated_->view());
    }
    for (const std::unique_ptr<cudf::table>& partial : pending_partials_) {
      views.push_back(partial->view());
    }
    const std::unique_ptr<cudf::table> concatenated =
        cudf::concatenate(views, context.stream, context.memory_resource);
    const cudf::table_view concat_view = concatenated->view();
    const auto group_by_count = static_cast<cudf::size_type>(group_by_.size());
    accumulated_ = run_groupby_and_assemble(
        column_range(concat_view, 0, group_by_count),
        column_range(concat_view, group_by_count, concat_view.num_columns()), context);
  }
  pending_partials_.clear();
  pending_rows_ = 0;

  if (accumulated_->num_rows() > max_distinct_keys_) {
    throw ExecutionError(
        fmt::format("HashAggregateOperator: distinct key count {} exceeds max_distinct_keys ({})",
                    accumulated_->num_rows(), max_distinct_keys_));
  }
}

std::optional<DeviceBatch> HashAggregateOperator::next(ExecutionContext& context) {
  if (produced_) return std::nullopt;

  while (std::optional<DeviceBatch> batch = child_->next(context)) {
    process_batch(*batch, context);
  }
  flush_pending(context);
  produced_ = true;

  if (!any_batch_seen_) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(output_schema_->field_count());
    for (const Field& field : output_schema_->fields()) {
      columns.push_back(cudf::make_empty_column(to_cudf_type(field.type)));
    }
    return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
  }

  std::vector<std::unique_ptr<cudf::column>> accumulated_columns = accumulated_->release();
  std::vector<std::unique_ptr<cudf::column>> final_columns;
  final_columns.reserve(accumulated_columns.size());

  for (std::size_t i = 0; i < group_by_.size(); ++i) {
    final_columns.push_back(std::move(accumulated_columns[i]));
  }

  std::size_t value_index = group_by_.size();
  for (AggregateOutputKind kind : aggregate_output_kind_) {
    if (kind == AggregateOutputKind::Direct) {
      final_columns.push_back(std::move(accumulated_columns[value_index]));
      ++value_index;
      continue;
    }

    // Average: divide our own two SUM-of-ones-derived results ourselves --
    // see the header's AggregateOutputKind comment for why, instead of
    // cudf's native (32-bit-internally-accumulated) MEAN aggregation.
    const std::unique_ptr<cudf::column>& sum_column = accumulated_columns[value_index];
    const std::unique_ptr<cudf::column>& count_column = accumulated_columns[value_index + 1];
    value_index += 2;

    const std::unique_ptr<cudf::column> sum_as_double = cudf::cast(
        sum_column->view(), cudf::data_type{cudf::type_id::FLOAT64}, context.stream, context.memory_resource);
    const std::unique_ptr<cudf::column> count_as_double =
        cudf::cast(count_column->view(), cudf::data_type{cudf::type_id::FLOAT64}, context.stream,
                   context.memory_resource);
    final_columns.push_back(cudf::binary_operation(
        sum_as_double->view(), count_as_double->view(), cudf::binary_operator::DIV,
        cudf::data_type{cudf::type_id::FLOAT64}, context.stream, context.memory_resource));
  }
  return DeviceBatch(std::make_unique<cudf::table>(std::move(final_columns)), output_schema_);
}

void HashAggregateOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
