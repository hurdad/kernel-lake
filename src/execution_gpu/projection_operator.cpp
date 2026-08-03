#include "kernellake/execution_gpu/projection_operator.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

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

ProjectionOperator::ProjectionOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                                       std::vector<NamedExpression> items)
    : id_(id),
      child_(std::move(child)),
      items_(std::move(items)),
      output_schema_(build_output_schema(items_)) {}

ProjectionOperator::CompiledValue ProjectionOperator::compile_value(const Expression& expr) {
  if (const auto* column_ref = dynamic_cast<const ColumnExpression*>(&expr)) {
    return CompiledValue{static_cast<cudf::size_type>(column_ref->column_index()), nullptr, nullptr, nullptr};
  }
  if (const auto* literal = dynamic_cast<const LiteralExpression*>(&expr)) {
    return CompiledValue{std::nullopt, literal_to_scalar(*literal), nullptr, nullptr};
  }
  if (const auto* cast_expr = dynamic_cast<const CastExpression*>(&expr);
      cast_expr != nullptr && cast_expr->result_type().id == TypeId::Decimal) {
    auto decimal_cast = std::make_shared<CompiledDecimalCast>();
    decimal_cast->operand = compile_value(*cast_expr->operand());
    decimal_cast->target_type = cast_expr->result_type();
    CompiledValue value;
    value.decimal_cast = std::move(decimal_cast);
    return value;
  }
  return CompiledValue{std::nullopt, nullptr, &compiler_.compile(expr), nullptr};
}

ProjectionOperator::CompiledItem ProjectionOperator::compile_item(const Expression& expr) {
  const auto* case_expr = dynamic_cast<const CaseExpression*>(&expr);
  if (case_expr == nullptr) return CompiledItem{compile_value(expr), nullptr};

  auto compiled_case = std::make_unique<CompiledCase>();
  compiled_case->result_type = case_expr->result_type();
  compiled_case->branches.reserve(case_expr->when_then().size());
  for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
    compiled_case->branches.push_back(
        CompiledCaseBranch{compile_value(*branch.condition), compile_value(*branch.result)});
  }
  if (case_expr->else_branch() != nullptr) {
    compiled_case->else_value = compile_value(*case_expr->else_branch());
  }
  return CompiledItem{CompiledValue{}, std::move(compiled_case)};
}

void ProjectionOperator::open(ExecutionContext& context) {
  child_->open(context);
  compiled_items_.reserve(items_.size());
  for (const NamedExpression& item : items_) compiled_items_.push_back(compile_item(*item.expr));
}

std::unique_ptr<cudf::column> ProjectionOperator::materialize_value(const CompiledValue& value,
                                                                    const cudf::table_view& batch,
                                                                    ExecutionContext& context) {
  if (value.decimal_cast != nullptr) {
    const std::unique_ptr<cudf::column> operand =
        materialize_value(value.decimal_cast->operand, batch, context);
    return cudf::cast(operand->view(), to_cudf_type(value.decimal_cast->target_type), context.stream,
                      context.memory_resource);
  }
  if (value.source_column_index.has_value()) {
    return std::make_unique<cudf::column>(batch.column(*value.source_column_index), context.stream,
                                          context.memory_resource);
  }
  if (value.literal_scalar != nullptr) {
    return cudf::make_column_from_scalar(*value.literal_scalar, batch.num_rows(), context.stream,
                                         context.memory_resource);
  }
  return cudf::compute_column(batch, *value.expr, context.stream, context.memory_resource);
}

std::unique_ptr<cudf::column> ProjectionOperator::materialize_case(const CompiledCase& case_expr,
                                                                   const cudf::table_view& batch,
                                                                   ExecutionContext& context) {
  // Folds from the last branch backward: result = "if c1 then r1 else (if
  // c2 then r2 else (... else base))", where `base` is either the compiled
  // ELSE value or an all-NULL column when there is none.
  std::unique_ptr<cudf::column> result;
  if (case_expr.else_value.has_value()) {
    result = materialize_value(*case_expr.else_value, batch, context);
  } else {
    const std::unique_ptr<cudf::scalar> null_scalar = cudf::make_default_constructed_scalar(
        to_cudf_type(case_expr.result_type), context.stream, context.memory_resource);
    result = cudf::make_column_from_scalar(*null_scalar, batch.num_rows(), context.stream,
                                           context.memory_resource);
  }
  for (auto it = case_expr.branches.rbegin(); it != case_expr.branches.rend(); ++it) {
    std::unique_ptr<cudf::column> condition = materialize_value(it->condition, batch, context);
    std::unique_ptr<cudf::column> branch_result = materialize_value(it->result, batch, context);
    result = cudf::copy_if_else(branch_result->view(), result->view(), condition->view(), context.stream,
                                context.memory_resource);
  }
  return result;
}

std::optional<DeviceBatch> ProjectionOperator::next(ExecutionContext& context) {
  std::optional<DeviceBatch> batch = child_->next(context);
  if (!batch.has_value()) return std::nullopt;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(compiled_items_.size());
  for (const CompiledItem& item : compiled_items_) {
    columns.push_back(item.case_expr != nullptr ? materialize_case(*item.case_expr, batch->view(), context)
                                                : materialize_value(item.value, batch->view(), context));
  }
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
}

void ProjectionOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
