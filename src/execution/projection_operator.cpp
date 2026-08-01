#include "kernellake/execution/projection_operator.hpp"

#include <cudf/transform.hpp>

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

void ProjectionOperator::open(ExecutionContext& context) {
  child_->open(context);
  compiled_items_.reserve(items_.size());
  for (const NamedExpression& item : items_) {
    compiled_items_.push_back(&compiler_.compile(*item.expr));
  }
}

std::optional<DeviceBatch> ProjectionOperator::next(ExecutionContext& context) {
  std::optional<DeviceBatch> batch = child_->next(context);
  if (!batch.has_value()) return std::nullopt;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(compiled_items_.size());
  for (const cudf::ast::expression* expr : compiled_items_) {
    columns.push_back(cudf::compute_column(batch->view(), *expr, context.stream, context.memory_resource));
  }
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
}

void ProjectionOperator::close(ExecutionContext& context) { child_->close(context); }

}  // namespace kernellake
