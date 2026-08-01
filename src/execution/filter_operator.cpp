#include "kernellake/execution/filter_operator.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>

namespace kernellake {

FilterOperator::FilterOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                                ExpressionPtr predicate)
    : id_(id), child_(std::move(child)), predicate_(std::move(predicate)) {}

void FilterOperator::open(ExecutionContext& context) {
  child_->open(context);
  compiled_predicate_ = &compiler_.compile(*predicate_);
}

std::optional<DeviceBatch> FilterOperator::next(ExecutionContext& context) {
  while (std::optional<DeviceBatch> batch = child_->next(context)) {
    std::unique_ptr<cudf::column> mask = cudf::compute_column(
        batch->view(), *compiled_predicate_, context.stream, context.memory_resource);
    std::shared_ptr<const Schema> schema = batch->schema_ptr();
    std::unique_ptr<cudf::table> filtered =
        cudf::apply_boolean_mask(batch->view(), mask->view(), context.stream, context.memory_resource);
    if (filtered->num_rows() == 0) continue;
    return DeviceBatch(std::move(filtered), std::move(schema));
  }
  return std::nullopt;
}

void FilterOperator::close(ExecutionContext& context) { child_->close(context); }

}  // namespace kernellake
