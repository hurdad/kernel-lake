#include "kernellake/execution_gpu/limit_operator.hpp"

#include <cudf/copying.hpp>

#include "kernellake/common/errors.hpp"

namespace kernellake {

LimitOperator::LimitOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child, std::int64_t limit)
    : id_(id), child_(std::move(child)), limit_(limit) {
  if (limit_ < 0) throw PlanningError("LIMIT must be non-negative");
}

void LimitOperator::open(ExecutionContext& context) {
  child_->open(context);
  remaining_ = limit_;
}

std::optional<DeviceBatch> LimitOperator::next(ExecutionContext& context) {
  if (remaining_ <= 0) return std::nullopt;

  std::optional<DeviceBatch> batch = child_->next(context);
  if (!batch.has_value()) return std::nullopt;

  const auto batch_rows = static_cast<std::int64_t>(batch->row_count());
  if (batch_rows <= remaining_) {
    remaining_ -= batch_rows;
    return batch;
  }

  const auto keep = static_cast<cudf::size_type>(remaining_);
  std::shared_ptr<const Schema> schema = batch->schema_ptr();
  const std::vector<cudf::table_view> sliced = cudf::slice(batch->view(), {0, keep});
  std::unique_ptr<cudf::table> truncated =
      std::make_unique<cudf::table>(sliced.front(), context.stream, context.memory_resource);
  remaining_ = 0;
  return DeviceBatch(std::move(truncated), std::move(schema));
}

void LimitOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
