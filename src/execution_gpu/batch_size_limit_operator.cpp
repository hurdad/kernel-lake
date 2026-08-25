#include "kernellake/execution_gpu/batch_size_limit_operator.hpp"

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>

#include <algorithm>
#include <utility>

namespace kernellake {

BatchSizeLimitOperator::BatchSizeLimitOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                                               std::size_t max_rows)
    : id_(id), child_(std::move(child)), max_rows_(max_rows) {}

void BatchSizeLimitOperator::open(ExecutionContext& context) {
  child_->open(context);
}

void BatchSizeLimitOperator::close(ExecutionContext& context) {
  pending_.reset();
  pending_offset_ = 0;
  child_->close(context);
}

std::optional<DeviceBatch> BatchSizeLimitOperator::next(ExecutionContext& context) {
  if (!pending_.has_value()) {
    std::optional<DeviceBatch> batch = child_->next(context);
    if (!batch.has_value()) {
      return std::nullopt;
    }
    if (batch->row_count() <= max_rows_) {
      return batch;  // Common case: already within the cap, no copy.
    }
    pending_ = std::move(batch);
    pending_offset_ = 0;
  }

  const auto offset = static_cast<cudf::size_type>(pending_offset_);
  const std::size_t chunk_rows = std::min(max_rows_, pending_->row_count() - pending_offset_);
  const cudf::table_view slice_view =
      cudf::slice(pending_->view(), {offset, offset + static_cast<cudf::size_type>(chunk_rows)}).front();
  auto sliced_table = std::make_unique<cudf::table>(slice_view, context.stream, context.memory_resource);
  DeviceBatch chunk(std::move(sliced_table), pending_->schema_ptr());

  pending_offset_ += chunk_rows;
  if (pending_offset_ >= pending_->row_count()) {
    pending_.reset();
    pending_offset_ = 0;
  }
  return chunk;
}

}  // namespace kernellake
