#include "kernellake/execution/hash_join_operator.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include <utility>
#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

// Wraps a device_uvector<size_type> gather map (as returned by
// cudf::hash_join::inner_join()) in a non-owning column_view, the shape
// cudf::gather() expects -- there is no cudf-provided conversion helper for
// this, just the raw pointer/size the vendored header itself documents.
cudf::column_view as_gather_map(const rmm::device_uvector<cudf::size_type>& indices) {
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(indices.size()), indices.data(), nullptr, 0);
}

std::unique_ptr<cudf::table> concatenate_batches(std::vector<DeviceBatch> batches,
                                                 ExecutionContext& context) {
  std::vector<cudf::table_view> views;
  views.reserve(batches.size());
  for (const DeviceBatch& batch : batches) views.push_back(batch.view());
  return cudf::concatenate(views, context.stream, context.memory_resource);
}

}  // namespace

HashJoinOperator::HashJoinOperator(OperatorId id, std::unique_ptr<PhysicalOperator> left,
                                   std::unique_ptr<PhysicalOperator> right, std::size_t left_key_index,
                                   std::size_t right_key_index, std::shared_ptr<const Schema> output_schema)
    : id_(id),
      left_(std::move(left)),
      right_(std::move(right)),
      left_key_index_(static_cast<cudf::size_type>(left_key_index)),
      right_key_index_(static_cast<cudf::size_type>(right_key_index)),
      output_schema_(std::move(output_schema)) {}

void HashJoinOperator::open(ExecutionContext& context) {
  left_->open(context);
  right_->open(context);

  std::vector<DeviceBatch> right_batches;
  while (std::optional<DeviceBatch> batch = right_->next(context)) {
    right_batches.push_back(std::move(*batch));
  }
  if (right_batches.empty()) {
    right_is_empty_ = true;
    return;
  }
  right_table_ = concatenate_batches(std::move(right_batches), context);
  if (right_table_->num_rows() == 0) {
    right_is_empty_ = true;
    return;
  }

  const cudf::table_view right_key_view({right_table_->view().column(right_key_index_)});
  hash_join_ = std::make_unique<cudf::hash_join>(right_key_view, cudf::null_equality::EQUAL, context.stream);
}

std::optional<DeviceBatch> HashJoinOperator::next(ExecutionContext& context) {
  if (right_is_empty_) {
    // An INNER JOIN against an empty build side can never produce a row;
    // drain the probe side so its resources are released the same way a
    // fully-consumed operator's would be, then report exhausted -- matching
    // the "produces no batches at all" convention every other operator uses
    // for an empty result (not a single empty-schema batch, which is only
    // what scalar-aggregate-style operators need).
    while (left_->next(context)) {
    }
    return std::nullopt;
  }

  // Mirrors FilterOperator's next(): a probe batch with no matching rows is
  // skipped rather than returned as an empty batch, so a downstream
  // operator never has to special-case a zero-row batch that isn't the
  // final one.
  while (std::optional<DeviceBatch> left_batch = left_->next(context)) {
    const cudf::table_view left_view = left_batch->view();
    const cudf::table_view left_key_view({left_view.column(left_key_index_)});
    auto [left_indices, right_indices] = hash_join_->inner_join(left_key_view, std::nullopt, context.stream);
    if (left_indices->is_empty()) continue;

    const cudf::column_view left_map = as_gather_map(*left_indices);
    const cudf::column_view right_map = as_gather_map(*right_indices);
    std::unique_ptr<cudf::table> gathered_left = cudf::gather(
        left_view, left_map, cudf::out_of_bounds_policy::DONT_CHECK, context.stream, context.memory_resource);
    std::unique_ptr<cudf::table> gathered_right =
        cudf::gather(right_table_->view(), right_map, cudf::out_of_bounds_policy::DONT_CHECK, context.stream,
                     context.memory_resource);

    std::vector<std::unique_ptr<cudf::column>> columns = gathered_left->release();
    std::vector<std::unique_ptr<cudf::column>> right_columns = gathered_right->release();
    columns.insert(columns.end(), std::make_move_iterator(right_columns.begin()),
                   std::make_move_iterator(right_columns.end()));
    return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
  }
  return std::nullopt;
}

void HashJoinOperator::close(ExecutionContext& context) {
  left_->close(context);
  right_->close(context);
}

}  // namespace kernellake
