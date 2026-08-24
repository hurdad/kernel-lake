#include "kernellake/execution_gpu/semi_anti_join_operator.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <fmt/format.h>

#include <utility>
#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

// Wraps a device_uvector<size_type> gather map (as returned by
// cudf::filtered_join::semi_join()/anti_join()) in a non-owning
// column_view, the shape cudf::gather() expects -- mirrors
// hash_join_operator.cpp's identical helper (there is no cudf-provided
// conversion for this, just the raw pointer/size the vendored header
// itself documents).
cudf::column_view as_gather_map(const rmm::device_uvector<cudf::size_type>& indices) {
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(indices.size()), indices.data(), nullptr, 0);
}

}  // namespace

SemiAntiJoinOperator::SemiAntiJoinOperator(OperatorId id, std::unique_ptr<PhysicalOperator> left,
                                           std::unique_ptr<PhysicalOperator> right,
                                           std::size_t left_key_index, std::size_t right_key_index,
                                           std::shared_ptr<const Schema> output_schema, JoinType join_type)
    : id_(id),
      left_(std::move(left)),
      right_(std::move(right)),
      left_key_index_(static_cast<cudf::size_type>(left_key_index)),
      right_key_index_(static_cast<cudf::size_type>(right_key_index)),
      output_schema_(std::move(output_schema)),
      join_type_(join_type) {
  if (join_type_ != JoinType::LeftSemi && join_type_ != JoinType::LeftAnti) {
    throw ExecutionError(fmt::format(
        "internal error: SemiAntiJoinOperator constructed with join_type={} (must be LEFT SEMI or LEFT ANTI)",
        kernellake::to_string(join_type_)));
  }
}

SemiAntiJoinOperator::~SemiAntiJoinOperator() = default;

void SemiAntiJoinOperator::open(ExecutionContext& context) {
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
  std::vector<cudf::table_view> views;
  views.reserve(right_batches.size());
  for (const DeviceBatch& batch : right_batches) {
    views.push_back(batch.view());
  }
  right_table_ = cudf::concatenate(views, context.stream, context.memory_resource);
  if (right_table_->num_rows() == 0) {
    right_is_empty_ = true;
    right_table_.reset();
    return;
  }
  right_is_empty_ = false;
  const cudf::table_view right_key_view({right_table_->view().column(right_key_index_)});
  filtered_join_ =
      std::make_unique<cudf::filtered_join>(right_key_view, cudf::null_equality::EQUAL, context.stream);
}

std::optional<DeviceBatch> SemiAntiJoinOperator::probe_one_batch(const DeviceBatch& left_batch,
                                                                 ExecutionContext& context) {
  const cudf::table_view left_view = left_batch.view();
  const cudf::table_view left_key_view({left_view.column(left_key_index_)});
  const std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices =
      join_type_ == JoinType::LeftSemi
          ? filtered_join_->semi_join(left_key_view, context.stream, context.memory_resource)
          : filtered_join_->anti_join(left_key_view, context.stream, context.memory_resource);
  if (left_indices->is_empty()) {
    return std::nullopt;
  }
  const cudf::column_view left_map = as_gather_map(*left_indices);
  std::unique_ptr<cudf::table> gathered = cudf::gather(
      left_view, left_map, cudf::out_of_bounds_policy::DONT_CHECK, context.stream, context.memory_resource);
  return DeviceBatch(std::move(gathered), output_schema_);
}

std::optional<DeviceBatch> SemiAntiJoinOperator::next(ExecutionContext& context) {
  if (right_is_empty_) {
    if (join_type_ == JoinType::LeftAnti) {
      // Nothing to match against -- every left row trivially "has no
      // match", so each left batch passes through unchanged (its own
      // schema already *is* output_schema_, no gather/rebuild needed
      // beyond re-stamping the schema pointer -- release_table() moves
      // the already-owned cudf::table across with no device-memory copy).
      std::optional<DeviceBatch> left_batch = left_->next(context);
      if (!left_batch) {
        return std::nullopt;
      }
      return DeviceBatch(std::move(*left_batch).release_table(), output_schema_);
    }
    // LeftSemi against an empty build side can never produce a row; drain
    // the probe side so its resources are released the same way a fully-
    // consumed operator's would be, then report exhausted -- matching
    // HashJoinOperator's identical INNER-JOIN-against-empty-build-side
    // convention.
    while (left_->next(context)) {
    }
    return std::nullopt;
  }
  // Mirrors HashJoinOperator's next(): a probe batch with no matching
  // rows is skipped rather than returned as an empty batch, so a
  // downstream operator never has to special-case a zero-row batch that
  // isn't the final one.
  while (std::optional<DeviceBatch> left_batch = left_->next(context)) {
    if (std::optional<DeviceBatch> result = probe_one_batch(*left_batch, context)) {
      return result;
    }
  }
  return std::nullopt;
}

void SemiAntiJoinOperator::close(ExecutionContext& context) {
  left_->close(context);
  right_->close(context);
}

}  // namespace kernellake
