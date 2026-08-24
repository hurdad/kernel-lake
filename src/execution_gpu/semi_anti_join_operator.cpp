#include "kernellake/execution_gpu/semi_anti_join_operator.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <fmt/format.h>

#include <utility>
#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

// Wraps a device_uvector<size_type> gather map (as returned by
// cudf::hash_join::inner_join()/left_join()) in a non-owning column_view,
// the shape cudf::gather() expects -- mirrors hash_join_operator.cpp's
// identical helper (there is no cudf-provided conversion for this, just
// the raw pointer/size the vendored header itself documents).
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
  hash_join_ = std::make_unique<cudf::hash_join>(right_key_view, cudf::null_equality::EQUAL, context.stream);
}

std::optional<DeviceBatch> SemiAntiJoinOperator::probe_one_batch(const DeviceBatch& left_batch,
                                                                 ExecutionContext& context) {
  const cudf::table_view left_view = left_batch.view();
  const cudf::table_view left_key_view({left_view.column(left_key_index_)});

  if (join_type_ == JoinType::LeftSemi) {
    auto [left_indices, right_indices] = hash_join_->inner_join(left_key_view, std::nullopt, context.stream);
    static_cast<void>(right_indices);  // never gathered -- see this class's own header comment.
    if (left_indices->is_empty()) {
      return std::nullopt;
    }
    // A probe row can match more than one build row (TPC-H Q4: one order
    // can have several lineitem rows), but LEFT SEMI JOIN wants each
    // matching probe row at most once -- dedupe inner_join()'s
    // one-entry-per-match left_indices down to the distinct set.
    const cudf::table_view left_indices_table({as_gather_map(*left_indices)});
    const std::unique_ptr<cudf::table> distinct_indices = cudf::distinct(
        left_indices_table, {0}, cudf::duplicate_keep_option::KEEP_ANY, cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL, context.stream, context.memory_resource);
    std::unique_ptr<cudf::table> gathered =
        cudf::gather(left_view, distinct_indices->view().column(0), cudf::out_of_bounds_policy::DONT_CHECK,
                     context.stream, context.memory_resource);
    return DeviceBatch(std::move(gathered), output_schema_);
  }

  // LeftAnti: left_join() guarantees every probe row appears at least
  // once -- a row with zero matches appears exactly once, paired with
  // the JoinNoMatch sentinel; a row with >=1 matches never carries that
  // sentinel (see cudf::hash_join::left_join()'s own doc comment). So
  // filtering right_indices down to exactly that sentinel value gives
  // the anti-join's left_indices directly, with no dedup needed.
  auto [left_indices, right_indices] = hash_join_->left_join(left_key_view, std::nullopt, context.stream);
  if (left_indices->is_empty()) {
    return std::nullopt;
  }
  const cudf::numeric_scalar<cudf::size_type> no_match_sentinel(cudf::JoinNoMatch, /*is_valid=*/true,
                                                                context.stream, context.memory_resource);
  const std::unique_ptr<cudf::column> is_unmatched =
      cudf::binary_operation(as_gather_map(*right_indices), no_match_sentinel, cudf::binary_operator::EQUAL,
                             cudf::data_type{cudf::type_id::BOOL8}, context.stream, context.memory_resource);
  const cudf::table_view left_indices_table({as_gather_map(*left_indices)});
  const std::unique_ptr<cudf::table> unmatched_indices = cudf::apply_boolean_mask(
      left_indices_table, is_unmatched->view(), context.stream, context.memory_resource);
  if (unmatched_indices->num_rows() == 0) {
    return std::nullopt;
  }
  std::unique_ptr<cudf::table> gathered =
      cudf::gather(left_view, unmatched_indices->view().column(0), cudf::out_of_bounds_policy::DONT_CHECK,
                   context.stream, context.memory_resource);
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
