#include "kernellake/execution_gpu/semi_anti_join_operator.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <fmt/format.h>

#include <system_error>
#include <utility>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/arrow_bridge.hpp"
#include "kernellake/execution_gpu/spill_partitioned_join.hpp"

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
                                           std::shared_ptr<const Schema> output_schema,
                                           std::size_t partition_count, std::string spill_directory,
                                           JoinType join_type, ExpressionPtr residual_predicate)
    : id_(id),
      left_(std::move(left)),
      right_(std::move(right)),
      left_key_index_(static_cast<cudf::size_type>(left_key_index)),
      right_key_index_(static_cast<cudf::size_type>(right_key_index)),
      output_schema_(std::move(output_schema)),
      partition_count_(partition_count == 0 ? 1 : partition_count),
      spill_directory_(std::move(spill_directory)),
      join_type_(join_type),
      residual_predicate_(std::move(residual_predicate)) {
  if (join_type_ != JoinType::LeftSemi && join_type_ != JoinType::LeftAnti) {
    throw ExecutionError(fmt::format(
        "internal error: SemiAntiJoinOperator constructed with join_type={} (must be LEFT SEMI or LEFT ANTI)",
        kernellake::to_string(join_type_)));
  }
}

SemiAntiJoinOperator::~SemiAntiJoinOperator() {
  remove_scratch_dir();
}

void SemiAntiJoinOperator::remove_scratch_dir() noexcept {
  if (scratch_dir_.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::remove_all(scratch_dir_, ec);  // best-effort; a leaked temp dir isn't worth throwing over
  scratch_dir_.clear();
}

void SemiAntiJoinOperator::build_hash_join(ExecutionContext& context) {
  if (right_table_ == nullptr || right_table_->num_rows() == 0) {
    right_table_.reset();
    hash_join_.reset();
    right_is_empty_ = true;
    return;
  }
  right_is_empty_ = false;
  // A residual predicate uses cudf::mixed_left_semi_join()/
  // mixed_left_anti_join() instead (see probe_one_batch()), which take
  // right_table_'s own table_view directly, on every call -- no prebuilt
  // cudf::hash_join object needed or used in this mode.
  if (residual_predicate_ != nullptr) {
    return;
  }
  const cudf::table_view right_key_view({right_table_->view().column(right_key_index_)});
  hash_join_ = std::make_unique<cudf::hash_join>(right_key_view, cudf::null_equality::EQUAL, context.stream);
}

void SemiAntiJoinOperator::ensure_partition_built(ExecutionContext& context) {
  if (current_partition_built_) {
    return;
  }
  if (!build_partition_nonempty_[current_partition_]) {
    right_table_.reset();
    hash_join_.reset();
    right_is_empty_ = true;
    current_partition_built_ = true;
    return;
  }
  // Safe to concatenate this bucket's rows in full: by construction it's
  // ~1/partition_count_ of the whole build side, comfortably under the
  // budget choose_partition_count() sized partition_count_ against.
  const std::vector<DeviceBatch> batches =
      read_spill_partition_batches(build_partition_paths_[current_partition_], right_schema_, context);
  right_table_ = concatenate_device_batches(batches, context);
  build_hash_join(context);
}

void SemiAntiJoinOperator::open(ExecutionContext& context) {
  left_->open(context);
  right_->open(context);

  if (residual_predicate_ != nullptr) {
    // output_schema_ *is* left's own schema unchanged (see this class's
    // own doc comment), so its field count is exactly the threshold
    // residual_predicate_'s own column indices already split on.
    compiled_residual_ =
        &compiler_.compile_two_table(*residual_predicate_, output_schema_->field_count(), context);
  }

  if (partition_count_ <= 1) {
    std::vector<DeviceBatch> right_batches;
    while (std::optional<DeviceBatch> batch = right_->next(context)) {
      right_batches.push_back(std::move(*batch));
    }
    right_table_ = right_batches.empty() ? nullptr : concatenate_device_batches(right_batches, context);
    build_hash_join(context);
    return;
  }

  // Partitioned path (see this class's own doc comment): fully drain and
  // hash-partition both sides into partition_count_ per-bucket files under
  // a fresh scratch directory here; actually reloading a bucket to GPU and
  // probing it happens lazily, bucket by bucket, in
  // next()/ensure_partition_built() -- mirrors HashJoinOperator::open()'s
  // identical partitioned path exactly.
  const std::filesystem::path base = spill_directory_.empty() ? std::filesystem::temp_directory_path()
                                                              : std::filesystem::path(spill_directory_);
  scratch_dir_ = base / fmt::format("kernellake-semianti-{}-{}", id_, static_cast<const void*>(this));
  std::filesystem::create_directories(scratch_dir_);

  spill_partitioned_to_disk(*right_, right_key_index_, partition_count_, scratch_dir_, "build",
                            build_partition_paths_, build_partition_nonempty_, right_schema_, context);
  spill_partitioned_to_disk(*left_, left_key_index_, partition_count_, scratch_dir_, "probe",
                            probe_partition_paths_, probe_partition_nonempty_, left_schema_, context);
}

std::optional<DeviceBatch> SemiAntiJoinOperator::probe_one_batch(const DeviceBatch& left_batch,
                                                                 ExecutionContext& context) {
  const cudf::table_view left_view = left_batch.view();
  const cudf::table_view left_key_view({left_view.column(left_key_index_)});

  if (residual_predicate_ != nullptr) {
    // cudf::mixed_left_semi_join()/mixed_left_anti_join() already return
    // each qualifying left row exactly once (semi) or every left row with
    // zero qualifying matches (anti) -- no dedup/sentinel-filter step
    // needed the way the plain-hash_join path below still requires, since
    // the equality-plus-conditional matching happens in one call. See
    // this class's own header comment.
    const cudf::table_view right_key_view({right_table_->view().column(right_key_index_)});
    const std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices =
        join_type_ == JoinType::LeftSemi
            ? cudf::mixed_left_semi_join(left_key_view, right_key_view, left_view, right_table_->view(),
                                         *compiled_residual_, cudf::null_equality::EQUAL, context.stream,
                                         context.memory_resource)
            : cudf::mixed_left_anti_join(left_key_view, right_key_view, left_view, right_table_->view(),
                                         *compiled_residual_, cudf::null_equality::EQUAL, context.stream,
                                         context.memory_resource);
    if (left_indices->is_empty()) {
      return std::nullopt;
    }
    std::unique_ptr<cudf::table> gathered =
        cudf::gather(left_view, as_gather_map(*left_indices), cudf::out_of_bounds_policy::DONT_CHECK,
                     context.stream, context.memory_resource);
    return DeviceBatch(std::move(gathered), output_schema_);
  }

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
  if (partition_count_ <= 1) {
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

  // Partitioned path: walk buckets in order, streaming each bucket's
  // spilled probe file off disk batch by batch against that bucket's
  // (lazily rebuilt) hash_join_, resuming mid-bucket across calls via
  // probe_reader_/probe_batch_index_ -- mirrors HashJoinOperator::next()'s
  // partitioned path exactly, except a build-empty bucket is a real result
  // for LEFT ANTI (every one of its probe rows passes through unchanged,
  // see this class's own header comment), not just LEFT OUTER's case the
  // way HashJoinOperator's version handles it.
  while (current_partition_ < partition_count_) {
    const bool has_left_anti_orphan_probe_rows =
        join_type_ == JoinType::LeftAnti && !build_partition_nonempty_[current_partition_];
    if (probe_partition_nonempty_[current_partition_] &&
        (build_partition_nonempty_[current_partition_] || has_left_anti_orphan_probe_rows)) {
      ensure_partition_built(context);
      if (!right_is_empty_ || join_type_ == JoinType::LeftAnti) {
        if (probe_reader_ == nullptr) {
          probe_reader_ = open_spill_partition_reader(probe_partition_paths_[current_partition_]);
          probe_batch_index_ = 0;
        }
        while (probe_batch_index_ < static_cast<std::size_t>(probe_reader_->num_record_batches())) {
          arrow::Result<std::shared_ptr<arrow::RecordBatch>> batch_result =
              probe_reader_->ReadRecordBatch(static_cast<int>(probe_batch_index_));
          if (!batch_result.ok()) {
            throw ExecutionError(fmt::format("failed to read semi/anti-join probe spill batch from '{}': {}",
                                             probe_partition_paths_[current_partition_],
                                             batch_result.status().ToString()));
          }
          ++probe_batch_index_;
          DeviceBatch left_batch =
              from_arrow_record_batch(**batch_result, left_schema_, context.stream, context.memory_resource);
          std::optional<DeviceBatch> result =
              right_is_empty_
                  ? std::make_optional(DeviceBatch(std::move(left_batch).release_table(), output_schema_))
                  : probe_one_batch(left_batch, context);
          if (result) {
            return result;
          }
        }
      }
    }
    // Exhausted this bucket -- release its GPU-resident build table and
    // close its probe file before moving on, so at most one bucket's
    // worth of build-side device memory and one open probe file are ever
    // alive at once, then advance to the next bucket.
    right_table_.reset();
    hash_join_.reset();
    probe_reader_.reset();
    ++current_partition_;
    probe_batch_index_ = 0;
    current_partition_built_ = false;
  }
  return std::nullopt;
}

void SemiAntiJoinOperator::close(ExecutionContext& context) {
  left_->close(context);
  right_->close(context);
  probe_reader_.reset();
  remove_scratch_dir();
}

}  // namespace kernellake
