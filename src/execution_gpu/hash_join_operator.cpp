#include "kernellake/execution_gpu/hash_join_operator.hpp"

#include <arrow/io/file.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <system_error>
#include <utility>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/arrow_bridge.hpp"
#include "kernellake/execution_gpu/cudf_adapter.hpp"

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

std::unique_ptr<cudf::table> concatenate_batches(const std::vector<DeviceBatch>& batches,
                                                 ExecutionContext& context) {
  std::vector<cudf::table_view> views;
  views.reserve(batches.size());
  for (const DeviceBatch& batch : batches) {
    views.push_back(batch.view());
  }
  return cudf::concatenate(views, context.stream, context.memory_resource);
}

std::shared_ptr<arrow::ipc::RecordBatchFileReader> open_partition_reader(const std::string& path) {
  arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> file_result = arrow::io::ReadableFile::Open(path);
  if (!file_result.ok()) {
    throw ExecutionError(
        fmt::format("failed to reopen hash-join spill file '{}': {}", path, file_result.status().ToString()));
  }
  arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchFileReader>> reader_result =
      arrow::ipc::RecordBatchFileReader::Open(*file_result);
  if (!reader_result.ok()) {
    throw ExecutionError(
        fmt::format("failed to read hash-join spill file '{}': {}", path, reader_result.status().ToString()));
  }
  return *reader_result;
}

std::vector<DeviceBatch> read_partition_batches(const std::string& path,
                                                const std::shared_ptr<const Schema>& schema,
                                                ExecutionContext& context) {
  const std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader = open_partition_reader(path);
  std::vector<DeviceBatch> batches;
  batches.reserve(static_cast<std::size_t>(reader->num_record_batches()));
  for (int i = 0; i < reader->num_record_batches(); ++i) {
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> batch_result = reader->ReadRecordBatch(i);
    if (!batch_result.ok()) {
      throw ExecutionError(fmt::format("failed to read hash-join spill batch from '{}': {}", path,
                                       batch_result.status().ToString()));
    }
    batches.push_back(
        from_arrow_record_batch(**batch_result, schema, context.stream, context.memory_resource));
  }
  return batches;
}

// Streams `child` to exhaustion, hash-partitioning each incoming batch by
// its `key_index` column into `partition_count` buckets (cudf::hash_partition,
// deterministic MURMUR3 with a fixed seed -- see hash_join_operator.hpp's
// class-level comment for why calling this identically on both sides of a
// join guarantees equal keys land in the same bucket index), and spills
// each bucket's slice straight to a *disk* file under `scratch_dir`
// (`{side_prefix}-{i}.arrow`, Arrow IPC file format) instead of
// accumulating it on GPU *or* in host RAM -- device memory stays bounded
// to ~one incoming batch (plus its own partitioned reorder) throughout,
// and host memory stays bounded to whatever's needed to stream one batch
// through, regardless of `child`'s total output size. Each bucket's own
// arrow::ipc::RecordBatchWriter is opened lazily, on that bucket's first
// row (needs a real arrow::RecordBatch to read the schema off of), and
// closed once `child` is exhausted.
//
// Each reordered/sliced device table's destruction (a stream-ordered free
// on context.stream, via context.memory_resource) is safe once
// to_arrow_host()'s device->host copy below (inside to_arrow_record_batch(),
// called with this same context.stream/context.memory_resource) has
// completed -- properly stream-ordered against the rest of this query's
// work, not an implicit dependency on cudf's null-stream default the way
// this used to be documented here.
void spill_partitioned_to_disk(PhysicalOperator& child, cudf::size_type key_index,
                               std::size_t partition_count, const std::filesystem::path& scratch_dir,
                               const std::string& side_prefix, std::vector<std::string>& paths_out,
                               std::vector<bool>& nonempty_out, std::shared_ptr<const Schema>& schema_out,
                               ExecutionContext& context) {
  paths_out.assign(partition_count, std::string());
  nonempty_out.assign(partition_count, false);
  std::vector<std::shared_ptr<arrow::io::FileOutputStream>> sinks(partition_count);
  std::vector<std::shared_ptr<arrow::ipc::RecordBatchWriter>> writers(partition_count);

  auto ensure_writer_open = [&](std::size_t i, const arrow::RecordBatch& sample) {
    if (writers[i] != nullptr) {
      return;
    }
    paths_out[i] = (scratch_dir / fmt::format("{}-{}.arrow", side_prefix, i)).string();
    arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> sink_result =
        arrow::io::FileOutputStream::Open(paths_out[i]);
    if (!sink_result.ok()) {
      throw ExecutionError(fmt::format("failed to open hash-join spill file '{}': {}", paths_out[i],
                                       sink_result.status().ToString()));
    }
    sinks[i] = *sink_result;
    arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchWriter>> writer_result =
        arrow::ipc::MakeFileWriter(sinks[i], sample.schema());
    if (!writer_result.ok()) {
      throw ExecutionError(fmt::format("failed to open hash-join spill writer '{}': {}", paths_out[i],
                                       writer_result.status().ToString()));
    }
    writers[i] = *writer_result;
  };

  while (std::optional<DeviceBatch> batch = child.next(context)) {
    if (!schema_out) {
      schema_out = batch->schema_ptr();
    }
    if (batch->row_count() == 0) {
      continue;
    }
    auto [reordered, offsets] = cudf::hash_partition(
        batch->view(), {key_index}, static_cast<int>(partition_count), cudf::hash_id::HASH_MURMUR3,
        cudf::DEFAULT_HASH_SEED, context.stream, context.memory_resource);
    for (std::size_t i = 0; i < partition_count; ++i) {
      const cudf::size_type begin = offsets[i];
      const cudf::size_type end = offsets[i + 1];
      if (begin == end) {
        continue;
      }
      const std::vector<cudf::table_view> sliced =
          cudf::slice(reordered->view(), {begin, end}, context.stream);
      const std::shared_ptr<arrow::RecordBatch> record_batch =
          to_arrow_record_batch(sliced.front(), *schema_out, context.stream, context.memory_resource);
      ensure_writer_open(i, *record_batch);
      const arrow::Status status = writers[i]->WriteRecordBatch(*record_batch);
      if (!status.ok()) {
        throw ExecutionError(fmt::format("failed to write hash-join spill batch to '{}': {}", paths_out[i],
                                         status.ToString()));
      }
      nonempty_out[i] = true;
    }
  }
  for (std::size_t i = 0; i < partition_count; ++i) {
    if (writers[i] == nullptr) {
      continue;
    }
    const arrow::Status status = writers[i]->Close();
    if (!status.ok()) {
      throw ExecutionError(
          fmt::format("failed to close hash-join spill writer '{}': {}", paths_out[i], status.ToString()));
    }
  }
}

}  // namespace

std::size_t estimate_row_width_bytes(const Schema& schema) {
  std::size_t total = 0;
  for (const Field& field : schema.fields()) {
    switch (field.type.id) {
      case TypeId::Boolean:
        total += 1;
        break;
      case TypeId::Int32:
      case TypeId::UInt32:
      case TypeId::Float32:
      case TypeId::Date32:
        total += 4;
        break;
      case TypeId::Int64:
      case TypeId::UInt64:
      case TypeId::Float64:
      case TypeId::Timestamp:
      case TypeId::Decimal:
        total += 8;
        break;
      case TypeId::String:
        // No data has been read yet at plan time -- this is a flat,
        // documented-rough heuristic (TPC-H's own short categorical/text
        // columns, e.g. o_orderpriority/p_type/p_brand, are in this
        // ballpark), not a real estimate. Erring high is safe here (see
        // this function's own doc comment in the header); erring low
        // risks under-partitioning and a repeat OOM.
        total += 24;
        break;
    }
  }
  return total;
}

std::size_t choose_partition_count(std::optional<std::int64_t> estimated_build_rows,
                                   const Schema& build_side_schema, std::size_t budget_bytes) {
  if (!estimated_build_rows || *estimated_build_rows <= 0 || budget_bytes == 0) {
    return 1;
  }
  const std::size_t row_width = estimate_row_width_bytes(build_side_schema);
  const double estimated_bytes = static_cast<double>(*estimated_build_rows) * static_cast<double>(row_width);
  if (estimated_bytes <= static_cast<double>(budget_bytes)) {
    return 1;
  }

  // 1.5x safety factor on top of the already-conservative row-width
  // heuristic above: erring toward more/smaller partitions is cheap
  // (slightly more spill/reload overhead), erring toward too few risks
  // the exact bad_alloc this whole path exists to avoid. 64 is a sanity
  // ceiling against a wildly wrong estimate producing degenerate,
  // pointlessly tiny buckets.
  constexpr double kSafetyFactor = 1.5;
  constexpr std::size_t kMaxPartitions = 64;
  const double raw = std::ceil(estimated_bytes * kSafetyFactor / static_cast<double>(budget_bytes));
  const std::size_t partitions = raw < 2.0 ? 2 : static_cast<std::size_t>(raw);
  return std::min(partitions, kMaxPartitions);
}

HashJoinOperator::HashJoinOperator(OperatorId id, std::unique_ptr<PhysicalOperator> left,
                                   std::unique_ptr<PhysicalOperator> right, std::size_t left_key_index,
                                   std::size_t right_key_index, std::shared_ptr<const Schema> output_schema,
                                   std::size_t partition_count, std::string spill_directory, JoinType join_type)
    : id_(id),
      left_(std::move(left)),
      right_(std::move(right)),
      left_key_index_(static_cast<cudf::size_type>(left_key_index)),
      right_key_index_(static_cast<cudf::size_type>(right_key_index)),
      output_schema_(std::move(output_schema)),
      join_type_(join_type),
      partition_count_(partition_count == 0 ? 1 : partition_count),
      spill_directory_(std::move(spill_directory)) {}

HashJoinOperator::~HashJoinOperator() {
  remove_scratch_dir();
}

void HashJoinOperator::remove_scratch_dir() noexcept {
  if (scratch_dir_.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::remove_all(scratch_dir_, ec);  // best-effort; a leaked temp dir isn't worth throwing over
  scratch_dir_.clear();
}

void HashJoinOperator::build_hash_join(ExecutionContext& context) {
  if (right_table_ == nullptr || right_table_->num_rows() == 0) {
    right_table_.reset();
    hash_join_.reset();
    right_is_empty_ = true;
    return;
  }
  right_is_empty_ = false;
  const cudf::table_view right_key_view({right_table_->view().column(right_key_index_)});
  hash_join_ = std::make_unique<cudf::hash_join>(right_key_view, cudf::null_equality::EQUAL, context.stream);
}

std::optional<DeviceBatch> HashJoinOperator::probe_one_batch(const DeviceBatch& left_batch,
                                                             ExecutionContext& context) {
  const cudf::table_view left_view = left_batch.view();
  const cudf::table_view left_key_view({left_view.column(left_key_index_)});
  // left_join()'s right_indices carries JoinNoMatch (an out-of-bounds
  // sentinel) for any left row with no match -- NULLIFY turns that gather
  // position into a real NULL instead of undefined behavior/garbage;
  // inner_join() never produces an out-of-bounds index, so DONT_CHECK
  // (skipping the bounds check entirely) is still correct and cheaper
  // there. left_indices is guaranteed non-empty for left_join() whenever
  // left_batch itself has rows -- every left row appears at least once --
  // so the is_empty() check below still means exactly "left_batch was
  // itself empty" for both join types.
  auto [left_indices, right_indices] = join_type_ == JoinType::LeftOuter
                                           ? hash_join_->left_join(left_key_view, std::nullopt, context.stream)
                                           : hash_join_->inner_join(left_key_view, std::nullopt, context.stream);
  if (left_indices->is_empty()) {
    return std::nullopt;
  }

  const cudf::column_view left_map = as_gather_map(*left_indices);
  const cudf::column_view right_map = as_gather_map(*right_indices);
  const cudf::out_of_bounds_policy right_gather_policy =
      join_type_ == JoinType::LeftOuter ? cudf::out_of_bounds_policy::NULLIFY
                                        : cudf::out_of_bounds_policy::DONT_CHECK;
  std::unique_ptr<cudf::table> gathered_left = cudf::gather(
      left_view, left_map, cudf::out_of_bounds_policy::DONT_CHECK, context.stream, context.memory_resource);
  std::unique_ptr<cudf::table> gathered_right = cudf::gather(
      right_table_->view(), right_map, right_gather_policy, context.stream, context.memory_resource);

  std::vector<std::unique_ptr<cudf::column>> columns = gathered_left->release();
  std::vector<std::unique_ptr<cudf::column>> right_columns = gathered_right->release();
  columns.insert(columns.end(), std::make_move_iterator(right_columns.begin()),
                 std::make_move_iterator(right_columns.end()));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
}

std::optional<DeviceBatch> HashJoinOperator::null_extend_batch(const DeviceBatch& left_batch,
                                                                ExecutionContext& context) {
  const cudf::table_view left_view = left_batch.view();
  if (left_view.num_rows() == 0) {
    return std::nullopt;
  }
  std::unique_ptr<cudf::table> owned_left =
      std::make_unique<cudf::table>(left_view, context.stream, context.memory_resource);
  std::vector<std::unique_ptr<cudf::column>> columns = owned_left->release();

  const std::size_t left_field_count = columns.size();
  const std::size_t right_field_count = output_schema_->field_count() - left_field_count;
  columns.reserve(output_schema_->field_count());
  for (std::size_t i = 0; i < right_field_count; ++i) {
    const DataType& type = output_schema_->field(left_field_count + i).type;
    const std::unique_ptr<cudf::scalar> null_scalar =
        cudf::make_default_constructed_scalar(to_cudf_type(type), context.stream, context.memory_resource);
    columns.push_back(cudf::make_column_from_scalar(*null_scalar, left_view.num_rows(), context.stream,
                                                     context.memory_resource));
  }
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), output_schema_);
}

void HashJoinOperator::ensure_partition_built(ExecutionContext& context) {
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
      read_partition_batches(build_partition_paths_[current_partition_], right_schema_, context);
  right_table_ = concatenate_batches(batches, context);
  build_hash_join(context);
  current_partition_built_ = true;
}

void HashJoinOperator::open(ExecutionContext& context) {
  left_->open(context);
  right_->open(context);

  if (partition_count_ <= 1) {
    std::vector<DeviceBatch> right_batches;
    while (std::optional<DeviceBatch> batch = right_->next(context)) {
      right_batches.push_back(std::move(*batch));
    }
    right_table_ = right_batches.empty() ? nullptr : concatenate_batches(right_batches, context);
    build_hash_join(context);
    return;
  }

  // Partitioned path (see this class's own doc comment): fully drain and
  // hash-partition both sides into partition_count_ per-bucket files under
  // a fresh scratch directory here; actually reloading a bucket to GPU and
  // probing it happens lazily, bucket by bucket, in
  // next()/ensure_partition_built().
  const std::filesystem::path base = spill_directory_.empty() ? std::filesystem::temp_directory_path()
                                                              : std::filesystem::path(spill_directory_);
  scratch_dir_ = base / fmt::format("kernellake-hashjoin-{}-{}", id_, static_cast<const void*>(this));
  std::filesystem::create_directories(scratch_dir_);

  spill_partitioned_to_disk(*right_, right_key_index_, partition_count_, scratch_dir_, "build",
                            build_partition_paths_, build_partition_nonempty_, right_schema_, context);
  spill_partitioned_to_disk(*left_, left_key_index_, partition_count_, scratch_dir_, "probe",
                            probe_partition_paths_, probe_partition_nonempty_, left_schema_, context);
}

std::optional<DeviceBatch> HashJoinOperator::next(ExecutionContext& context) {
  if (partition_count_ <= 1) {
    if (right_is_empty_) {
      if (join_type_ == JoinType::LeftOuter) {
        // Nothing to probe against, but every left row must still appear,
        // NULL-extended -- unlike INNER JOIN below, an empty build side
        // does not mean an empty result.
        while (std::optional<DeviceBatch> left_batch = left_->next(context)) {
          if (std::optional<DeviceBatch> result = null_extend_batch(*left_batch, context)) {
            return result;
          }
        }
        return std::nullopt;
      }
      // An INNER JOIN against an empty build side can never produce a row;
      // drain the probe side so its resources are released the same way a
      // fully-consumed operator's would be, then report exhausted --
      // matching the "produces no batches at all" convention every other
      // operator uses for an empty result (not a single empty-schema
      // batch, which is only what scalar-aggregate-style operators need).
      while (left_->next(context)) {
      }
      return std::nullopt;
    }
    // Mirrors FilterOperator's next(): a probe batch with no matching rows
    // is skipped rather than returned as an empty batch, so a downstream
    // operator never has to special-case a zero-row batch that isn't the
    // final one.
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
  // probe_reader_/probe_batch_index_. A bucket with no rows on *either*
  // side can never produce a match -- skipped without ever reloading it to
  // GPU, or even opening its probe file, at all. A bucket with probe rows
  // but no build rows is a real result for LEFT OUTER JOIN, unlike INNER
  // (see next()'s own non-partitioned-path comment above) -- still
  // processed there (skipping only the now-pointless hash_join_ build),
  // null-extending each probe batch instead of probing it.
  while (current_partition_ < partition_count_) {
    const bool has_left_outer_orphan_probe_rows =
        join_type_ == JoinType::LeftOuter && !build_partition_nonempty_[current_partition_];
    if (probe_partition_nonempty_[current_partition_] &&
        (build_partition_nonempty_[current_partition_] || has_left_outer_orphan_probe_rows)) {
      ensure_partition_built(context);
      if (!right_is_empty_ || join_type_ == JoinType::LeftOuter) {
        if (probe_reader_ == nullptr) {
          probe_reader_ = open_partition_reader(probe_partition_paths_[current_partition_]);
          probe_batch_index_ = 0;
        }
        while (probe_batch_index_ < static_cast<std::size_t>(probe_reader_->num_record_batches())) {
          arrow::Result<std::shared_ptr<arrow::RecordBatch>> batch_result =
              probe_reader_->ReadRecordBatch(static_cast<int>(probe_batch_index_));
          if (!batch_result.ok()) {
            throw ExecutionError(fmt::format("failed to read hash-join probe spill batch from '{}': {}",
                                             probe_partition_paths_[current_partition_],
                                             batch_result.status().ToString()));
          }
          ++probe_batch_index_;
          DeviceBatch left_batch =
              from_arrow_record_batch(**batch_result, left_schema_, context.stream, context.memory_resource);
          std::optional<DeviceBatch> result =
              right_is_empty_ ? null_extend_batch(left_batch, context) : probe_one_batch(left_batch, context);
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

void HashJoinOperator::close(ExecutionContext& context) {
  left_->close(context);
  right_->close(context);
  probe_reader_.reset();
  remove_scratch_dir();
}

}  // namespace kernellake
