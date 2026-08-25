#include "kernellake/execution_gpu/spill_partitioned_join.hpp"

#include <arrow/io/file.h>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table_view.hpp>
#include <fmt/format.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/arrow_bridge.hpp"

namespace kernellake {

std::unique_ptr<cudf::table> concatenate_device_batches(const std::vector<DeviceBatch>& batches,
                                                        ExecutionContext& context) {
  std::vector<cudf::table_view> views;
  views.reserve(batches.size());
  for (const DeviceBatch& batch : batches) {
    views.push_back(batch.view());
  }
  return cudf::concatenate(views, context.stream, context.memory_resource);
}

std::shared_ptr<arrow::ipc::RecordBatchFileReader> open_spill_partition_reader(const std::string& path) {
  arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> file_result = arrow::io::ReadableFile::Open(path);
  if (!file_result.ok()) {
    throw ExecutionError(
        fmt::format("failed to reopen spill file '{}': {}", path, file_result.status().ToString()));
  }
  arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchFileReader>> reader_result =
      arrow::ipc::RecordBatchFileReader::Open(*file_result);
  if (!reader_result.ok()) {
    throw ExecutionError(
        fmt::format("failed to read spill file '{}': {}", path, reader_result.status().ToString()));
  }
  return *reader_result;
}

std::vector<DeviceBatch> read_spill_partition_batches(const std::string& path,
                                                      const std::shared_ptr<const Schema>& schema,
                                                      ExecutionContext& context) {
  const std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader = open_spill_partition_reader(path);
  std::vector<DeviceBatch> batches;
  batches.reserve(static_cast<std::size_t>(reader->num_record_batches()));
  for (int i = 0; i < reader->num_record_batches(); ++i) {
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> batch_result = reader->ReadRecordBatch(i);
    if (!batch_result.ok()) {
      throw ExecutionError(
          fmt::format("failed to read spill batch from '{}': {}", path, batch_result.status().ToString()));
    }
    batches.push_back(
        from_arrow_record_batch(**batch_result, schema, context.stream, context.memory_resource));
  }
  return batches;
}

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
      throw ExecutionError(
          fmt::format("failed to open spill file '{}': {}", paths_out[i], sink_result.status().ToString()));
    }
    sinks[i] = *sink_result;
    arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchWriter>> writer_result =
        arrow::ipc::MakeFileWriter(sinks[i], sample.schema());
    if (!writer_result.ok()) {
      throw ExecutionError(fmt::format("failed to open spill writer '{}': {}", paths_out[i],
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
        throw ExecutionError(
            fmt::format("failed to write spill batch to '{}': {}", paths_out[i], status.ToString()));
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
          fmt::format("failed to close spill writer '{}': {}", paths_out[i], status.ToString()));
    }
  }
}

}  // namespace kernellake
