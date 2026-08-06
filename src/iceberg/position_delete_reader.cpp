#include "kernellake/iceberg/position_delete_reader.hpp"

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <fmt/format.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>
#include <parquet/file_reader.h>

#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake::iceberg {

namespace {

constexpr const char* kFilePathColumn = "file_path";
constexpr const char* kPosColumn = "pos";

}  // namespace

std::unordered_map<std::string, std::int64_t> read_position_delete_counts(ObjectStore& store,
                                                                          const Uri& uri) {
  std::unique_ptr<parquet::ParquetFileReader> raw_reader;
  try {
    raw_reader = parquet::ParquetFileReader::Open(store.open(uri)->as_arrow_file());
  } catch (const parquet::ParquetException& e) {
    throw StorageError(
        fmt::format("iceberg position delete file '{}': failed to open: {}", uri.value(), e.what()));
  }
  // Every row group -- unlike a real data scan, nothing here is prunable:
  // every position this file records must be counted.
  const int row_group_count = raw_reader->metadata()->num_row_groups();
  std::vector<int> all_row_groups(static_cast<std::size_t>(row_group_count));
  for (int i = 0; i < row_group_count; ++i) {
    all_row_groups[static_cast<std::size_t>(i)] = i;
  }

  arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> reader_result =
      parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(raw_reader));
  if (!reader_result.ok()) {
    throw StorageError(fmt::format("iceberg position delete file '{}': failed to open: {}", uri.value(),
                                   reader_result.status().ToString()));
  }
  const std::unique_ptr<parquet::arrow::FileReader> file_reader = std::move(*reader_result);

  std::shared_ptr<arrow::Schema> file_schema;
  const arrow::Status schema_status = file_reader->GetSchema(&file_schema);
  if (!schema_status.ok()) {
    throw StorageError(fmt::format("iceberg position delete file '{}': failed to read schema: {}",
                                   uri.value(), schema_status.ToString()));
  }
  const int file_path_index = file_schema->GetFieldIndex(kFilePathColumn);
  const int pos_index = file_schema->GetFieldIndex(kPosColumn);
  if (file_path_index < 0 || pos_index < 0) {
    throw StorageError(
        fmt::format("iceberg position delete file '{}': missing required column '{}' or '{}' -- not a valid "
                    "position-delete file (an equality-delete file misidentified as one, perhaps?)",
                    uri.value(), kFilePathColumn, kPosColumn));
  }

  arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> batch_reader_result =
      file_reader->GetRecordBatchReader(all_row_groups, {file_path_index, pos_index});
  if (!batch_reader_result.ok()) {
    throw StorageError(fmt::format("iceberg position delete file '{}': failed to read: {}", uri.value(),
                                   batch_reader_result.status().ToString()));
  }
  const std::unique_ptr<arrow::RecordBatchReader> batch_reader = std::move(*batch_reader_result);

  std::unordered_map<std::string, std::int64_t> counts;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    const arrow::Status next_status = batch_reader->ReadNext(&batch);
    if (!next_status.ok()) {
      throw StorageError(fmt::format("iceberg position delete file '{}': failed to read: {}", uri.value(),
                                     next_status.ToString()));
    }
    if (batch == nullptr) {
      break;
    }
    const auto* file_path_array = dynamic_cast<const arrow::StringArray*>(batch->column(0).get());
    const auto* pos_array = dynamic_cast<const arrow::Int64Array*>(batch->column(1).get());
    if (file_path_array == nullptr || pos_array == nullptr) {
      throw StorageError(
          fmt::format("iceberg position delete file '{}': '{}'/'{}' columns have an unexpected Arrow type "
                      "(expected string/int64)",
                      uri.value(), kFilePathColumn, kPosColumn));
    }
    // pos itself is never needed here -- resolve_iceberg_table() only asks
    // "how many distinct positions does this file delete for
    // <referenced path>", to compare against that data file's own
    // record_count (see this function's own header comment); it doesn't
    // need to know *which* positions for whole-file-deletion detection.
    for (std::int64_t i = 0; i < file_path_array->length(); ++i) {
      if (file_path_array->IsNull(i)) {
        throw StorageError(fmt::format(
            "iceberg position delete file '{}': row {} has a null '{}' -- required by the Iceberg spec",
            uri.value(), i, kFilePathColumn));
      }
      counts[file_path_array->GetString(i)] += 1;
    }
  }
  return counts;
}

}  // namespace kernellake::iceberg
