#include "kernellake/execution/parquet_scan_operator.hpp"

#include "kernellake/common/errors.hpp"

namespace kernellake {

ParquetScanOperator::ParquetScanOperator(OperatorId id, std::vector<PhysicalFileFragment> fragments,
                                          std::vector<std::string> columns,
                                          std::shared_ptr<const Schema> schema,
                                          std::size_t pass_read_limit_bytes)
    : id_(id),
      fragments_(std::move(fragments)),
      columns_(std::move(columns)),
      schema_(std::move(schema)),
      pass_read_limit_bytes_(pass_read_limit_bytes) {}

void ParquetScanOperator::open(ExecutionContext& context) {
  if (fragments_.empty()) return;  // Every file was pruned away entirely; next() reports empty.

  std::vector<std::string> file_paths;
  std::vector<std::vector<cudf::size_type>> row_groups;
  file_paths.reserve(fragments_.size());
  row_groups.reserve(fragments_.size());
  for (const PhysicalFileFragment& fragment : fragments_) {
    file_paths.push_back(fragment.file.value());
    row_groups.emplace_back(fragment.selected_row_groups.begin(), fragment.selected_row_groups.end());
  }

  cudf::io::parquet_reader_options options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(file_paths))
          .column_names(columns_)
          .row_groups(row_groups)
          .build();

  try {
    reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
        /*chunk_read_limit=*/0, pass_read_limit_bytes_, options, context.stream, context.memory_resource);
  } catch (const std::exception& e) {
    throw StorageError("failed to open Parquet source for scanning: " + std::string(e.what()));
  }
}

std::optional<DeviceBatch> ParquetScanOperator::next(ExecutionContext&) {
  if (!reader_) return std::nullopt;

  while (reader_->has_next()) {
    cudf::io::table_with_metadata result = reader_->read_chunk();
    if (result.tbl->num_rows() > 0) {
      return DeviceBatch(std::move(result.tbl), schema_);
    }
    // An empty chunk (e.g. an empty source file) is valid but uninteresting
    // to downstream operators; keep pulling until a non-empty chunk or
    // genuine exhaustion.
  }
  return std::nullopt;
}

void ParquetScanOperator::close(ExecutionContext&) { reader_.reset(); }

}  // namespace kernellake
