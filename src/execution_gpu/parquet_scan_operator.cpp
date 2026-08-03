#include "kernellake/execution_gpu/parquet_scan_operator.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <fmt/format.h>

#include <algorithm>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/object_store_datasource.hpp"

namespace kernellake {

ParquetScanOperator::ParquetScanOperator(OperatorId id, std::vector<PhysicalFileFragment> fragments,
                                         std::vector<std::string> columns,
                                         std::shared_ptr<const Schema> schema, ObjectStore& store,
                                         std::size_t pass_read_limit_bytes)
    : id_(id),
      fragments_(std::move(fragments)),
      columns_(std::move(columns)),
      schema_(std::move(schema)),
      store_(store),
      pass_read_limit_bytes_(pass_read_limit_bytes) {}

void ParquetScanOperator::open(ExecutionContext& context) {
  if (fragments_.empty()) return;  // Every file was pruned away entirely; next() reports empty.

  std::vector<std::vector<cudf::size_type>> row_groups;
  row_groups.reserve(fragments_.size());
  for (const PhysicalFileFragment& fragment : fragments_) {
    row_groups.emplace_back(fragment.selected_row_groups.begin(), fragment.selected_row_groups.end());
  }

  // All-local is the common case and keeps cudf's own local-path source_info
  // constructor with zero extra indirection. Any fragment with a non-"file"
  // scheme routes *every* fragment through ObjectStoreDatasource instead --
  // cudf's chunked reader takes one uniform source list, and a mixed local/
  // remote scan is rare enough not to deserve its own fast path.
  const bool all_local =
      std::all_of(fragments_.begin(), fragments_.end(),
                  [](const PhysicalFileFragment& fragment) { return fragment.file.scheme() == "file"; });

  try {
    if (all_local) {
      std::vector<std::string> file_paths;
      file_paths.reserve(fragments_.size());
      for (const PhysicalFileFragment& fragment : fragments_) file_paths.push_back(fragment.file.value());

      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder(cudf::io::source_info(file_paths))
              .column_names(columns_)
              .row_groups(row_groups)
              .build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, options, context.stream, context.memory_resource);
    } else {
      std::vector<std::unique_ptr<cudf::io::datasource>> sources;
      sources.reserve(fragments_.size());
      for (const PhysicalFileFragment& fragment : fragments_) {
        sources.push_back(std::make_unique<ObjectStoreDatasource>(store_.open(fragment.file)));
      }

      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder().column_names(columns_).row_groups(row_groups).build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, std::move(sources),
          /*parquet_metadatas=*/std::vector<cudf::io::parquet::FileMetaData>{}, options, context.stream,
          context.memory_resource);
    }
  } catch (const std::exception& e) {
    throw StorageError(fmt::format("failed to open Parquet source for scanning: {}", e.what()));
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

void ParquetScanOperator::close(ExecutionContext&) {
  reader_.reset();
}

}  // namespace kernellake
