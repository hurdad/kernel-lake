#include "kernellake/execution_gpu/parquet_scan_operator.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <fmt/format.h>

#include <algorithm>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cudf_adapter.hpp"
#include "kernellake/execution_gpu/object_store_datasource.hpp"
#include "kernellake/expression/expression.hpp"

namespace kernellake {

ParquetScanOperator::ParquetScanOperator(OperatorId id, std::vector<PhysicalFileFragment> fragments,
                                         std::vector<std::string> columns,
                                         std::shared_ptr<const Schema> schema, ObjectStore& store,
                                         std::size_t pass_read_limit_bytes,
                                         std::vector<PartitionColumn> partition_columns)
    : id_(id),
      fragments_(std::move(fragments)),
      columns_(std::move(columns)),
      schema_(std::move(schema)),
      store_(store),
      pass_read_limit_bytes_(pass_read_limit_bytes),
      partition_columns_(std::move(partition_columns)) {}

void ParquetScanOperator::open(ExecutionContext& context) {
  if (fragments_.empty()) return;  // Every file was pruned away entirely; next() reports empty.

  if (!partition_columns_.empty()) {
    // Per-fragment mode (see class comment): reader_ is opened lazily, one
    // fragment at a time, from next()/open_current_fragment() -- nothing to
    // do here beyond leaving current_fragment_index_ at its initial 0.
    return;
  }

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

void ParquetScanOperator::open_current_fragment(ExecutionContext& context) {
  const PhysicalFileFragment& fragment = fragments_[current_fragment_index_];
  const std::vector<cudf::size_type> row_groups(fragment.selected_row_groups.begin(),
                                                fragment.selected_row_groups.end());

  try {
    if (fragment.file.scheme() == "file") {
      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder(
              cudf::io::source_info(std::vector<std::string>{fragment.file.value()}))
              .column_names(columns_)
              .row_groups(std::vector<std::vector<cudf::size_type>>{row_groups})
              .build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, options, context.stream, context.memory_resource);
    } else {
      std::vector<std::unique_ptr<cudf::io::datasource>> sources;
      sources.push_back(std::make_unique<ObjectStoreDatasource>(store_.open(fragment.file)));

      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder()
              .column_names(columns_)
              .row_groups(std::vector<std::vector<cudf::size_type>>{row_groups})
              .build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, std::move(sources),
          /*parquet_metadatas=*/std::vector<cudf::io::parquet::FileMetaData>{}, options, context.stream,
          context.memory_resource);
    }
  } catch (const std::exception& e) {
    throw StorageError(fmt::format("failed to open Parquet source '{}' for partitioned scanning: {}",
                                   fragment.file.value(), e.what()));
  }
}

std::optional<DeviceBatch> ParquetScanOperator::next(ExecutionContext& context) {
  if (fragments_.empty()) return std::nullopt;

  if (partition_columns_.empty()) {
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

  // Per-fragment mode: pull from the current fragment's own reader,
  // advancing to the next fragment (opening its own fresh reader) whenever
  // the current one is exhausted, until every fragment has been read. Every
  // chunk returned here, by construction, comes entirely from
  // fragments_[current_fragment_index_], so appending that one fragment's
  // partition_values as constant columns is always correct -- see the
  // class's own comment for why a single reader spanning every fragment
  // (this operator's normal fast path) can't offer that same guarantee.
  while (true) {
    if (!reader_) {
      if (current_fragment_index_ >= fragments_.size()) return std::nullopt;
      open_current_fragment(context);
    }
    if (!reader_->has_next()) {
      reader_.reset();
      ++current_fragment_index_;
      continue;
    }

    cudf::io::table_with_metadata result = reader_->read_chunk();
    if (result.tbl->num_rows() == 0) {
      continue;  // empty chunk; keep pulling from the same (still-open) fragment reader.
    }

    const cudf::size_type num_rows = result.tbl->num_rows();
    std::vector<std::unique_ptr<cudf::column>> columns = result.tbl->release();
    const PhysicalFileFragment& fragment = fragments_[current_fragment_index_];
    for (std::size_t i = 0; i < partition_columns_.size(); ++i) {
      const LiteralExpression literal(fragment.partition_values[i], partition_columns_[i].type);
      const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(literal);
      columns.push_back(
          cudf::make_column_from_scalar(*scalar, num_rows, context.stream, context.memory_resource));
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    return DeviceBatch(std::move(table), schema_);
  }
}

void ParquetScanOperator::close(ExecutionContext&) {
  reader_.reset();
}

}  // namespace kernellake
