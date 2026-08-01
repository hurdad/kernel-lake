#include "kernellake/io/parquet_metadata.hpp"

#include <parquet/arrow/schema.h>
#include <parquet/exception.h>
#include <parquet/file_reader.h>
#include <parquet/statistics.h>

#include <algorithm>
#include <optional>

#include "kernellake/common/errors.hpp"
#include "kernellake/types/arrow_adapter.hpp"

namespace kernellake {

namespace {

std::optional<LiteralStorage> decode_statistic(const parquet::Statistics& stats, bool want_min) {
  switch (stats.physical_type()) {
    case parquet::Type::BOOLEAN: {
      const auto& typed = static_cast<const parquet::BoolStatistics&>(stats);
      return LiteralStorage{want_min ? typed.min() : typed.max()};
    }
    case parquet::Type::INT32: {
      const auto& typed = static_cast<const parquet::Int32Statistics&>(stats);
      return LiteralStorage{static_cast<std::int64_t>(want_min ? typed.min() : typed.max())};
    }
    case parquet::Type::INT64: {
      const auto& typed = static_cast<const parquet::Int64Statistics&>(stats);
      return LiteralStorage{static_cast<std::int64_t>(want_min ? typed.min() : typed.max())};
    }
    case parquet::Type::FLOAT: {
      const auto& typed = static_cast<const parquet::FloatStatistics&>(stats);
      return LiteralStorage{static_cast<double>(want_min ? typed.min() : typed.max())};
    }
    case parquet::Type::DOUBLE: {
      const auto& typed = static_cast<const parquet::DoubleStatistics&>(stats);
      return LiteralStorage{want_min ? typed.min() : typed.max()};
    }
    case parquet::Type::BYTE_ARRAY: {
      const auto& typed = static_cast<const parquet::ByteArrayStatistics&>(stats);
      const parquet::ByteArray& value = want_min ? typed.min() : typed.max();
      return LiteralStorage{std::string(reinterpret_cast<const char*>(value.ptr), value.len)};
    }
    default:
      // INT96 and FIXED_LEN_BYTE_ARRAY (decimal) are not yet decoded here;
      // returning nullopt leaves ColumnStatistics::has_min_max false so
      // pruning correctly never uses these.
      return std::nullopt;
  }
}

ColumnStatistics extract_column_statistics(const parquet::ColumnChunkMetaData& column) {
  ColumnStatistics result;
  if (!column.is_stats_set()) return result;

  const std::shared_ptr<parquet::Statistics> stats = column.statistics();
  if (stats == nullptr) return result;

  if (stats->HasNullCount()) {
    result.has_null_count = true;
    result.null_count = stats->null_count();
  }
  if (stats->HasMinMax()) {
    std::optional<LiteralStorage> min_value = decode_statistic(*stats, /*want_min=*/true);
    std::optional<LiteralStorage> max_value = decode_statistic(*stats, /*want_min=*/false);
    if (min_value.has_value() && max_value.has_value()) {
      result.has_min_max = true;
      result.min_value = std::move(*min_value);
      result.max_value = std::move(*max_value);
    }
  }
  return result;
}

}  // namespace

FileMetadata inspect_parquet_file(ObjectStore& store, const Uri& path) {
  std::unique_ptr<RandomAccessObject> object = store.open(path);

  std::unique_ptr<parquet::ParquetFileReader> reader;
  std::shared_ptr<parquet::FileMetaData> file_meta;
  try {
    reader = parquet::ParquetFileReader::Open(object->as_arrow_file());
    file_meta = reader->metadata();
  } catch (const parquet::ParquetException& e) {
    throw StorageError("failed to read Parquet metadata for '" + path.value() + "': " + e.what());
  }

  std::shared_ptr<arrow::Schema> arrow_schema;
  const arrow::Status status = parquet::arrow::FromParquetSchema(file_meta->schema(), &arrow_schema);
  if (!status.ok()) {
    throw StorageError("failed to convert Parquet schema for '" + path.value() + "': " + status.ToString());
  }

  FileMetadata result;
  result.path = path;
  result.schema = from_arrow_schema(arrow_schema);
  result.row_count = file_meta->num_rows();
  result.row_groups.reserve(static_cast<std::size_t>(file_meta->num_row_groups()));

  for (int rg = 0; rg < file_meta->num_row_groups(); ++rg) {
    const std::unique_ptr<parquet::RowGroupMetaData> rg_meta = file_meta->RowGroup(rg);
    RowGroupMetadata row_group;
    row_group.index = rg;
    row_group.row_count = rg_meta->num_rows();
    row_group.compressed_size_bytes = rg_meta->total_compressed_size();
    row_group.uncompressed_size_bytes = rg_meta->total_byte_size();

    for (int col = 0; col < rg_meta->num_columns(); ++col) {
      const std::unique_ptr<parquet::ColumnChunkMetaData> col_meta = rg_meta->ColumnChunk(col);
      const std::string& column_name = result.schema.field(static_cast<std::size_t>(col)).name;
      row_group.column_statistics.emplace(column_name, extract_column_statistics(*col_meta));
    }
    result.row_groups.push_back(std::move(row_group));
  }

  return result;
}

void validate_schema_compatibility(const std::vector<FileMetadata>& files) {
  if (files.size() < 2) return;
  const Schema& reference = files.front().schema;
  for (std::size_t i = 1; i < files.size(); ++i) {
    const Schema& candidate = files[i].schema;
    if (candidate.equals(reference)) continue;

    const std::size_t common = std::min(reference.field_count(), candidate.field_count());
    for (std::size_t f = 0; f < common; ++f) {
      if (!(reference.field(f) == candidate.field(f))) {
        throw StorageError("schema mismatch between '" + files.front().path.value() + "' and '" +
                           files[i].path.value() + "': field " + std::to_string(f) + " is " +
                           reference.field(f).name + " " + reference.field(f).type.to_string() + " vs " +
                           candidate.field(f).name + " " + candidate.field(f).type.to_string());
      }
    }
    throw StorageError("schema mismatch between '" + files.front().path.value() + "' and '" +
                       files[i].path.value() + "': different column counts (" +
                       std::to_string(reference.field_count()) + " vs " +
                       std::to_string(candidate.field_count()) + ")");
  }
}

}  // namespace kernellake
