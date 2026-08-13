#include "kernellake/delta/delta_table_resolution.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "kernellake/common/date_util.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/delta/schema_translation.hpp"
#include "kernellake/io/parquet_metadata.hpp"

namespace kernellake::delta {

namespace {

// AddFile.path is table-root-relative (see this file's own header comment)
// -- except for the rare, spec-legal case where a writer recorded an
// absolute URI directly, detected here the same way kernellake::Uri::scheme()
// does (a bare "file" scheme with no "://" is never a legitimate absolute
// Delta AddFile.path, so its absence is exactly the "needs joining" signal).
std::string resolve_file_uri(const std::string& table_uri, const std::string& file_path) {
  if (file_path.find("://") != std::string::npos) {
    return file_path;
  }
  if (!table_uri.empty() && table_uri.back() == '/') {
    return table_uri + file_path;
  }
  return table_uri + "/" + file_path;
}

LiteralStorage parse_delta_partition_value(const std::string& value, const DataType& type,
                                           const std::string& file_path, const std::string& column_name) {
  switch (type.id) {
    case TypeId::Boolean:
      if (value == "true") {
        return true;
      }
      if (value == "false") {
        return false;
      }
      throw StorageError(fmt::format(
          "delta table resolution: data file '{}' has partition value '{}' for boolean column '{}' -- "
          "expected 'true' or 'false'",
          file_path, value, column_name));
    case TypeId::Int32:
    case TypeId::Int64:
    case TypeId::UInt32:
    case TypeId::UInt64:
      try {
        return static_cast<std::int64_t>(std::stoll(value));
      } catch (const std::exception& e) {
        throw StorageError(fmt::format(
            "delta table resolution: data file '{}' has partition value '{}' for integer column '{}' that "
            "isn't a valid integer: {}",
            file_path, value, column_name, e.what()));
      }
    case TypeId::Float32:
    case TypeId::Float64:
      try {
        return std::stod(value);
      } catch (const std::exception& e) {
        throw StorageError(fmt::format(
            "delta table resolution: data file '{}' has partition value '{}' for floating-point column '{}' "
            "that isn't a valid number: {}",
            file_path, value, column_name, e.what()));
      }
    case TypeId::Date32:
      return static_cast<std::int64_t>(parse_iso_date(value));
    case TypeId::String:
    case TypeId::Decimal:
      return value;
    case TypeId::Timestamp:
      throw StorageError(
          fmt::format("delta table resolution: data file '{}' has a timestamp-typed partition column '{}' -- "
                      "timestamp partition values aren't supported yet",
                      file_path, column_name));
  }
  throw StorageError(
      fmt::format("delta table resolution: partition column '{}' has an unhandled type", column_name));
}

}  // namespace

ResolvedTable resolve_delta_table(ObjectStore& store, DeltaTxnClient& client, const std::string& table_uri) {
  const DeltaActiveFileListing listing = client.list_active_files(table_uri);
  const Schema full_schema = delta_schema_to_kernellake_schema(listing.table.schema_string);
  const std::vector<std::string>& partition_column_names = listing.table.partition_columns;

  std::vector<Field> physical_fields;
  physical_fields.reserve(full_schema.field_count());
  for (const Field& field : full_schema.fields()) {
    const bool is_partition_column = std::find(partition_column_names.begin(), partition_column_names.end(),
                                               field.name) != partition_column_names.end();
    if (!is_partition_column) {
      physical_fields.push_back(field);
    }
  }
  const Schema physical_schema(physical_fields);

  std::vector<PartitionColumn> partition_columns;
  partition_columns.reserve(partition_column_names.size());
  for (const std::string& name : partition_column_names) {
    const std::optional<std::size_t> index = full_schema.find_field(name);
    if (!index.has_value()) {
      throw StorageError(fmt::format(
          "delta table resolution: table '{}' declares partition column '{}' that isn't present in its "
          "own schema",
          table_uri, name));
    }
    partition_columns.push_back(PartitionColumn{name, full_schema.field(*index).type});
  }

  std::vector<Field> fields = physical_fields;
  for (const PartitionColumn& column : partition_columns) {
    fields.push_back(Field{column.name, column.type});
  }
  Schema schema(std::move(fields));

  std::vector<ResolvedFile> resolved_files;
  resolved_files.reserve(listing.files.size());
  for (const DeltaActiveFile& file : listing.files) {
    const Uri file_uri(resolve_file_uri(table_uri, file.path));
    FileMetadata file_metadata = inspect_parquet_file(store, file_uri);
    if (!file_metadata.schema.equals(physical_schema)) {
      const std::size_t common = std::min(physical_schema.field_count(), file_metadata.schema.field_count());
      for (std::size_t f = 0; f < common; ++f) {
        if (!(physical_schema.field(f) == file_metadata.schema.field(f))) {
          throw StorageError(fmt::format(
              "delta table resolution: data file '{}' doesn't match the table's current physical schema "
              "at field {}: table has {} {}, file has {} {} -- reading files written under a different "
              "(evolved) schema version isn't supported yet",
              file.path, f, physical_schema.field(f).name, physical_schema.field(f).type.to_string(),
              file_metadata.schema.field(f).name, file_metadata.schema.field(f).type.to_string()));
        }
      }
      throw StorageError(fmt::format(
          "delta table resolution: data file '{}' doesn't match the table's current physical schema: "
          "different column counts (table has {}, file has {}) -- reading files written under a different "
          "(evolved) schema version isn't supported yet",
          file.path, physical_schema.field_count(), file_metadata.schema.field_count()));
    }

    std::vector<LiteralStorage> partition_values;
    partition_values.reserve(partition_columns.size());
    for (const PartitionColumn& column : partition_columns) {
      const auto it = file.partition_values.find(column.name);
      if (it == file.partition_values.end()) {
        throw StorageError(
            fmt::format("delta table resolution: data file '{}' is missing a value for partition column '{}'",
                        file.path, column.name));
      }
      partition_values.push_back(
          parse_delta_partition_value(it->second, column.type, file.path, column.name));
    }
    resolved_files.push_back(ResolvedFile{std::move(file_metadata), std::move(partition_values)});
  }

  return ResolvedTable{std::move(resolved_files), std::move(schema), std::move(partition_columns)};
}

}  // namespace kernellake::delta
