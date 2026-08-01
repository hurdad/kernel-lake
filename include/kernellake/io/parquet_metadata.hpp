#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernellake/expression/expression.hpp"
#include "kernellake/storage/object_store.hpp"
#include "kernellake/types/schema.hpp"

namespace kernellake {

// Column-chunk statistics for one row group. Populated only when Parquet
// actually recorded them; has_min_max/has_null_count are false (rather than
// a guessed value) when the writer omitted, truncated, or could not encode
// them, or when the physical type is not one KernelLake currently decodes
// (INT96, FIXED_LEN_BYTE_ARRAY/DECIMAL). Pruning must treat "false" as "do
// not prune this column here", never as "assume no nulls" or similar.
struct ColumnStatistics {
  bool has_min_max = false;
  bool has_null_count = false;
  std::int64_t null_count = 0;
  // Uses the same LiteralStorage variant as LiteralExpression so pruning
  // can reuse the same value representation (dates/timestamps as the
  // int64_t alternative, matching LiteralExpression::make_date32/timestamp).
  LiteralStorage min_value;
  LiteralStorage max_value;
};

struct RowGroupMetadata {
  int index = 0;
  std::int64_t row_count = 0;
  std::int64_t compressed_size_bytes = 0;
  std::int64_t uncompressed_size_bytes = 0;
  std::unordered_map<std::string, ColumnStatistics> column_statistics;
};

struct FileMetadata {
  Uri path;
  Schema schema{std::vector<Field>{}};
  std::int64_t row_count = 0;
  std::vector<RowGroupMetadata> row_groups;
};

// Opens `path` through `store` and reads its Parquet footer metadata:
// schema, row count, row groups, and per-row-group column statistics. Does
// not decode any column data. Throws StorageError with the file path on any
// failure to open or parse the footer.
[[nodiscard]] FileMetadata inspect_parquet_file(ObjectStore& store, const Uri& path);

// Validates that every file has the same field names and types (order
// included) as the first. Throws StorageError naming the first mismatched
// file, its differing field, and both types, if any file disagrees.
void validate_schema_compatibility(const std::vector<FileMetadata>& files);

}  // namespace kernellake
