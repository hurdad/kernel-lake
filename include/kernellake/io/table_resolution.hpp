#pragma once

#include <string>
#include <vector>

#include "kernellake/expression/expression.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/storage/object_store.hpp"
#include "kernellake/types/partition_column.hpp"
#include "kernellake/types/schema.hpp"

namespace kernellake {

// One physical Parquet file's full footer metadata (schema, row count,
// row groups with column statistics -- everything convert_scan() needs for
// row-group pruning) plus the partition-column values that apply to every
// row in it -- derived from its path (e.g.
// ".../region=US/date=2026-01-01/part-0.parquet"), never present in the
// file's own footer or metadata. Parallel to ResolvedTable::partition_columns
// (same size, same order).
struct ResolvedFile {
  FileMetadata metadata;
  std::vector<LiteralStorage> partition_values;
};

// The result of resolving a FROM-clause source into a concrete file list,
// schema (physical columns, plus any partition columns discovered from
// directory structure appended after them), and partition-column metadata.
// This is the one seam every future table-format/catalog integration
// (Iceberg, Delta, Unity Catalog -- see docs/ROADMAP.md) plugs into instead
// of calling discover_parquet_files()/inspect_parquet_file() directly.
struct ResolvedTable {
  std::vector<ResolvedFile> files;
  Schema schema;
  std::vector<PartitionColumn> partition_columns;
};

// Resolves `sources` (as given to `read_parquet(...)`) into concrete files,
// inspecting and validating every file's Parquet schema exactly as
// discover_parquet_files()/inspect_parquet_file() already do. If every
// discovered file's path yields the same non-empty sequence of Hive-style
// `key=value` directory segments immediately above the file itself, those
// segments become partition columns (type-inferred per-column: integer if
// every observed value parses as one, else an ISO-8601 date if every value
// is a valid calendar date, else string) appended to the returned schema.
// A source with no such segments on any file resolves with zero partition
// columns, identical to today's plain (non-partitioned) behavior. A source
// where files disagree on whether/how they're partitioned (some
// partitioned, some not, or different keys/depth) throws StorageError
// rather than guessing -- same "explicit errors over silent partial
// behavior" rule as discover_parquet_files() itself.
[[nodiscard]] ResolvedTable resolve_table(ObjectStore& store, const std::vector<std::string>& sources);

}  // namespace kernellake
