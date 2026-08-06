#pragma once

#include <string>
#include <vector>

#include "kernellake/expression/expression.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/planner/logical_plan.hpp"
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

// Pluggable resolution for a source shape this library doesn't itself
// understand (currently: Iceberg's `iceberg://catalog.namespace.table`
// marker -- see kernellake::sql::parse_sql()'s `read_iceberg(...)`
// preprocessing and kernellake::iceberg::IcebergSourceResolver). Lives here
// as an abstract interface, not a concrete class, specifically so this
// library never needs to depend on kernellake_iceberg (which already
// depends on this one, for inspect_parquet_file()) -- a real implementation
// is constructed and injected by a higher layer (kernellake_api's
// QueryEngine) that can see both. Delta Lake/Unity Catalog integration
// (see docs/ROADMAP.md) is expected to plug in the same way.
class TableSourceResolver {
 public:
  virtual ~TableSourceResolver() = default;

  [[nodiscard]] virtual bool can_resolve(const std::vector<std::string>& sources) const = 0;
  // `predicates`: the WHERE clause's pushable predicates (see
  // PushablePredicate's own doc comment), for a resolver that can prune
  // whole files by metadata alone before ever opening one (see
  // kernellake::iceberg::resolve_iceberg_table()) -- empty when called
  // before the optimizer has computed them yet (QueryEngine::plan_logical()'s
  // own schema-discovery resolve, which only needs the schema/partition
  // columns, never the file list itself). A resolver that doesn't do its
  // own file-level pruning is free to ignore this parameter entirely (see
  // kernellake::delta::DeltaSourceResolver).
  [[nodiscard]] virtual ResolvedTable resolve(ObjectStore& store, const std::vector<std::string>& sources,
                                              const std::vector<PushablePredicate>& predicates) = 0;
};

// Delegates to `extra_resolver` when it's non-null and claims `sources`
// (via can_resolve()); falls back to the plain resolve_table() above
// otherwise. Both of this library's own resolve_table() call sites
// (QueryEngine::plan_logical()'s schema discovery, physical_planner.cpp's
// convert_scan()) go through this instead of calling resolve_table()
// directly, so a source kind resolve_table() can't handle itself still
// resolves consistently at both places. `predicates` is passed straight
// through to `extra_resolver->resolve()` -- see TableSourceResolver::resolve()'s
// own doc comment; resolve_table() itself (the plain/Hive-partitioned
// path) doesn't do file-level predicate pruning, so it's unused when no
// extra_resolver claims the source.
[[nodiscard]] ResolvedTable resolve_table_or_delegate(ObjectStore& store,
                                                      const std::vector<std::string>& sources,
                                                      TableSourceResolver* extra_resolver,
                                                      const std::vector<PushablePredicate>& predicates);

}  // namespace kernellake
