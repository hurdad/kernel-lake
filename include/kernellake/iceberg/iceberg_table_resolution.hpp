#pragma once

#include <string>
#include <vector>

#include "kernellake/iceberg/rest_catalog_client.hpp"
#include "kernellake/io/table_resolution.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake::iceberg {

// Resolves one Iceberg table (identified by an already-constructed catalog
// client plus namespace/table name) into the same ResolvedTable shape
// kernellake::resolve_table() (plain/Hive-partitioned Parquet) produces, so
// both plug into the same downstream seam (convert_scan() and friends --
// see kernellake/io/table_resolution.hpp's own doc comment).
//
// Pipeline: IcebergRestCatalogClient::load_table_metadata() -> current
// snapshot's manifest list (read_manifest_list()) -> each data manifest's
// entries (read_manifest()) -> live (status ADDED/EXISTING) data files ->
// inspect_parquet_file() per file for row-group statistics.
//
// The table's *current* Iceberg schema (via schema_translation.hpp) is the
// authoritative column list -- not any individual data file's own Parquet
// footer schema, matching how Iceberg's schema evolution is meant to be
// interpreted (the table schema is independent of what any one file
// happens to contain). Every live data file's physical Parquet schema must
// currently match that schema *exactly*: reconciling genuine schema
// evolution across files (added/renamed/widened columns between snapshots)
// is real Iceberg behavior this resolver doesn't attempt yet, and a
// mismatch throws StorageError naming the offending file rather than
// silently misreading it -- see docs/ROADMAP.md's lakehouse roadmap.
//
// Row-level deletes (a manifest-list entry with content == 1) make silently
// ignoring them a correctness bug, not just a missing feature -- reading a
// table with any live delete manifest throws StorageError rather than
// returning rows that should have been deleted.
//
// Partition values are not extracted here (ResolvedTable::partition_columns
// is always empty from this path): unlike Hive-style directory
// partitioning, Iceberg's partition columns are already ordinary columns
// in the table's schema (present in every data file, for the common
// identity-transform case this resolver targets first) rather than values
// that must be reconstructed from something absent in the file -- so no
// partition-column materialization is needed for correctness.
//
// `predicates` (the WHERE clause's pushable predicates -- empty at
// schema-discovery time, real once the optimizer has run, see
// TableSourceResolver::resolve()'s own doc comment) drives file-level
// partition pruning: a data file whose manifest-recorded partition values
// (matched against the manifest-list entry's own partition_spec_id, so
// spec evolution across snapshots is handled correctly) prove no
// predicate could match is skipped entirely -- inspect_parquet_file() is
// never called on it at all, unlike row-group pruning (evaluate_pruning(),
// kernellake/io/parquet_pruning.hpp) which needs a file already open to
// read its footer first. See kernellake/iceberg/partition_pruning.hpp for
// exactly which transforms this can prove anything for.
[[nodiscard]] ResolvedTable resolve_iceberg_table(ObjectStore& store, IcebergRestCatalogClient& catalog,
                                                  const std::vector<std::string>& namespace_parts,
                                                  const std::string& table,
                                                  const std::vector<PushablePredicate>& predicates);

}  // namespace kernellake::iceberg
