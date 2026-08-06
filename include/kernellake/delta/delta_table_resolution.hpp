#pragma once

#include <string>

#include "kernellake/delta/delta_txn_client.hpp"
#include "kernellake/io/table_resolution.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake::delta {

// Resolves one Delta table (identified by its own storage URI, addressed
// directly -- delta-txn-service has no catalog/namespace concept to key a
// map by, see DeltaSection's own comment in common/config.hpp) into the
// same ResolvedTable shape kernellake::resolve_table() (plain/Hive-
// partitioned Parquet) and kernellake::iceberg::resolve_iceberg_table()
// both already produce, so all three plug into the same downstream seam.
//
// Pipeline: one DeltaTxnClient::list_active_files() gRPC call returns the
// table's version/schema/partition-columns plus every currently-active
// data file, each already carrying its own AddFile.partitionValues -- no
// directory-name parsing needed, unlike Hive-style discovery -- followed by
// a per-file inspect_parquet_file() for row-group statistics.
//
// DeltaTableInfo::partition_columns names which of the full schema's
// fields are partition columns; those columns are never physically present
// in a data file's own Parquet footer (Delta's writers omit them, exactly
// like Hive-style partitioning does) -- split out from the full schema
// here, validated against each file's real physical schema, and appended
// back to the end of ResolvedTable::schema (the same field-ordering
// convention kernellake::resolve_table() and the Iceberg path both already
// use), with per-file values taken directly from that file's own
// partition_values map (typed per the partition column's own schema type)
// rather than inferred from a path -- there is no directory-name
// convention to infer from in Delta's own storage layout.
//
// Each AddFile.path is table-root-relative (delta-rs's own
// LogicalFileView::path() -- see this function's own implementation
// comment), not an absolute URI, so it's joined with `table_uri` here
// before being handed to inspect_parquet_file(). An AddFile.path that's
// already an absolute URI (containing "://") is used as-is -- valid per
// the Delta protocol spec, though rare in practice.
[[nodiscard]] ResolvedTable resolve_delta_table(ObjectStore& store, DeltaTxnClient& client,
                                                const std::string& table_uri);

}  // namespace kernellake::delta
