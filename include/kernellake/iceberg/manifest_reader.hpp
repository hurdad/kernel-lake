#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "kernellake/storage/object_store.hpp"

namespace kernellake::iceberg {

// One entry of a manifest-list Avro Object Container File
// (metadata/snap-*.avro): points at one manifest file, which in turn lists
// a batch of data/delete files. Mirrors the Iceberg spec's "manifest_file"
// struct -- only the subset this reader needs is extracted; the
// per-manifest partition-summary ("partitions") stats aren't (no pruning
// at this granularity yet, see docs/ROADMAP.md's lakehouse roadmap).
struct ManifestListEntry {
  std::string manifest_path;
  int64_t manifest_length = 0;
  int32_t partition_spec_id = 0;
  // 0 = data files, 1 = delete files (Iceberg v2 row/positional deletes,
  // not otherwise handled by this reader -- callers should skip
  // content == 1 entries until delete-file support exists).
  int32_t content = 0;
  int64_t added_snapshot_id = 0;
};

// One partition field's value out of a manifest entry's "partition"
// struct, decoded generically by position: this reader has no
// partition-spec/schema of its own to interpret field meaning against
// (that's a later integration step, see docs/ROADMAP.md), so a caller
// matches values back to partition-spec fields by index. Iceberg partition
// transforms only ever produce these two underlying Avro types (plus
// null, for a source value that was itself null).
using PartitionFieldValue = std::variant<std::monostate, int64_t, std::string>;

// One data/delete file entry out of a manifest Avro Object Container File
// (metadata/<uuid>-m0.avro). Mirrors the Iceberg spec's "manifest_entry" +
// nested "data_file" structs -- column-level stats (value_counts,
// lower_bounds/upper_bounds, etc.) are decoded by avro-c along with
// everything else in the record but intentionally not extracted here;
// nothing yet consumes them.
struct ManifestDataFileEntry {
  // 0 = EXISTING, 1 = ADDED, 2 = DELETED -- callers scanning a table's
  // *current* snapshot only want ADDED/EXISTING, never DELETED.
  int32_t status = 0;
  std::string file_path;
  std::string file_format;  // "PARQUET" | "ORC" | "AVRO"
  int64_t record_count = 0;
  int64_t file_size_in_bytes = 0;
  std::vector<PartitionFieldValue> partition_values;
  // v2 table metadata's data_file.content: 0 = DATA, 1 = POSITION_DELETES,
  // 2 = EQUALITY_DELETES -- present on every data_file record in a v2
  // table (both data and delete manifests), but only actually meaningful
  // to inspect from within a delete manifest (ManifestListEntry::content
  // == 1); a data manifest's entries are always 0 and this reader never
  // needs to check it there. Optional in the underlying Avro schema (v1
  // manifests, and every existing test fixture's own manifest schema,
  // predate this field) -- defaults to 0 (DATA) when absent, which is the
  // only value that would ever legitimately appear in a schema old enough
  // not to have the field at all (v1 has no row-level deletes, so no
  // manifest a v1-shaped schema could describe would ever need content=1
  // or 2 in the first place).
  int32_t content = 0;
};

// Reads a manifest-list Avro Object Container File's bytes (already fully
// read into memory by the caller) and returns its entries in file order.
// Throws StorageError on any malformed/unreadable input.
[[nodiscard]] std::vector<ManifestListEntry> read_manifest_list_bytes(const std::string& avro_bytes);

// Reads a manifest Avro Object Container File's bytes and returns its
// data/delete file entries in file order. Throws StorageError on any
// malformed/unreadable input.
[[nodiscard]] std::vector<ManifestDataFileEntry> read_manifest_bytes(const std::string& avro_bytes);

// Fetches `uri` in full via `store` and parses it as a manifest-list file.
[[nodiscard]] std::vector<ManifestListEntry> read_manifest_list(ObjectStore& store, const Uri& uri);

// Fetches `uri` in full via `store` and parses it as a manifest file.
[[nodiscard]] std::vector<ManifestDataFileEntry> read_manifest(ObjectStore& store, const Uri& uri);

}  // namespace kernellake::iceberg
