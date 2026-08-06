#include "kernellake/iceberg/iceberg_table_resolution.hpp"

#include <fmt/format.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/iceberg/manifest_reader.hpp"
#include "kernellake/iceberg/partition_pruning.hpp"
#include "kernellake/iceberg/position_delete_reader.hpp"
#include "kernellake/iceberg/schema_translation.hpp"
#include "kernellake/io/parquet_metadata.hpp"

#include <unordered_map>

namespace kernellake::iceberg {

namespace {

// Manifest-entry `status` (Iceberg spec): which entries represent a file
// actually live in the snapshot being read.
constexpr int32_t kManifestEntryStatusExisting = 0;
constexpr int32_t kManifestEntryStatusAdded = 1;
constexpr int32_t kManifestEntryStatusDeleted = 2;

// Manifest-list entry `content`: which manifests list data files vs.
// delete files.
constexpr int32_t kManifestContentData = 0;
constexpr int32_t kManifestContentDeletes = 1;

// data_file.content (present on every data_file record in a v2 manifest,
// data or delete alike -- see ManifestDataFileEntry::content's own
// comment): which kind of file a delete-manifest entry actually is. 0
// (DATA) is never expected here, only within an ordinary data manifest --
// see its own check below.
constexpr int32_t kDataFileContentPositionDeletes = 1;
constexpr int32_t kDataFileContentEqualityDeletes = 2;

bool is_live_status(int32_t status) {
  return status == kManifestEntryStatusExisting || status == kManifestEntryStatusAdded;
}

}  // namespace

ResolvedTable resolve_iceberg_table(ObjectStore& store, IcebergRestCatalogClient& catalog,
                                    const std::vector<std::string>& namespace_parts, const std::string& table,
                                    const std::vector<PushablePredicate>& predicates) {
  const IcebergTableMetadata table_metadata = catalog.load_table_metadata(namespace_parts, table);
  const Schema schema = iceberg_schema_to_kernellake_schema(table_metadata.schema_fields);

  const std::optional<std::string> manifest_list_path = table_metadata.current_manifest_list();
  if (!manifest_list_path.has_value()) {
    // No current snapshot -- a real, valid state for a freshly created
    // table with no commits yet, not an error: zero files, still a usable
    // (empty) schema.
    return ResolvedTable{{}, schema, {}};
  }

  const std::vector<ManifestListEntry> manifest_list_entries =
      read_manifest_list(store, Uri(*manifest_list_path));

  std::vector<ManifestDataFileEntry> live_data_files;
  // Every position-delete file's contribution, summed across every delete
  // manifest in this snapshot (a data file can be targeted by more than
  // one, e.g. from separate compaction passes) -- keyed by the *data*
  // file path each count applies to, not the delete file's own path.
  // Deliberately collected across the whole manifest list before being
  // consulted below, since a delete manifest can be listed before *or*
  // after the data manifest for the file it targets.
  std::unordered_map<std::string, std::int64_t> deleted_position_counts;

  for (const ManifestListEntry& manifest_entry : manifest_list_entries) {
    if (manifest_entry.content == kManifestContentDeletes) {
      for (ManifestDataFileEntry& delete_file : read_manifest(store, Uri(manifest_entry.manifest_path))) {
        if (!is_live_status(delete_file.status)) {
          continue;  // This delete file itself was superseded by a later snapshot.
        }
        if (delete_file.content == kDataFileContentEqualityDeletes) {
          throw StorageError(fmt::format(
              "iceberg table resolution: table has an equality delete file ('{}') -- equality deletes "
              "aren't supported yet, so this table can't be read correctly",
              delete_file.file_path));
        }
        if (delete_file.content != kDataFileContentPositionDeletes) {
          // Shouldn't happen against a spec-compliant writer (every entry
          // in a content==1 manifest-list entry's manifest should itself
          // be content==1 or 2 -- data_file.content's own default of 0
          // only legitimately applies within a *data* manifest) --
          // guarded against rather than assumed, since misreading an
          // unrecognized delete kind as a position delete would be a real
          // correctness risk.
          throw StorageError(fmt::format(
              "iceberg table resolution: delete manifest entry '{}' has unexpected data_file.content {} "
              "(expected POSITION_DELETES=1)",
              delete_file.file_path, delete_file.content));
        }
        if (delete_file.file_format != "PARQUET") {
          throw StorageError(fmt::format(
              "iceberg table resolution: position delete file '{}' has format '{}' -- only PARQUET is "
              "supported",
              delete_file.file_path, delete_file.file_format));
        }
        for (auto& [referenced_path, count] :
             read_position_delete_counts(store, Uri(delete_file.file_path))) {
          deleted_position_counts[referenced_path] += count;
        }
      }
      continue;
    }

    // The partition spec this manifest's data files were written under --
    // nullptr when the table's own metadata carries no "partition-specs"
    // at all (older/compat REST server, or an unpartitioned table), in
    // which case partition pruning simply never applies below (never a
    // correctness issue, see IcebergTableMetadata::find_partition_spec()'s
    // own comment).
    const IcebergPartitionSpec* spec = table_metadata.find_partition_spec(manifest_entry.partition_spec_id);

    for (ManifestDataFileEntry& data_file : read_manifest(store, Uri(manifest_entry.manifest_path))) {
      if (!is_live_status(data_file.status)) {
        continue;  // DELETED: superseded by a later snapshot, not part of this read.
      }
      if (data_file.file_format != "PARQUET") {
        throw StorageError(fmt::format(
            "iceberg table resolution: data file '{}' has format '{}' -- only PARQUET is supported",
            data_file.file_path, data_file.file_format));
      }
      if (spec != nullptr &&
          partition_values_prove_empty(table_metadata, *spec, data_file.partition_values, predicates)) {
        continue;  // Proven empty by partition pruning -- never opened at all.
      }
      live_data_files.push_back(std::move(data_file));
    }
  }

  // Now that deleted_position_counts is complete (every delete manifest in
  // the snapshot has been read), drop any data file every one of whose
  // rows was deleted; reject (rather than silently misread) any file with
  // only *some* rows deleted -- see this function's own header comment.
  std::vector<ManifestDataFileEntry> surviving_data_files;
  surviving_data_files.reserve(live_data_files.size());
  for (ManifestDataFileEntry& data_file : live_data_files) {
    const auto it = deleted_position_counts.find(data_file.file_path);
    if (it != deleted_position_counts.end()) {
      if (it->second >= data_file.record_count) {
        continue;  // Every row deleted -- drop the file entirely.
      }
      throw StorageError(fmt::format(
          "iceberg table resolution: data file '{}' has {} row(s) deleted out of {} -- partial row-level "
          "deletes aren't supported yet, only whole-file deletes",
          data_file.file_path, it->second, data_file.record_count));
    }
    surviving_data_files.push_back(std::move(data_file));
  }

  std::vector<ResolvedFile> resolved_files;
  resolved_files.reserve(surviving_data_files.size());
  for (const ManifestDataFileEntry& data_file : surviving_data_files) {
    FileMetadata file_metadata = inspect_parquet_file(store, Uri(data_file.file_path));
    if (!file_metadata.schema.equals(schema)) {
      // Same "name the first mismatched field" shape as
      // validate_schema_compatibility() (parquet_metadata.cpp) uses for the
      // plain/Hive path, just against the table's declared Iceberg schema
      // instead of another file's schema.
      const std::size_t common = std::min(schema.field_count(), file_metadata.schema.field_count());
      for (std::size_t f = 0; f < common; ++f) {
        if (!(schema.field(f) == file_metadata.schema.field(f))) {
          throw StorageError(fmt::format(
              "iceberg table resolution: data file '{}' doesn't match the table's current schema at field "
              "{}: table has {} {}, file has {} {} -- reading files written under a different (evolved) "
              "schema version isn't supported yet",
              data_file.file_path, f, schema.field(f).name, schema.field(f).type.to_string(),
              file_metadata.schema.field(f).name, file_metadata.schema.field(f).type.to_string()));
        }
      }
      throw StorageError(fmt::format(
          "iceberg table resolution: data file '{}' doesn't match the table's current schema: different "
          "column counts (table has {}, file has {}) -- reading files written under a different (evolved) "
          "schema version isn't supported yet",
          data_file.file_path, schema.field_count(), file_metadata.schema.field_count()));
    }
    resolved_files.push_back(ResolvedFile{std::move(file_metadata), /*partition_values=*/{}});
  }

  return ResolvedTable{std::move(resolved_files), schema, /*partition_columns=*/{}};
}

}  // namespace kernellake::iceberg
