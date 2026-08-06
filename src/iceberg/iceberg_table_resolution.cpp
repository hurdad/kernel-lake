#include "kernellake/iceberg/iceberg_table_resolution.hpp"

#include <fmt/format.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/iceberg/manifest_reader.hpp"
#include "kernellake/iceberg/partition_pruning.hpp"
#include "kernellake/iceberg/schema_translation.hpp"
#include "kernellake/io/parquet_metadata.hpp"

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
  for (const ManifestListEntry& manifest_entry : manifest_list_entries) {
    if (manifest_entry.content == kManifestContentDeletes) {
      // Silently skipping these would mean returning rows the table's
      // current snapshot says are deleted -- a correctness bug, not a
      // missing feature, so this must fail loudly instead.
      throw StorageError(fmt::format(
          "iceberg table resolution: table has a live delete manifest ('{}') -- row-level deletes aren't "
          "supported yet, so this table can't be read correctly",
          manifest_entry.manifest_path));
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

  std::vector<ResolvedFile> resolved_files;
  resolved_files.reserve(live_data_files.size());
  for (const ManifestDataFileEntry& data_file : live_data_files) {
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
