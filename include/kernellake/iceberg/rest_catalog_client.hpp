#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kernellake/common/config.hpp"

namespace kernellake::iceberg {

// One entry of Iceberg table metadata's "snapshots" array
// (https://iceberg.apache.org/spec/#table-metadata): enough to locate the
// manifest list for a given snapshot-id. Every other per-snapshot field
// (summary, parent-snapshot-id, schema-id, ...) is left unparsed -- not
// needed until a later phase (manifest reading, time travel) uses it.
struct IcebergSnapshot {
  int64_t snapshot_id = 0;
  std::string manifest_list;  // path/URI to the Avro manifest-list file
};

// One field of an Iceberg table schema's "fields" array
// (https://iceberg.apache.org/spec/#schemas), in the raw form the REST
// server sends it -- `type` is the unparsed Iceberg type string (e.g.
// "long", "decimal(10,2)"), not yet translated to a kernellake::DataType;
// see schema_translation.hpp for that step. Nested types (list/map/struct)
// aren't unpacked -- `type` would hold their JSON-object form, which
// schema_translation.hpp's translator rejects rather than silently
// mishandles (see its own comment for why nested types aren't supported
// yet).
struct IcebergSchemaField {
  int32_t id = 0;
  std::string name;
  bool required = false;
  std::string type;
};

// The subset of a REST Catalog "LoadTableResult" -> table metadata JSON
// this client extracts: enough for a manifest reader
// (src/iceberg/manifest_reader.cpp) to locate and read the current
// snapshot's manifest list, and for schema_translation.hpp to build a
// kernellake::Schema for the table's current columns. Partition-spec
// translation (interpreting a manifest entry's positional `partition`
// values against named partition-spec fields and their transform, e.g.
// bucket/truncate/day) is a separate, later integration step, not
// attempted here.
struct IcebergTableMetadata {
  std::string location;
  int32_t format_version = 0;
  std::optional<int64_t> current_snapshot_id;
  std::vector<IcebergSnapshot> snapshots;
  // The current schema's fields, in column order. Populated from the v2
  // "schemas"/"current-schema-id" pair, falling back to a bare v1
  // "schema" field if "schemas" is absent (older/compat REST servers).
  std::vector<IcebergSchemaField> schema_fields;

  // The manifest-list path for current_snapshot_id, or nullopt if the table
  // has no current snapshot (freshly created, no commits yet) or --
  // shouldn't happen against a spec-compliant server, but guarded against
  // rather than assumed -- current_snapshot_id doesn't appear in snapshots.
  [[nodiscard]] std::optional<std::string> current_manifest_list() const;
};

// A client for one named Iceberg REST catalog (see IcebergCatalogSection,
// kernellake/common/config.hpp): the spec's OAuth2 client-credentials token
// flow (POST {catalog_uri}/v1/oauth/tokens) or a pre-obtained static bearer
// token, plus GET {catalog_uri}/v1/{prefix}/namespaces/{ns}/tables/{table}
// table-metadata loads. Config is assumed already validated (see
// validate_config() in src/common/config.cpp) -- credentials_kind is one of
// the three known values, and the fields each requires are non-empty.
class IcebergRestCatalogClient final {
 public:
  explicit IcebergRestCatalogClient(IcebergCatalogSection config);

  IcebergRestCatalogClient(const IcebergRestCatalogClient&) = delete;
  IcebergRestCatalogClient& operator=(const IcebergRestCatalogClient&) = delete;

  // `namespace_parts` is the (possibly multi-level) namespace, e.g. {"db"}
  // or {"db", "schema"}; `table` the table name within it. Throws
  // StorageError on any transport failure, non-2xx HTTP response, or
  // malformed/incomplete response JSON.
  [[nodiscard]] IcebergTableMetadata load_table_metadata(const std::vector<std::string>& namespace_parts,
                                                         const std::string& table);

 private:
  [[nodiscard]] std::string bearer_token_for_request();
  [[nodiscard]] std::string fetch_oauth2_token();

  IcebergCatalogSection config_;
  std::string cached_oauth2_token_;
  double oauth2_token_expiry_unix_seconds_ = 0.0;
};

}  // namespace kernellake::iceberg
