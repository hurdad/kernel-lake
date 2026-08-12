#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kernellake/common/config.hpp"

namespace kernellake::unitycatalog {

// One entry of a Unity Catalog TableInfo's "columns" array. `type_json` is
// the raw type descriptor Unity Catalog sends (a JSON-encoded type spec,
// e.g. `"\"INT\""` for a primitive or a nested object for a struct/array/
// map) -- captured as text, not parsed into a kernellake::DataType here;
// that translation isn't needed by this slice's dispatch-only resolver
// (see unity_catalog_source_resolver.hpp), which reads a UC-resolved
// table's schema from the underlying Parquet/Delta/Iceberg metadata
// instead, the same way plain read_parquet(...)/read_delta(...)/
// read_iceberg(...) already do. `type_name` is UC's short type identifier
// (e.g. "INT", "STRUCT", "ARRAY") -- kept for a future real UC-type-to-
// Arrow-type mapping (see docs/ROADMAP.md), unused by this slice's dispatch
// logic today.
struct UnityCatalogColumn {
  std::string name;
  std::string type_name;
  std::string type_json;
  bool nullable = true;
  std::int32_t position = 0;
};

// The subset of Unity Catalog's TableInfo (GET /tables/{full_name}) this
// client extracts: enough for UnityCatalogSourceResolver to pick which
// existing format resolver (resolve_table()/resolve_delta_table()/
// resolve_iceberg_table()) to dispatch to and where that table's data
// actually lives. `table_id` is UC's own opaque identifier, required (not
// the catalog.schema.table name) by the temporary-table-credentials
// endpoint below.
struct UnityCatalogTableInfo {
  std::string table_id;
  std::string table_type;         // e.g. "MANAGED", "EXTERNAL"
  std::string data_source_format;  // e.g. "DELTA", "PARQUET", "ICEBERG"
  std::string storage_location;
  std::vector<UnityCatalogColumn> columns;
};

// The subset of Unity Catalog's TemporaryCredentials (POST
// /temporary-table-credentials) this client extracts -- AWS S3 only (see
// docs/ROADMAP.md's Unity Catalog entry for why GCS/Azure vended
// credentials are out of scope for this slice). `expiration_time` isn't
// captured: these credentials are used to build one S3ObjectStore for the
// duration of a single resolve() call and never cached past it (see
// UnityCatalogSourceResolver), so there is nothing here that would ever
// check an expiry against.
struct UnityCatalogTemporaryCredentials {
  std::string access_key_id;
  std::string secret_access_key;
  std::string session_token;
};

// A client for one named Unity Catalog instance (see
// UnityCatalogInstanceSection, kernellake/common/config.hpp): the same
// OAuth2 client-credentials token flow / static bearer token convention
// IcebergRestCatalogClient already uses, against Unity Catalog's own (not
// Iceberg-REST-spec) endpoints -- GET {uc_url}/tables/{catalog.schema.table}
// and POST {uc_url}/temporary-table-credentials. Unlike the Iceberg REST
// spec, Unity Catalog's OAuth2 token endpoint is not a fixed path under
// uc_url (Databricks: "https://<workspace>/oidc/v1/token"; other
// deployments may differ), so it's always read from
// config.oauth2_token_endpoint rather than derived. Config is assumed
// already validated (see validate_config() in src/common/config.cpp).
class UnityCatalogClient final {
 public:
  explicit UnityCatalogClient(UnityCatalogInstanceSection config);

  UnityCatalogClient(const UnityCatalogClient&) = delete;
  UnityCatalogClient& operator=(const UnityCatalogClient&) = delete;

  // Throws StorageError on any transport failure, non-2xx HTTP response, or
  // malformed/incomplete response JSON.
  [[nodiscard]] UnityCatalogTableInfo get_table(const std::string& catalog, const std::string& schema,
                                                const std::string& table);

  // `operation` is Unity Catalog's own vocabulary ("READ" is the only value
  // this project ever sends -- write support is out of scope, see
  // docs/ROADMAP.md).
  [[nodiscard]] UnityCatalogTemporaryCredentials get_temporary_table_credentials(const std::string& table_id,
                                                                                 const std::string& operation);

  // Exposed (not just used internally by get_table()/
  // get_temporary_table_credentials() above) so UnityCatalogSourceResolver
  // can reuse the exact same token when dispatching a UC-reported Iceberg
  // table to a freshly-built IcebergRestCatalogClient pointed at Unity
  // Catalog's own Iceberg-REST-compatible endpoint, rather than
  // authenticating to the same Unity Catalog instance twice.
  [[nodiscard]] std::string bearer_token_for_request();

 private:
  [[nodiscard]] std::string fetch_oauth2_token();

  UnityCatalogInstanceSection config_;
  std::string cached_oauth2_token_;
  double oauth2_token_expiry_unix_seconds_ = 0.0;
};

}  // namespace kernellake::unitycatalog
