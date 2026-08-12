#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

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
// /temporary-table-credentials) this client extracts. `expiration_time`
// isn't captured for any cloud: these credentials are used to build one
// vended-credentialed ObjectStore for the duration of a single resolve()
// call and never cached past it (see UnityCatalogSourceResolver), so
// there is nothing here that would ever check an expiry against.
//
// UC's response carries exactly one of "aws_temp_credentials"/
// "gcp_oauth_token"/"azure_user_delegation_sas" depending on the table's
// actual cloud, so exactly one of the three field groups below is
// populated (the other two left default-empty) -- callers pick a
// dispatch target by checking which group is non-empty, the same
// resolve()-time scheme check (`s3://`/`gs://`/`abfs://`) that already
// decides which cloud a table's storage_location is on.
//
// The AWS field names/shape (access_key_id/secret_access_key/
// session_token under "aws_temp_credentials") were confirmed against a
// real unitycatalog/unitycatalog OSS server this project actually stood
// up (see docs/ROADMAP.md). The GCP/Azure field names below were not --
// no live GCP/Azure Unity Catalog deployment or credential was available
// to verify against, so `gcp_oauth_token`/`azure_user_delegation_sas`
// (and their own inner field names, "oauth_token"/"sas_token") are based
// on the same naming convention the AWS shape and the wider Databricks
// SDK use, not independently confirmed -- flagged here rather than
// silently presented with the same confidence as the AWS path.
struct UnityCatalogTemporaryCredentials {
  // AWS ("aws_temp_credentials") -- verified against a real server.
  std::string access_key_id;
  std::string secret_access_key;
  std::string session_token;
  // GCP ("gcp_oauth_token") -- field name not independently verified.
  std::string gcp_oauth_token;
  // Azure ("azure_user_delegation_sas") -- field name not independently
  // verified.
  std::string azure_sas_token;
};

// One entry of GET /catalogs' "catalogs" array.
struct UnityCatalogCatalogInfo {
  std::string name;
  std::string comment;
};

// One entry of GET /schemas' "schemas" array.
struct UnityCatalogSchemaInfo {
  std::string name;
  std::string catalog_name;
  std::string full_name;
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

  // list_catalogs()/list_schemas()/list_tables() each follow Unity
  // Catalog's own cursor-style pagination (a "next_page_token" in every
  // response, echoed back as a "page_token" query parameter on the next
  // request -- confirmed against a real unitycatalog/unitycatalog server,
  // both the field names and that the token is opaque, not assumed to be
  // parseable) internally, returning every page's results already
  // concatenated -- a caller never sees a partial listing or has to drive
  // the pagination loop itself. Each throws StorageError under the same
  // conditions get_table() does.
  [[nodiscard]] std::vector<UnityCatalogCatalogInfo> list_catalogs();
  [[nodiscard]] std::vector<UnityCatalogSchemaInfo> list_schemas(const std::string& catalog);
  [[nodiscard]] std::vector<UnityCatalogTableInfo> list_tables(const std::string& catalog,
                                                                const std::string& schema);

  // Exposed (not just used internally by get_table()/
  // get_temporary_table_credentials() above) so UnityCatalogSourceResolver
  // can reuse the exact same token when dispatching a UC-reported Iceberg
  // table to a freshly-built IcebergRestCatalogClient pointed at Unity
  // Catalog's own Iceberg-REST-compatible endpoint, rather than
  // authenticating to the same Unity Catalog instance twice.
  [[nodiscard]] std::string bearer_token_for_request();

 private:
  [[nodiscard]] std::string fetch_oauth2_token();
  // GET `url` with the current bearer token (if any) attached, parsed as
  // JSON -- shared by get_table() and the three list_*() methods, all of
  // which are plain authenticated GETs differing only in URL and response
  // shape.
  [[nodiscard]] nlohmann::json authenticated_get_json(const std::string& url);

  UnityCatalogInstanceSection config_;
  std::string cached_oauth2_token_;
  double oauth2_token_expiry_unix_seconds_ = 0.0;
};

}  // namespace kernellake::unitycatalog
