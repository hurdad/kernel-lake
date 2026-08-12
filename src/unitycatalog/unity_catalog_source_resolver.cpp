#include "kernellake/unitycatalog/unity_catalog_source_resolver.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/delta/delta_table_resolution.hpp"
#include "kernellake/delta/delta_txn_client.hpp"
#include "kernellake/iceberg/iceberg_table_resolution.hpp"
#include "kernellake/iceberg/rest_catalog_client.hpp"
#include "kernellake/storage/azure_object_store.hpp"
#include "kernellake/storage/gcs_object_store.hpp"
#include "kernellake/storage/s3_object_store.hpp"
#include "kernellake/unitycatalog/unity_catalog_client.hpp"

namespace kernellake::unitycatalog {

namespace {

constexpr std::string_view kScheme = "unitycatalog";

struct QualifiedName {
  std::string instance;
  std::string catalog;
  std::string schema;
  std::string table;
};

// "instance.catalog.schema.table" -> the four parts. Exactly four --
// unlike Iceberg's own namespace (which can nest arbitrarily deep), Unity
// Catalog's own naming is always exactly catalog.schema.table (a fixed
// 3-level hierarchy), plus this project's own instance-alias prefix in
// front (see UnityCatalogSourceResolver's own doc comment for why).
QualifiedName parse_qualified_name(const std::string& text) {
  std::vector<std::string> parts;
  std::string current;
  for (const char c : text) {
    if (c == '.') {
      parts.push_back(current);
      current.clear();
    } else {
      current += c;
    }
  }
  parts.push_back(current);

  const bool has_empty_part =
      std::any_of(parts.begin(), parts.end(), [](const std::string& p) { return p.empty(); });
  if (parts.size() != 4 || has_empty_part) {
    throw StorageError(fmt::format(
        "read_unity_catalog(...): '{}' isn't a valid instance.catalog.schema.table reference (need exactly "
        "4 non-empty, dot-separated parts: a configured instance name, then Unity Catalog's own "
        "catalog.schema.table)",
        text));
  }
  return QualifiedName{parts[0], parts[1], parts[2], parts[3]};
}

// Real Unity Catalog servers commonly report a local-filesystem
// storage_location as an explicit "file://" URI (confirmed against a
// real unitycatalog/unitycatalog OSS server, not assumed) -- but this
// codebase's own LocalObjectStore, unlike the S3/GCS/Azure backends
// (which all strip their own scheme prefix via
// generic_fs_object_store.cpp's strip_scheme()), expects a bare path with
// no scheme prefix at all, the same convention every other
// read_parquet(...)/read_delta(...) call in this project already
// follows. Stripped here rather than teaching LocalObjectStore a URI form
// nothing else in this codebase produces.
std::string strip_file_scheme(const std::string& location) {
  constexpr std::string_view kPrefix = "file://";
  if (location.rfind(kPrefix, 0) == 0) {
    return location.substr(kPrefix.size());
  }
  return location;
}

}  // namespace

bool UnityCatalogSourceResolver::can_resolve(const std::vector<std::string>& sources) const {
  return sources.size() == 1 && Uri(sources[0]).scheme() == kScheme;
}

ResolvedTable UnityCatalogSourceResolver::resolve(ObjectStore& store, const std::vector<std::string>& sources,
                                                  const std::vector<PushablePredicate>& /*predicates*/) {
  const std::string& source = sources.at(0);
  const std::size_t scheme_end = source.find("://");
  const QualifiedName name = parse_qualified_name(source.substr(scheme_end + 3));

  const auto it = unity_catalog_config_.instances.find(name.instance);
  if (it == unity_catalog_config_.instances.end()) {
    throw ConfigurationError(fmt::format(
        "read_unity_catalog(...): no unity_catalog.instances entry named '{}' is configured", name.instance));
  }

  // The client itself is still constructed fresh per call (unlike
  // IcebergSourceResolver's IcebergRestCatalogClient/DeltaSourceResolver's
  // DeltaTxnClient, which this class used to match exactly) -- get_table()/
  // list_*() must always hit the real server, so there's nothing else
  // worth caching here. Its OAuth2 *token*, though, can genuinely outlive
  // one client -- token_cache_, when non-null, lets it survive across
  // resolve() calls and separate queries (see UnityCatalogTokenCache's
  // own class comment).
  UnityCatalogClient client(it->second, token_cache_);
  const UnityCatalogTableInfo table = client.get_table(name.catalog, name.schema, name.table);

  // Only fetch and apply Unity Catalog's vended temporary credentials when
  // the table's storage actually lives in S3 -- a local-filesystem storage
  // location (real for a Unity Catalog server's own local/dev-mode
  // storage, and for this project's own tests) needs no credential
  // exchange at all, and building an S3ObjectStore for a non-"s3://"
  // location would fail outright rather than do anything useful. Left as
  // the shared `store` parameter (the caller's own ObjectStoreRegistry) in
  // that case, exactly like a plain read_parquet(...)/read_delta(...)
  // source already is.
  // A shared_ptr (not unique_ptr): once this table's ResolvedTable carries
  // it onward as ResolvedTable::owned_store (see that field's own doc
  // comment), it needs to survive past this function's own return long
  // enough for scan *execution* -- not just this resolve() call's own
  // metadata reads through data_store below -- to read the table's actual
  // file bytes through it.
  std::shared_ptr<ObjectStore> temp_store;
  ObjectStore* data_store = &store;
  const bool is_delta_or_parquet =
      table.data_source_format == "DELTA" || table.data_source_format == "PARQUET" ||
      table.data_source_format.empty();
  // Copied into an owned std::string, not left as the Uri::scheme()
  // string_view -- that view points into the temporary Uri object above,
  // which is destroyed at the end of this statement, making a
  // string_view-typed binding a real dangling-reference bug (caught by
  // this exact real-server verification: the "file" comparison below
  // silently read freed memory and never matched, so the "file://" prefix
  // never got stripped).
  const std::string storage_scheme(Uri(table.storage_location).scheme());
  const bool is_cloud_scheme = storage_scheme == "s3" || storage_scheme == "gs" || storage_scheme == "gcs" ||
                               storage_scheme == "abfs" || storage_scheme == "abfss" || storage_scheme == "az";
  if (is_delta_or_parquet && is_cloud_scheme) {
    // Same scheme-set ObjectStoreRegistry itself dispatches on
    // (object_store_registry.cpp) -- "gs"/"gcs" both mean GCS, "abfs"/
    // "abfss"/"az" all mean Azure.
    const UnityCatalogTemporaryCredentials credentials =
        client.get_temporary_table_credentials(table.table_id, "READ");
    if (storage_scheme == "s3") {
      temp_store = std::make_shared<S3ObjectStore>(s3_config_.options, credentials.access_key_id,
                                                   credentials.secret_access_key, credentials.session_token);
    } else if (storage_scheme == "gs" || storage_scheme == "gcs") {
      if (credentials.gcp_oauth_token.empty()) {
        throw StorageError(fmt::format(
            "read_unity_catalog(...): table '{}' is GCS-backed but Unity Catalog's temporary-credentials "
            "response carried no gcp_oauth_token",
            source));
      }
      temp_store =
          std::make_shared<GcsObjectStore>(gcs_config_.options, credentials.gcp_oauth_token);
    } else {
      if (credentials.azure_sas_token.empty()) {
        throw StorageError(fmt::format(
            "read_unity_catalog(...): table '{}' is Azure-backed but Unity Catalog's temporary-credentials "
            "response carried no azure_sas_token",
            source));
      }
      temp_store =
          std::make_shared<AzureObjectStore>(azure_config_.options, credentials.azure_sas_token);
    }
    data_store = temp_store.get();
  }
  const std::string effective_location =
      storage_scheme == "file" ? strip_file_scheme(table.storage_location) : table.storage_location;

  if (table.data_source_format == "DELTA") {
    if (delta_config_.grpc_endpoint.empty()) {
      throw ConfigurationError(fmt::format(
          "read_unity_catalog(...): table '{}' is Delta-formatted, but no delta.grpc_endpoint is "
          "configured (see DeltaSection in config.hpp) -- Unity Catalog only brokers the table's "
          "identity/credentials, not the Delta log itself",
          source));
    }
    delta::DeltaTxnClient delta_client(delta_config_);
    ResolvedTable result = delta::resolve_delta_table(*data_store, delta_client, effective_location);
    result.owned_store = temp_store;
    return result;
  }

  if (table.data_source_format == "ICEBERG") {
    IcebergCatalogSection iceberg_config;
    iceberg_config.catalog_uri = fmt::format("{}/iceberg", it->second.uc_url);
    iceberg_config.prefix = name.catalog;
    iceberg_config.credentials_kind = "bearer_token";
    iceberg_config.bearer_token = client.bearer_token_for_request();
    iceberg::IcebergRestCatalogClient iceberg_client(iceberg_config);
    // No temporary-credential fetch on this path (see this resolver's own
    // class-level doc comment) -- relies on static storage.s3/gcs/azure
    // config, same as a plain Iceberg REST catalog does today, so `store`
    // (the caller's own default) is the right one for execution too; no
    // owned_store to attach.
    return iceberg::resolve_iceberg_table(store, iceberg_client, {name.schema}, name.table, {});
  }

  if (is_delta_or_parquet) {
    ResolvedTable result = resolve_table(*data_store, {effective_location});
    result.owned_store = temp_store;
    return result;
  }

  throw StorageError(fmt::format(
      "read_unity_catalog(...): table '{}' has data_source_format '{}', which isn't supported yet (only "
      "DELTA, PARQUET, and ICEBERG are) -- see docs/ROADMAP.md",
      source, table.data_source_format));
}

}  // namespace kernellake::unitycatalog
