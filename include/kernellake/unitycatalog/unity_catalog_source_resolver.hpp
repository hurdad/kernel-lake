#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/io/table_resolution.hpp"
#include "kernellake/unitycatalog/unity_catalog_token_cache.hpp"

namespace kernellake::unitycatalog {

// A TableSourceResolver (kernellake/io/table_resolution.hpp) for
// `read_unity_catalog('instance.catalog.schema.table')` sources.
// kernellake::sql::parse_sql()'s preprocessing encodes one of these as a
// single source path "unitycatalog://instance.catalog.schema.table" --
// the same URI-scheme dispatch idiom IcebergSourceResolver/
// DeltaSourceResolver already use. `instance` (not part of Unity Catalog's
// own catalog.schema.table naming) picks a configured
// UnityCatalogInstanceSection -- a deployment may talk to more than one
// Unity Catalog workspace, the same reason IcebergSection::catalogs is a
// map keyed by name rather than a single section.
//
// A Unity Catalog table is not itself a storage/table format -- it's
// Delta, plain Parquet, or Iceberg underneath, all three of which
// KernelLake can already read. resolve() authenticates, looks up the
// table's format and storage location, and dispatches to whichever
// existing resolution path matches:
//   - DELTA: if the table's storage_location is "s3://...", fetches AWS
//     temporary credentials (see UnityCatalogClient::
//     get_temporary_table_credentials()) and builds a short-lived
//     S3ObjectStore from them; otherwise uses the caller's own ObjectStore
//     unchanged (a local-filesystem storage_location needs no credential
//     exchange). Either way, calls the existing delta::resolve_delta_table()
//     (requires unity_catalog's paired delta.grpc_endpoint to be
//     configured, the same prerequisite plain read_delta(...) already
//     has).
//   - PARQUET (or any other externally-stored format with no recognized
//     log): same scheme-conditional temporary-credential fetch, then the
//     existing plain kernellake::resolve_table().
//   - ICEBERG: Unity Catalog natively exposes an Iceberg-REST-compatible
//     endpoint for tables it manages as Iceberg, so this builds an
//     IcebergCatalogSection pointed at it (reusing the bearer token this
//     resolver's own UnityCatalogClient already obtained) and calls the
//     existing iceberg::resolve_iceberg_table() -- no temporary-credential
//     fetch on this path; relies on static storage.s3 access, same as a
//     plain Iceberg REST catalog does today.
// Any other data_source_format throws StorageError naming it explicitly.
//
// S3, GCS, and Azure vended credentials are all handled (dispatched by the
// storage_location URI's own scheme: "s3", "gs", "abfs"/"abfss"). Only the
// AWS "aws_temp_credentials" response shape has been verified against a
// real Unity Catalog server; the GCP/Azure field names
// (UnityCatalogTemporaryCredentials::gcp_oauth_token/azure_sas_token) are
// unverified, based on Databricks SDK naming conventions only -- see that
// struct's own comment. Constructed fresh per query, same deliberate MVP
// simplification IcebergSourceResolver/DeltaSourceResolver already use --
// unlike its backing UnityCatalogClient's own OAuth2 token, though, which
// this resolver can share across queries via `token_cache` (see
// UnityCatalogTokenCache's own class comment).
class UnityCatalogSourceResolver final : public TableSourceResolver {
 public:
  // `token_cache`, when non-null, is handed to every UnityCatalogClient
  // this resolver constructs inside resolve() -- the caller (QueryEngine)
  // owns it and must keep it alive for at least as long as this resolver.
  // Defaults to nullptr (token caching then stays scoped to whichever
  // single UnityCatalogClient a given resolve() call constructs, this
  // class's original behavior) so existing call sites/tests keep
  // compiling and behaving identically without passing one.
  UnityCatalogSourceResolver(UnityCatalogSection unity_catalog_config, DeltaSection delta_config,
                             S3Section s3_config, GcsSection gcs_config, AzureSection azure_config,
                             const UnityCatalogTokenCache* token_cache = nullptr)
      : unity_catalog_config_(std::move(unity_catalog_config)),
        delta_config_(std::move(delta_config)),
        s3_config_(std::move(s3_config)),
        gcs_config_(std::move(gcs_config)),
        azure_config_(std::move(azure_config)),
        token_cache_(token_cache) {}

  [[nodiscard]] bool can_resolve(const std::vector<std::string>& sources) const override;
  // `predicates` is currently unused: none of the three dispatch targets
  // this resolver reaches does its own file-level predicate pruning beyond
  // what they already do for a plain read_parquet(...)/read_delta(...)/
  // read_iceberg(...) source. Still accepted, to satisfy the
  // TableSourceResolver interface.
  [[nodiscard]] ResolvedTable resolve(ObjectStore& store, const std::vector<std::string>& sources,
                                      const std::vector<PushablePredicate>& predicates) override;

 private:
  UnityCatalogSection unity_catalog_config_;
  DeltaSection delta_config_;
  S3Section s3_config_;
  GcsSection gcs_config_;
  AzureSection azure_config_;
  const UnityCatalogTokenCache* token_cache_;
};

}  // namespace kernellake::unitycatalog
