#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/io/table_resolution.hpp"

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
// AWS/S3 only for this slice -- GCS/Azure temporary credentials aren't
// requested or handled (see docs/ROADMAP.md). Constructed fresh per query,
// same deliberate MVP simplification IcebergSourceResolver/
// DeltaSourceResolver already use.
class UnityCatalogSourceResolver final : public TableSourceResolver {
 public:
  UnityCatalogSourceResolver(UnityCatalogSection unity_catalog_config, DeltaSection delta_config,
                             S3Section s3_config)
      : unity_catalog_config_(std::move(unity_catalog_config)),
        delta_config_(std::move(delta_config)),
        s3_config_(std::move(s3_config)) {}

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
};

}  // namespace kernellake::unitycatalog
