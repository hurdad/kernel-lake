#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/io/table_resolution.hpp"

namespace kernellake::iceberg {

// A TableSourceResolver (kernellake/io/table_resolution.hpp) for
// `read_iceberg('catalog.namespace.table')` sources. kernellake::sql::
// parse_sql()'s preprocessing encodes one of these as a single source path
// "iceberg://catalog.namespace.table" -- reusing this codebase's existing
// URI-scheme dispatch idiom (Uri::scheme(), the same mechanism
// ObjectStoreRegistry already uses to pick S3/GCS/Azure/HDFS) rather than
// inventing a separate "source kind" concept that would have to be
// threaded through the AST/binder/LogicalScan.
class IcebergSourceResolver final : public TableSourceResolver {
 public:
  explicit IcebergSourceResolver(IcebergSection catalogs) : catalogs_(std::move(catalogs)) {}

  [[nodiscard]] bool can_resolve(const std::vector<std::string>& sources) const override;
  // `predicates` is forwarded to resolve_iceberg_table() for file-level
  // partition pruning -- see partition_pruning.hpp and
  // TableSourceResolver::resolve()'s own doc comment.
  [[nodiscard]] ResolvedTable resolve(ObjectStore& store, const std::vector<std::string>& sources,
                                      const std::vector<PushablePredicate>& predicates) override;

 private:
  IcebergSection catalogs_;
};

}  // namespace kernellake::iceberg
