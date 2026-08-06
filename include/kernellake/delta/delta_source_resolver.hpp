#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/io/table_resolution.hpp"

namespace kernellake::delta {

// A TableSourceResolver (kernellake/io/table_resolution.hpp) for
// `read_delta('<table_uri>')` sources. kernellake::sql::parse_sql()'s
// preprocessing encodes one of these as a single source path
// "delta://<table_uri>" -- reusing this codebase's existing URI-scheme
// dispatch idiom (Uri::scheme(), the same mechanism
// kernellake::iceberg::IcebergSourceResolver already uses for
// "iceberg://catalog.namespace.table"). Unlike Iceberg's catalog-qualified
// name, a Delta table is addressed directly by its own storage URI --
// delta-txn-service has no catalog/namespace concept (see DeltaSection's
// own comment in common/config.hpp) -- so `<table_uri>` here is itself a
// full URI (e.g. "s3://bucket/warehouse/orders"), which is why the encoded
// source can contain a second "://" of its own; Uri::scheme() only looks at
// the first occurrence, so this composes correctly.
class DeltaSourceResolver final : public TableSourceResolver {
 public:
  explicit DeltaSourceResolver(DeltaSection config) : config_(std::move(config)) {}

  [[nodiscard]] bool can_resolve(const std::vector<std::string>& sources) const override;
  [[nodiscard]] ResolvedTable resolve(ObjectStore& store, const std::vector<std::string>& sources) override;

 private:
  DeltaSection config_;
};

}  // namespace kernellake::delta
