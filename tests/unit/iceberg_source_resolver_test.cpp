#include "kernellake/iceberg/iceberg_source_resolver.hpp"

#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake::iceberg {
namespace {

TEST(IcebergSourceResolver, CanResolveOnlyClaimsIcebergSchemeSources) {
  IcebergSourceResolver resolver{IcebergSection{}};
  EXPECT_TRUE(resolver.can_resolve({"iceberg://prod.db.orders"}));
  EXPECT_FALSE(resolver.can_resolve({"/data/sales.parquet"}));
  EXPECT_FALSE(resolver.can_resolve({"s3://bucket/table/"}));
  EXPECT_FALSE(resolver.can_resolve({}));
  EXPECT_FALSE(resolver.can_resolve({"iceberg://prod.db.orders", "iceberg://other.db.t"}));
}

TEST(IcebergSourceResolver, ThrowsOnUnknownCatalog) {
  IcebergSourceResolver resolver{IcebergSection{}};
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"iceberg://prod.db.orders"}, {})), ConfigurationError);
}

TEST(IcebergSourceResolver, ThrowsOnTooFewQualifiedNameParts) {
  IcebergCatalogSection catalog;
  catalog.catalog_uri = "http://localhost:8181";
  IcebergSection catalogs;
  catalogs.catalogs["prod"] = catalog;
  IcebergSourceResolver resolver{catalogs};
  LocalObjectStore store;
  // "prod.orders" has only 2 dot-separated parts (catalog + table, no
  // namespace level) -- rejected rather than guessed at as an empty
  // namespace.
  EXPECT_THROW((void)(resolver.resolve(store, {"iceberg://prod.orders"}, {})), StorageError);
}

TEST(IcebergSourceResolver, ThrowsOnEmptyQualifiedNamePart) {
  IcebergCatalogSection catalog;
  catalog.catalog_uri = "http://localhost:8181";
  IcebergSection catalogs;
  catalogs.catalogs["prod"] = catalog;
  IcebergSourceResolver resolver{catalogs};
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"iceberg://prod..orders"}, {})), StorageError);
}

}  // namespace
}  // namespace kernellake::iceberg
