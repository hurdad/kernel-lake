#include "kernellake/unitycatalog/unity_catalog_source_resolver.hpp"

#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake::unitycatalog {
namespace {

UnityCatalogSourceResolver make_resolver(UnityCatalogSection instances = {}, DeltaSection delta = {},
                                         S3Section s3 = {}) {
  return UnityCatalogSourceResolver(std::move(instances), std::move(delta), std::move(s3));
}

TEST(UnityCatalogSourceResolver, CanResolveOnlyClaimsUnityCatalogSchemeSources) {
  UnityCatalogSourceResolver resolver = make_resolver();
  EXPECT_TRUE(resolver.can_resolve({"unitycatalog://prod.main.db.orders"}));
  EXPECT_FALSE(resolver.can_resolve({"/data/sales.parquet"}));
  EXPECT_FALSE(resolver.can_resolve({"s3://bucket/table/"}));
  EXPECT_FALSE(resolver.can_resolve({"iceberg://prod.db.orders"}));
  EXPECT_FALSE(resolver.can_resolve({}));
  EXPECT_FALSE(
      resolver.can_resolve({"unitycatalog://prod.main.db.orders", "unitycatalog://prod.main.db.other"}));
}

TEST(UnityCatalogSourceResolver, ThrowsOnUnknownInstance) {
  UnityCatalogSourceResolver resolver = make_resolver();
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {})), ConfigurationError);
}

TEST(UnityCatalogSourceResolver, ThrowsOnTooFewQualifiedNameParts) {
  UnityCatalogInstanceSection instance;
  instance.uc_url = "http://localhost:8080/api/2.1/unity-catalog";
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;
  UnityCatalogSourceResolver resolver = make_resolver(instances);
  LocalObjectStore store;
  // "prod.db.orders" has only 3 dot-separated parts (instance + catalog +
  // table, no schema level) -- Unity Catalog's own naming is always
  // exactly catalog.schema.table, so this is rejected rather than guessed
  // at.
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod.db.orders"}, {})), StorageError);
}

TEST(UnityCatalogSourceResolver, ThrowsOnTooManyQualifiedNameParts) {
  UnityCatalogInstanceSection instance;
  instance.uc_url = "http://localhost:8080/api/2.1/unity-catalog";
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;
  UnityCatalogSourceResolver resolver = make_resolver(instances);
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod.main.db.schema.orders"}, {})),
               StorageError);
}

TEST(UnityCatalogSourceResolver, ThrowsOnEmptyQualifiedNamePart) {
  UnityCatalogInstanceSection instance;
  instance.uc_url = "http://localhost:8080/api/2.1/unity-catalog";
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;
  UnityCatalogSourceResolver resolver = make_resolver(instances);
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod..db.orders"}, {})), StorageError);
}

}  // namespace
}  // namespace kernellake::unitycatalog
