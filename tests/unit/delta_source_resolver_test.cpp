#include "kernellake/delta/delta_source_resolver.hpp"

#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake::delta {
namespace {

TEST(DeltaSourceResolver, CanResolveOnlyClaimsDeltaSchemeSources) {
  DeltaSourceResolver resolver{DeltaSection{}};
  EXPECT_TRUE(resolver.can_resolve({"delta://s3://bucket/warehouse/orders"}));
  EXPECT_TRUE(resolver.can_resolve({"delta:///local/path/orders"}));
  EXPECT_FALSE(resolver.can_resolve({"/data/sales.parquet"}));
  EXPECT_FALSE(resolver.can_resolve({"s3://bucket/table/"}));
  EXPECT_FALSE(resolver.can_resolve({"iceberg://prod.db.orders"}));
  EXPECT_FALSE(resolver.can_resolve({}));
  EXPECT_FALSE(
      resolver.can_resolve({"delta://s3://bucket/warehouse/orders", "delta://s3://bucket/warehouse/other"}));
}

TEST(DeltaSourceResolver, ThrowsWhenGrpcEndpointIsNotConfigured) {
  DeltaSourceResolver resolver{DeltaSection{}};  // grpc_endpoint left empty
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"delta://s3://bucket/warehouse/orders"}, {})),
               ConfigurationError);
}

}  // namespace
}  // namespace kernellake::delta
