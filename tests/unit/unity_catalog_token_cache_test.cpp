#include "kernellake/unitycatalog/unity_catalog_token_cache.hpp"

#include <gtest/gtest.h>

namespace kernellake::unitycatalog {
namespace {

TEST(UnityCatalogTokenCache, TryGetReturnsNulloptForUnknownKey) {
  const UnityCatalogTokenCache cache;
  EXPECT_EQ(cache.try_get("missing", /*now_unix_seconds=*/1000.0), std::nullopt);
}

TEST(UnityCatalogTokenCache, StoreThenTryGetReturnsTheStoredTokenBeforeExpiry) {
  UnityCatalogTokenCache cache;
  cache.store("instance-a", "token-a", /*expiry_unix_seconds=*/2000.0);

  const std::optional<std::string> found = cache.try_get("instance-a", /*now_unix_seconds=*/1000.0);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(*found, "token-a");
}

TEST(UnityCatalogTokenCache, TryGetReturnsNulloptOncePastExpiry) {
  UnityCatalogTokenCache cache;
  cache.store("instance-a", "token-a", /*expiry_unix_seconds=*/2000.0);

  EXPECT_EQ(cache.try_get("instance-a", /*now_unix_seconds=*/2000.0), std::nullopt);
  EXPECT_EQ(cache.try_get("instance-a", /*now_unix_seconds=*/2500.0), std::nullopt);
}

TEST(UnityCatalogTokenCache, DifferentKeysAreIndependent) {
  UnityCatalogTokenCache cache;
  cache.store("instance-a", "token-a", /*expiry_unix_seconds=*/2000.0);
  cache.store("instance-b", "token-b", /*expiry_unix_seconds=*/2000.0);

  EXPECT_EQ(*cache.try_get("instance-a", 1000.0), "token-a");
  EXPECT_EQ(*cache.try_get("instance-b", 1000.0), "token-b");
}

TEST(UnityCatalogTokenCache, StoreOverwritesAPreviousEntryForTheSameKey) {
  UnityCatalogTokenCache cache;
  cache.store("instance-a", "old-token", /*expiry_unix_seconds=*/2000.0);
  cache.store("instance-a", "new-token", /*expiry_unix_seconds=*/3000.0);

  EXPECT_EQ(*cache.try_get("instance-a", 1000.0), "new-token");
}

}  // namespace
}  // namespace kernellake::unitycatalog
