#include <arrow/filesystem/gcsfs.h>
#include <gtest/gtest.h>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/storage/gcs_object_store.hpp"

namespace kernellake {
namespace {

// GcsObjectStore's constructor validates access_token_expiration (a plain
// string-parsing check) before ever touching the network, so these are
// safe to run without a real GCS endpoint or emulator -- unlike list()/
// open(), which do need one.

TEST(GcsObjectStore, RejectsAccessTokenExpirationWithNonUtcOffset) {
  GcsSection config;
  config.credentials_kind = "access_token";
  config.access_token = "fake-token";
  // A real ISO-8601 timestamp, but with a timezone offset instead of "Z" --
  // std::get_time() only validates "%Y-%m-%dT%H:%M:%S" and would otherwise
  // leave "+05:00" unconsumed without failing, silently misinterpreting
  // this as UTC and producing a wrong expiration five hours off with no
  // error at all.
  config.access_token_expiration = "2026-01-01T00:00:00+05:00";

  EXPECT_THROW((void)(GcsObjectStore(config)), StorageError);
}

TEST(GcsObjectStore, RejectsAccessTokenExpirationMissingUtcSuffix) {
  GcsSection config;
  config.credentials_kind = "access_token";
  config.access_token = "fake-token";
  config.access_token_expiration = "2026-01-01T00:00:00";  // no trailing "Z" at all

  EXPECT_THROW((void)(GcsObjectStore(config)), StorageError);
}

TEST(GcsObjectStore, AcceptsValidUtcAccessTokenExpiration) {
  GcsSection config;
  config.credentials_kind = "access_token";
  config.access_token = "fake-token";
  config.access_token_expiration = "2026-01-01T00:00:00Z";

  // Construction itself may still fail later (no real GCS endpoint in this
  // test environment), but it must get *past* timestamp validation --
  // confirmed by asserting it does NOT throw the specific error
  // parse_iso8601_utc() raises for a malformed timestamp.
  try {
    GcsObjectStore store(config);
  } catch (const StorageError& e) {
    EXPECT_EQ(std::string(e.what()).find("is not a valid ISO-8601"), std::string::npos)
        << "valid UTC timestamp should not be rejected as malformed: " << e.what();
  }
}

TEST(GcsObjectStore, VendedAccessTokenConstructorDoesNotThrow) {
  // Mirrors the vended-credentials constructor Unity Catalog's
  // UnityCatalogSourceResolver actually calls (a fixed 1-hour placeholder
  // expiration, no timestamp parsing) -- like the static-config
  // constructor above, GcsFileSystem::Make() doesn't touch the network at
  // construction time, so this is safe to run with no real GCS endpoint.
  const arrow::fs::GcsOptions base_options = arrow::fs::GcsOptions::Defaults();
  EXPECT_NO_THROW((void)(GcsObjectStore(base_options, "vended-access-token")));
}

}  // namespace
}  // namespace kernellake
