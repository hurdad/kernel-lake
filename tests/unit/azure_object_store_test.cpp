#include <arrow/filesystem/azurefs.h>
#include <gtest/gtest.h>

#include "kernellake/common/config.hpp"
#include "kernellake/storage/azure_object_store.hpp"

namespace kernellake {
namespace {

TEST(AzureObjectStore, VendedSasTokenConstructorDoesNotThrow) {
  // Mirrors the vended-credentials constructor Unity Catalog's
  // UnityCatalogSourceResolver actually calls -- AzureFileSystem::Make()
  // doesn't touch the network at construction time (same as
  // S3FileSystem::Make()/GcsFileSystem::Make()), so this is safe to run
  // with no real Azure endpoint. account_name is required for
  // AzureOptions to be considered configured at all.
  arrow::fs::AzureOptions base_options;
  base_options.account_name = "devstoreaccount1";
  EXPECT_NO_THROW((void)(AzureObjectStore(base_options, "sv=2024-01-01&sig=vended-sas")));
}

}  // namespace
}  // namespace kernellake
