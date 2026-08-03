#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/storage/object_store_registry.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class ObjectStoreRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_object_store_registry_test_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" + test_info_name());
    fs::create_directories(dir_);
    std::ofstream out(dir_ / "a.parquet", std::ios::binary);
    out << "a";
  }

  void TearDown() override { fs::remove_all(dir_); }

  static std::string test_info_name() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  fs::path dir_;
};

// The registry must not eagerly construct any cloud backend at
// construction time -- an all-default StorageSection (no real S3/GCS/Azure
// credentials or endpoints configured) must not throw just from being
// wrapped, since nothing has referenced a matching URI scheme yet.
TEST_F(ObjectStoreRegistryTest, ConstructionIsLazyAndNeverThrows) {
  const StorageSection config;
  EXPECT_NO_THROW((void)({ ObjectStoreRegistry registry(config); }));
}

// "file" (the default/no-scheme case) dispatches to the same LocalObjectStore
// behavior as before -- a real file read through the registry, no network.
TEST_F(ObjectStoreRegistryTest, DispatchesFileSchemeToLocalBackend) {
  const StorageSection config;
  ObjectStoreRegistry registry(config);

  const std::vector<ObjectInfo> files = registry.list(Uri((dir_ / "a.parquet").string()));
  ASSERT_EQ(files.size(), 1u);
  EXPECT_EQ(files[0].uri.scheme(), "file");

  const std::unique_ptr<RandomAccessObject> object = registry.open(files[0].uri);
  EXPECT_EQ(object->size(), 1u);
}

// Real cloud backends (s3/gcs/azure) need either real credentials/endpoints
// or a real emulator to construct meaningfully -- covered by the manual
// smoke tests against real MinIO/fake-gcs-server/Azurite containers (see
// docs/ARCHITECTURE.md), not this offline unit test. What's safe and
// deterministic to assert here without any network is that an outright
// unrecognized scheme fails fast and clearly, before any backend
// construction is even attempted.
TEST_F(ObjectStoreRegistryTest, UnsupportedSchemeThrowsImmediately) {
  const StorageSection config;
  ObjectStoreRegistry registry(config);

  EXPECT_THROW((void)(registry.list(Uri("ftp://example.com/file.parquet"))), StorageError);
  EXPECT_THROW((void)(registry.open(Uri("ftp://example.com/file.parquet"))), StorageError);
}

}  // namespace
}  // namespace kernellake
