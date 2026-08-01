#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/file_discovery.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class LocalObjectStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ =
        fs::temp_directory_path() /
        fs::path("kernellake_storage_test_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" + test_info_name());
    fs::create_directories(dir_);
    write_file(dir_ / "a.parquet", "a");
    write_file(dir_ / "b.parquet", "b");
    write_file(dir_ / "c.txt", "not parquet");
  }

  void TearDown() override { fs::remove_all(dir_); }

  static std::string test_info_name() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  static void write_file(const fs::path& path, std::string_view content) {
    std::ofstream out(path, std::ios::binary);
    out << content;
  }

  fs::path dir_;
  LocalObjectStore store_;
};

TEST_F(LocalObjectStoreTest, ListsSingleFile) {
  const auto files = store_.list(Uri((dir_ / "a.parquet").string()));
  ASSERT_EQ(files.size(), 1u);
  EXPECT_EQ(files[0].size_bytes, 1u);
}

TEST_F(LocalObjectStoreTest, ListsGlobPatternSortedDeterministically) {
  const auto files = store_.list(Uri((dir_ / "*.parquet").string()));
  ASSERT_EQ(files.size(), 2u);
  EXPECT_TRUE(files[0].uri.value().ends_with("a.parquet"));
  EXPECT_TRUE(files[1].uri.value().ends_with("b.parquet"));
}

TEST_F(LocalObjectStoreTest, ListsDirectoryPickingOnlyParquetFiles) {
  const auto files = store_.list(Uri(dir_.string()));
  ASSERT_EQ(files.size(), 2u);
  for (const auto& file : files) {
    EXPECT_TRUE(file.uri.value().ends_with(".parquet"));
  }
}

TEST_F(LocalObjectStoreTest, ThrowsOnMissingPath) {
  EXPECT_THROW(store_.list(Uri((dir_ / "nonexistent.parquet").string())), StorageError);
}

TEST_F(LocalObjectStoreTest, ThrowsWhenGlobMatchesNothing) {
  EXPECT_THROW(store_.list(Uri((dir_ / "*.csv").string())), StorageError);
}

TEST_F(LocalObjectStoreTest, ThrowsOnMissingDirectoryForGlob) {
  EXPECT_THROW(store_.list(Uri((dir_ / "nonexistent_dir" / "*.parquet").string())), StorageError);
}

TEST_F(LocalObjectStoreTest, OpenReadsBackContent) {
  const auto handle = store_.open(Uri((dir_ / "a.parquet").string()));
  EXPECT_EQ(handle->size(), 1u);
  ASSERT_NE(handle->as_arrow_file(), nullptr);
}

TEST_F(LocalObjectStoreTest, OpenThrowsOnMissingFile) {
  EXPECT_THROW(store_.open(Uri((dir_ / "nonexistent.parquet").string())), StorageError);
}

TEST_F(LocalObjectStoreTest, FileDiscoveryRejectsNonParquetFile) {
  EXPECT_THROW(discover_parquet_files(store_, {(dir_ / "c.txt").string()}), StorageError);
}

TEST_F(LocalObjectStoreTest, FileDiscoveryMergesAndDedupsMultipleSources) {
  const auto files =
      discover_parquet_files(store_, {(dir_ / "a.parquet").string(), (dir_ / "*.parquet").string()});
  ASSERT_EQ(files.size(), 2u);  // a.parquet listed by both sources, deduplicated
}

TEST_F(LocalObjectStoreTest, FileDiscoveryRejectsEmptySourceList) {
  EXPECT_THROW(discover_parquet_files(store_, {}), StorageError);
}

TEST(Uri, DefaultsToFileScheme) {
  EXPECT_EQ(Uri("/data/sales.parquet").scheme(), "file");
  EXPECT_EQ(Uri("s3://bucket/key.parquet").scheme(), "s3");
}

}  // namespace
}  // namespace kernellake
