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
  EXPECT_THROW((void)(store_.list(Uri((dir_ / "nonexistent.parquet").string()))), StorageError);
}

TEST_F(LocalObjectStoreTest, ThrowsWhenGlobMatchesNothing) {
  EXPECT_THROW((void)(store_.list(Uri((dir_ / "*.csv").string()))), StorageError);
}

TEST_F(LocalObjectStoreTest, ThrowsOnMissingDirectoryForGlob) {
  EXPECT_THROW((void)(store_.list(Uri((dir_ / "nonexistent_dir" / "*.parquet").string()))), StorageError);
}

TEST_F(LocalObjectStoreTest, OpenReadsBackContent) {
  const auto handle = store_.open(Uri((dir_ / "a.parquet").string()));
  EXPECT_EQ(handle->size(), 1u);
  ASSERT_NE(handle->as_arrow_file(), nullptr);
}

TEST_F(LocalObjectStoreTest, OpenThrowsOnMissingFile) {
  EXPECT_THROW((void)(store_.open(Uri((dir_ / "nonexistent.parquet").string()))), StorageError);
}

TEST_F(LocalObjectStoreTest, FileDiscoveryRejectsNonParquetFile) {
  EXPECT_THROW((void)(discover_parquet_files(store_, {(dir_ / "c.txt").string()})), StorageError);
}

TEST_F(LocalObjectStoreTest, FileDiscoveryMergesAndDedupsMultipleSources) {
  const auto files =
      discover_parquet_files(store_, {(dir_ / "a.parquet").string(), (dir_ / "*.parquet").string()});
  ASSERT_EQ(files.size(), 2u);  // a.parquet listed by both sources, deduplicated
}

TEST_F(LocalObjectStoreTest, FileDiscoveryRejectsEmptySourceList) {
  EXPECT_THROW((void)(discover_parquet_files(store_, {})), StorageError);
}

// Regression test: fs::directory_iterator's own constructor/increment can
// throw fs::filesystem_error (e.g. this unreadable subdirectory) -- list()
// used to let that std::filesystem exception type escape uncaught instead
// of converting it to this file's own StorageError like every other
// failure path here.
TEST_F(LocalObjectStoreTest, ListThrowsStorageErrorOnUnreadableDirectory) {
  const fs::path unreadable = dir_ / "unreadable";
  fs::create_directories(unreadable);
  write_file(unreadable / "hidden.parquet", "x");
  fs::permissions(unreadable, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
  struct RestorePerms {
    fs::path path;
    ~RestorePerms() { fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace); }
  } restore{unreadable};

  EXPECT_THROW((void)(store_.list(Uri(unreadable.string()))), StorageError);
}

TEST_F(LocalObjectStoreTest, ListRecursiveThrowsStorageErrorOnUnreadableSubdirectory) {
  const fs::path unreadable = dir_ / "unreadable_recursive";
  fs::create_directories(unreadable);
  write_file(unreadable / "hidden.parquet", "x");
  fs::permissions(unreadable, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
  struct RestorePerms {
    fs::path path;
    ~RestorePerms() { fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace); }
  } restore{unreadable};

  EXPECT_THROW((void)(store_.list_recursive(Uri(dir_.string()))), StorageError);
}

class LocalObjectStoreRootConfinementTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ =
        fs::temp_directory_path() /
        fs::path("kernellake_root_confinement_test_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" + test_info_name());
    fs::create_directories(root_ / "allowed");
    write_file(root_ / "allowed" / "a.parquet", "a");
    write_file(outside_file(), "outside");
  }

  void TearDown() override {
    fs::remove_all(root_);
    fs::remove(outside_file());
  }

  static std::string test_info_name() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  static void write_file(const fs::path& path, std::string_view content) {
    std::ofstream out(path, std::ios::binary);
    out << content;
  }

  // A sibling of root_, i.e. genuinely outside it -- not just a
  // differently-named prefix (e.g. "root_2" vs "root_"), which a naive
  // string-prefix check would wrongly treat as contained.
  fs::path outside_file() const {
    return root_.parent_path() / (root_.filename().string() + "_sibling.parquet");
  }

  fs::path root_;
};

TEST_F(LocalObjectStoreRootConfinementTest, OpenWithinRootSucceeds) {
  LocalObjectStore store(root_.string());
  const auto handle = store.open(Uri((root_ / "allowed" / "a.parquet").string()));
  EXPECT_EQ(handle->size(), 1u);
}

TEST_F(LocalObjectStoreRootConfinementTest, OpenOutsideRootThrows) {
  LocalObjectStore store(root_.string());
  EXPECT_THROW((void)(store.open(Uri(outside_file().string()))), StorageError);
}

TEST_F(LocalObjectStoreRootConfinementTest, OpenViaDotDotEscapeThrows) {
  LocalObjectStore store(root_.string());
  const std::string escaped = (root_ / "allowed" / ".." / ".." / outside_file().filename()).string();
  EXPECT_THROW((void)(store.open(Uri(escaped))), StorageError);
}

TEST_F(LocalObjectStoreRootConfinementTest, ListOutsideRootThrows) {
  LocalObjectStore store(root_.string());
  EXPECT_THROW((void)(store.list(Uri(outside_file().string()))), StorageError);
}

TEST_F(LocalObjectStoreRootConfinementTest, ListGlobOutsideRootThrows) {
  LocalObjectStore store(root_.string());
  EXPECT_THROW((void)(store.list(Uri((root_.parent_path() / "*.parquet").string()))), StorageError);
}

TEST_F(LocalObjectStoreRootConfinementTest, DefaultRootAllowsAnyAbsolutePath) {
  LocalObjectStore store;  // default local_root == "/": no confinement, matches prior behavior
  const auto handle = store.open(Uri(outside_file().string()));
  EXPECT_EQ(handle->size(), 7u);
}

TEST(Uri, DefaultsToFileScheme) {
  EXPECT_EQ(Uri("/data/sales.parquet").scheme(), "file");
  EXPECT_EQ(Uri("s3://bucket/key.parquet").scheme(), "s3");
}

}  // namespace
}  // namespace kernellake
