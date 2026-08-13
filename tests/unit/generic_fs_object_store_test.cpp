// Direct unit tests for kernellake::detail's shared list()/open()
// implementation (src/storage/generic_fs_object_store.cpp), the real logic
// behind every cloud ObjectStore backend's (S3/GCS/Azure/HDFS) list()/open()
// -- previously exercised only indirectly, through backends that all need
// real cloud credentials/endpoints to construct against. glob_match()/
// generic_fs_list()/generic_fs_list_recursive()/generic_fs_open() are all
// generic over arrow::fs::FileSystem, so a real arrow::fs::LocalFileSystem
// drives them here instead -- no mocking, no cloud credentials needed, same
// spirit as storage_test.cpp's LocalObjectStore tests.
#include <gtest/gtest.h>

#include <arrow/buffer.h>
#include <arrow/filesystem/localfs.h>

#include <filesystem>
#include <fstream>
#include <memory>

#include "generic_fs_object_store.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

TEST(GlobMatch, MatchesStarAndQuestionMarkWildcards) {
  EXPECT_TRUE(detail::glob_match("*.parquet", "part-0.parquet"));
  EXPECT_TRUE(detail::glob_match("part-?.parquet", "part-0.parquet"));
  EXPECT_TRUE(detail::glob_match("*", "anything"));
  EXPECT_TRUE(detail::glob_match("exact.parquet", "exact.parquet"));
}

TEST(GlobMatch, RejectsNonMatchingText) {
  EXPECT_FALSE(detail::glob_match("*.parquet", "data.csv"));
  EXPECT_FALSE(detail::glob_match("part-?.parquet", "part-10.parquet"));
  EXPECT_FALSE(detail::glob_match("exact.parquet", "exact.parquet.bak"));
}

class GenericFsObjectStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_generic_fs_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    write_file(dir_ / "a.parquet", "aaa");
    write_file(dir_ / "b.parquet", "bb");
    write_file(dir_ / "c.txt", "not parquet");
  }

  void TearDown() override { fs::remove_all(dir_); }

  static void write_file(const fs::path& path, std::string_view content) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << content;
  }

  fs::path dir_;
  std::shared_ptr<arrow::fs::FileSystem> fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
};

TEST_F(GenericFsObjectStoreTest, ListGlobMatchesMultipleFilesSortedDeterministically) {
  const auto files = detail::generic_fs_list(fs_, "test", Uri((dir_ / "*.parquet").string()));
  ASSERT_EQ(files.size(), 2u);
  EXPECT_TRUE(files[0].uri.value().ends_with("a.parquet"));
  EXPECT_TRUE(files[1].uri.value().ends_with("b.parquet"));
  EXPECT_EQ(files[0].size_bytes, 3u);
  EXPECT_EQ(files[1].size_bytes, 2u);
}

TEST_F(GenericFsObjectStoreTest, ListGlobThrowsWhenNothingMatches) {
  EXPECT_THROW((void)(detail::generic_fs_list(fs_, "test", Uri((dir_ / "*.csv").string()))), StorageError);
}

TEST_F(GenericFsObjectStoreTest, ListSingleFileDispatchesDirectly) {
  const auto files = detail::generic_fs_list(fs_, "test", Uri((dir_ / "a.parquet").string()));
  ASSERT_EQ(files.size(), 1u);
  EXPECT_TRUE(files[0].uri.value().ends_with("a.parquet"));
  EXPECT_EQ(files[0].size_bytes, 3u);
}

TEST_F(GenericFsObjectStoreTest, ListDirectoryDispatchesToOnlyParquetFiles) {
  const auto files = detail::generic_fs_list(fs_, "test", Uri(dir_.string()));
  ASSERT_EQ(files.size(), 2u);
  for (const auto& file : files) {
    EXPECT_TRUE(file.uri.value().ends_with(".parquet"));
  }
}

TEST_F(GenericFsObjectStoreTest, ListThrowsOnMissingPath) {
  EXPECT_THROW((void)(detail::generic_fs_list(fs_, "test", Uri((dir_ / "missing.parquet").string()))),
               StorageError);
}

TEST_F(GenericFsObjectStoreTest, ListThrowsWhenDirectoryContainsNoParquetFiles) {
  const fs::path empty_of_parquet = dir_ / "no_parquet_here";
  fs::create_directories(empty_of_parquet);
  write_file(empty_of_parquet / "readme.txt", "not parquet");
  EXPECT_THROW((void)(detail::generic_fs_list(fs_, "test", Uri(empty_of_parquet.string()))), StorageError);
}

TEST_F(GenericFsObjectStoreTest, ListRecursiveEnumeratesNestedParquetFiles) {
  write_file(dir_ / "nested" / "deeper" / "d.parquet", "dddd");
  const auto files = detail::generic_fs_list_recursive(fs_, "test", Uri(dir_.string()));
  ASSERT_EQ(files.size(), 3u);  // a.parquet, b.parquet, nested/deeper/d.parquet
  bool saw_nested = false;
  for (const auto& file : files) {
    if (file.uri.value().ends_with("d.parquet")) {
      saw_nested = true;
      EXPECT_EQ(file.size_bytes, 4u);
    }
  }
  EXPECT_TRUE(saw_nested);
}

TEST_F(GenericFsObjectStoreTest, ListRecursiveThrowsWhenGivenAFileNotADirectory) {
  EXPECT_THROW((void)(detail::generic_fs_list_recursive(fs_, "test", Uri((dir_ / "a.parquet").string()))),
               StorageError);
}

TEST_F(GenericFsObjectStoreTest, ListRecursiveThrowsOnMissingPath) {
  EXPECT_THROW((void)(detail::generic_fs_list_recursive(fs_, "test", Uri((dir_ / "missing_dir").string()))),
               StorageError);
}

TEST_F(GenericFsObjectStoreTest, ListRecursiveThrowsWhenTreeHasNoParquetFiles) {
  const fs::path empty_of_parquet = dir_ / "no_parquet_tree";
  fs::create_directories(empty_of_parquet / "sub");
  write_file(empty_of_parquet / "sub" / "readme.txt", "not parquet");
  EXPECT_THROW((void)(detail::generic_fs_list_recursive(fs_, "test", Uri(empty_of_parquet.string()))),
               StorageError);
}

TEST_F(GenericFsObjectStoreTest, OpenReadsBackRealFileContent) {
  const std::unique_ptr<RandomAccessObject> object =
      detail::generic_fs_open(fs_, "test", Uri((dir_ / "a.parquet").string()));
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->size(), 3u);
  ASSERT_NE(object->as_arrow_file(), nullptr);
  const arrow::Result<std::shared_ptr<arrow::Buffer>> read_result =
      object->as_arrow_file()->ReadAt(0, static_cast<std::int64_t>(object->size()));
  ASSERT_TRUE(read_result.ok()) << read_result.status().ToString();
  EXPECT_EQ(std::string(reinterpret_cast<const char*>((*read_result)->data()),
                        static_cast<std::size_t>((*read_result)->size())),
            "aaa");
}

TEST_F(GenericFsObjectStoreTest, OpenWrapsFailureIntoStorageError) {
  EXPECT_THROW((void)(detail::generic_fs_open(fs_, "test", Uri((dir_ / "missing.parquet").string()))),
               StorageError);
}

}  // namespace
}  // namespace kernellake
