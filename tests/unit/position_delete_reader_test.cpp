#include "kernellake/iceberg/position_delete_reader.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <string>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake::iceberg {
namespace {

namespace fs = std::filesystem;

class PositionDeleteReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_position_delete_reader_test_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                    ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  // Writes a real "file_path: string, pos: long" Parquet file -- the
  // Iceberg spec's Position Delete Files schema -- with one row per
  // (path, pos) pair given.
  void write_position_delete_file(const fs::path& path,
                                  const std::vector<std::pair<std::string, int64_t>>& rows) {
    arrow::StringBuilder file_path_builder;
    arrow::Int64Builder pos_builder;
    for (const auto& [file_path, pos] : rows) {
      ASSERT_TRUE(file_path_builder.Append(file_path).ok());
      ASSERT_TRUE(pos_builder.Append(pos).ok());
    }
    std::shared_ptr<arrow::Array> file_path_array;
    std::shared_ptr<arrow::Array> pos_array;
    ASSERT_TRUE(file_path_builder.Finish(&file_path_array).ok());
    ASSERT_TRUE(pos_builder.Finish(&pos_array).ok());
    const auto schema = arrow::schema(
        {arrow::field("file_path", arrow::utf8(), false), arrow::field("pos", arrow::int64(), false)});
    const auto table = arrow::Table::Make(schema, {file_path_array, pos_array});
    auto sink_result = arrow::io::FileOutputStream::Open(path.string());
    ASSERT_TRUE(sink_result.ok()) << sink_result.status().ToString();
    const arrow::Status write_status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                                                  *sink_result, /*chunk_size=*/rows.size());
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();
  }

  fs::path dir_;
  LocalObjectStore store_;
};

TEST_F(PositionDeleteReaderTest, CountsDeletedPositionsPerReferencedFile) {
  const fs::path delete_file = dir_ / "delete-0.parquet";
  write_position_delete_file(delete_file, {
                                              {"s3://warehouse/db/orders/data-0.parquet", 0},
                                              {"s3://warehouse/db/orders/data-0.parquet", 1},
                                              {"s3://warehouse/db/orders/data-0.parquet", 5},
                                              {"s3://warehouse/db/orders/data-1.parquet", 2},
                                          });

  const std::unordered_map<std::string, std::int64_t> counts =
      read_position_delete_counts(store_, Uri(delete_file.string()));

  ASSERT_EQ(counts.size(), 2u);
  EXPECT_EQ(counts.at("s3://warehouse/db/orders/data-0.parquet"), 3);
  EXPECT_EQ(counts.at("s3://warehouse/db/orders/data-1.parquet"), 1);
}

TEST_F(PositionDeleteReaderTest, EmptyDeleteFileProducesEmptyCounts) {
  const fs::path delete_file = dir_ / "delete-0.parquet";
  write_position_delete_file(delete_file, {});

  const std::unordered_map<std::string, std::int64_t> counts =
      read_position_delete_counts(store_, Uri(delete_file.string()));
  EXPECT_TRUE(counts.empty());
}

TEST_F(PositionDeleteReaderTest, ThrowsWhenRequiredColumnsAreMissing) {
  // A Parquet file that isn't shaped like a position-delete file at all
  // (e.g. an equality-delete file with arbitrary user columns) -- must
  // fail loudly rather than silently return an empty/wrong count.
  arrow::Int64Builder id_builder;
  ASSERT_TRUE(id_builder.Append(1).ok());
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(schema, {id_array});
  const fs::path path = dir_ / "not-a-position-delete-file.parquet";
  auto sink_result = arrow::io::FileOutputStream::Open(path.string());
  ASSERT_TRUE(sink_result.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, 1).ok());

  EXPECT_THROW((void)(read_position_delete_counts(store_, Uri(path.string()))), StorageError);
}

TEST_F(PositionDeleteReaderTest, ThrowsWhenFileDoesNotExist) {
  EXPECT_THROW((void)(read_position_delete_counts(store_, Uri((dir_ / "does-not-exist.parquet").string()))),
               StorageError);
}

}  // namespace
}  // namespace kernellake::iceberg
