#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/io/parquet_pruning.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class ParquetMetadataTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_parquet_test_" +
                     std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();
    write_two_row_group_file(path_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  // Row group 0: id 0-4, region "A", amount 0.0-4.0.
  // Row group 1: id 5-9, region "B", amount 5.0-9.0.
  static void write_two_row_group_file(const std::string& path) {
    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::StringBuilder region_builder;
    for (int64_t i = 0; i < 10; ++i) {
      ASSERT_OK(id_builder.Append(i));
      ASSERT_OK(amount_builder.Append(static_cast<double>(i)));
      ASSERT_OK(region_builder.Append(i < 5 ? "A" : "B"));
    }
    std::shared_ptr<arrow::Array> id_array, amount_array, region_array;
    ASSERT_OK(id_builder.Finish(&id_array));
    ASSERT_OK(amount_builder.Finish(&amount_array));
    ASSERT_OK(region_builder.Finish(&region_array));

    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false),
                                        arrow::field("amount", arrow::float64(), false),
                                        arrow::field("region", arrow::utf8(), false)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array, region_array});

    auto sink_result = arrow::io::FileOutputStream::Open(path);
    ASSERT_TRUE(sink_result.ok()) << sink_result.status().ToString();
    const arrow::Status write_status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result,
                                    /*chunk_size=*/5);
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();
  }

  static void ASSERT_OK(const arrow::Status& status) { ASSERT_TRUE(status.ok()) << status.ToString(); }

  fs::path dir_;
  std::string path_;
  LocalObjectStore store_;
};

TEST_F(ParquetMetadataTest, ReadsSchemaAndRowCount) {
  const FileMetadata meta = inspect_parquet_file(store_, Uri(path_));
  EXPECT_EQ(meta.row_count, 10);
  ASSERT_EQ(meta.schema.field_count(), 3u);
  EXPECT_EQ(meta.schema.field(0).name, "id");
  EXPECT_EQ(meta.schema.field(0).type.id, TypeId::Int64);
  EXPECT_EQ(meta.schema.field(2).name, "region");
  EXPECT_EQ(meta.schema.field(2).type.id, TypeId::String);
}

TEST_F(ParquetMetadataTest, ReadsTwoRowGroups) {
  const FileMetadata meta = inspect_parquet_file(store_, Uri(path_));
  ASSERT_EQ(meta.row_groups.size(), 2u);
  EXPECT_EQ(meta.row_groups[0].row_count, 5);
  EXPECT_EQ(meta.row_groups[1].row_count, 5);
}

TEST_F(ParquetMetadataTest, ReadsMinMaxStatisticsPerRowGroup) {
  const FileMetadata meta = inspect_parquet_file(store_, Uri(path_));
  const ColumnStatistics& id_stats_rg0 = meta.row_groups[0].column_statistics.at("id");
  ASSERT_TRUE(id_stats_rg0.has_min_max);
  EXPECT_EQ(std::get<std::int64_t>(id_stats_rg0.min_value), 0);
  EXPECT_EQ(std::get<std::int64_t>(id_stats_rg0.max_value), 4);

  const ColumnStatistics& id_stats_rg1 = meta.row_groups[1].column_statistics.at("id");
  EXPECT_EQ(std::get<std::int64_t>(id_stats_rg1.min_value), 5);
  EXPECT_EQ(std::get<std::int64_t>(id_stats_rg1.max_value), 9);

  const ColumnStatistics& region_stats_rg0 = meta.row_groups[0].column_statistics.at("region");
  ASSERT_TRUE(region_stats_rg0.has_min_max);
  EXPECT_EQ(std::get<std::string>(region_stats_rg0.min_value), "A");
  EXPECT_EQ(std::get<std::string>(region_stats_rg0.max_value), "A");
}

TEST_F(ParquetMetadataTest, ThrowsOnMissingFile) {
  EXPECT_THROW(inspect_parquet_file(store_, Uri((dir_ / "missing.parquet").string())),
               StorageError);
}

TEST_F(ParquetMetadataTest, ValidateSchemaCompatibilityAcceptsIdenticalSchemas) {
  const FileMetadata a = inspect_parquet_file(store_, Uri(path_));
  const FileMetadata b = inspect_parquet_file(store_, Uri(path_));
  EXPECT_NO_THROW(validate_schema_compatibility({a, b}));
}

TEST_F(ParquetMetadataTest, ValidateSchemaCompatibilityRejectsMismatch) {
  const std::string other_path = (dir_ / "other.parquet").string();
  arrow::Int64Builder id_builder;
  ASSERT_OK(id_builder.Append(1));
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_OK(id_builder.Finish(&id_array));
  const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(schema, {id_array});
  auto sink_result = arrow::io::FileOutputStream::Open(other_path);
  ASSERT_OK(sink_result.status());
  ASSERT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result));

  const FileMetadata a = inspect_parquet_file(store_, Uri(path_));
  const FileMetadata b = inspect_parquet_file(store_, Uri(other_path));
  EXPECT_THROW(validate_schema_compatibility({a, b}), StorageError);
}

TEST_F(ParquetMetadataTest, PruningSkipsRowGroupProvenByEquality) {
  const FileMetadata meta = inspect_parquet_file(store_, Uri(path_));
  auto literal = std::make_shared<LiteralExpression>(LiteralExpression::make_string("B"));
  const std::vector<PushablePredicate> predicates = {
      PushablePredicate{"region", BinaryOperator::Equal, literal}};

  const ScanDecision decision = evaluate_pruning(meta, predicates);
  ASSERT_EQ(decision.skipped_row_groups.size(), 1u);
  EXPECT_EQ(decision.skipped_row_groups[0], 0);  // row group 0 is all "A"
  ASSERT_EQ(decision.selected_row_groups.size(), 1u);
  EXPECT_EQ(decision.selected_row_groups[0], 1);
  EXPECT_FALSE(decision.reasons.empty());
}

TEST_F(ParquetMetadataTest, PruningSkipsRowGroupProvenByRange) {
  const FileMetadata meta = inspect_parquet_file(store_, Uri(path_));
  auto literal = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(5));
  const std::vector<PushablePredicate> predicates = {
      PushablePredicate{"id", BinaryOperator::Less, literal}};

  const ScanDecision decision = evaluate_pruning(meta, predicates);
  // id < 5: row group 0 (0-4) fully qualifies-or-not but cannot be skipped
  // (some rows may match); row group 1 (5-9) has min=5 >= 5, provably empty.
  ASSERT_EQ(decision.skipped_row_groups.size(), 1u);
  EXPECT_EQ(decision.skipped_row_groups[0], 1);
}

TEST_F(ParquetMetadataTest, PruningKeepsEverythingWhenNoPredicates) {
  const FileMetadata meta = inspect_parquet_file(store_, Uri(path_));
  const ScanDecision decision = evaluate_pruning(meta, {});
  EXPECT_EQ(decision.selected_row_groups.size(), 2u);
  EXPECT_TRUE(decision.skipped_row_groups.empty());
}

TEST_F(ParquetMetadataTest, PruningDoesNotSkipWhenColumnStatsMissingFromPredicate) {
  const FileMetadata meta = inspect_parquet_file(store_, Uri(path_));
  auto literal = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(100));
  const std::vector<PushablePredicate> predicates = {
      PushablePredicate{"nonexistent_column", BinaryOperator::Equal, literal}};
  const ScanDecision decision = evaluate_pruning(meta, predicates);
  EXPECT_EQ(decision.selected_row_groups.size(), 2u);
}

}  // namespace
}  // namespace kernellake
