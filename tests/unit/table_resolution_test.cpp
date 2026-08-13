#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <fmt/format.h>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/table_resolution.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class TableResolutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_table_resolution_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  static void write_file(const fs::path& path, int64_t id_start, int64_t count) {
    fs::create_directories(path.parent_path());
    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    for (int64_t i = 0; i < count; ++i) {
      ASSERT_TRUE(id_builder.Append(id_start + i).ok());
      ASSERT_TRUE(amount_builder.Append(static_cast<double>(id_start + i)).ok());
    }
    std::shared_ptr<arrow::Array> id_array, amount_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    const auto schema = arrow::schema(
        {arrow::field("id", arrow::int64(), false), arrow::field("amount", arrow::float64(), false)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array});
    auto sink_result = arrow::io::FileOutputStream::Open(path.string());
    ASSERT_TRUE(sink_result.ok()) << sink_result.status().ToString();
    const arrow::Status write_status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, /*chunk_size=*/count);
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();
  }

  fs::path dir_;
  LocalObjectStore store_;
};

TEST_F(TableResolutionTest, PlainNonPartitionedDirectoryHasZeroPartitionColumns) {
  write_file(dir_ / "part-0.parquet", 0, 5);
  write_file(dir_ / "part-1.parquet", 5, 5);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  EXPECT_TRUE(resolved.partition_columns.empty());
  ASSERT_EQ(resolved.schema.field_count(), 2u);
  EXPECT_EQ(resolved.schema.field(0).name, "id");
  EXPECT_EQ(resolved.schema.field(1).name, "amount");
  ASSERT_EQ(resolved.files.size(), 2u);
  EXPECT_TRUE(resolved.files[0].partition_values.empty());
}

// Regression test for the batched-concurrency fix in resolve_table():
// metadata inspection used to launch one std::async per file with no cap
// (risking RLIMIT_NPROC exhaustion for a table with many small files);
// batching that into fixed-size chunks (kMaxConcurrentFileInspections = 64
// in table_resolution.cpp) must still preserve exact per-file ordering and
// correctness across a batch boundary -- this table has more files than
// one batch holds, so metadata[i]/files[i] lockstep (assumed by the
// partition-segment extraction right after) actually gets exercised across
// two batches, not just within a single one.
TEST_F(TableResolutionTest, ResolvesMoreFilesThanOneInspectionBatchHolds) {
  constexpr int kFileCount = 130;  // > kMaxConcurrentFileInspections (64)
  for (int i = 0; i < kFileCount; ++i) {
    write_file(dir_ / fmt::format("part-{:03d}.parquet", i), i, 1);
  }

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  ASSERT_EQ(resolved.files.size(), static_cast<std::size_t>(kFileCount));
  std::int64_t total_rows = 0;
  for (const ResolvedFile& file : resolved.files) {
    total_rows += file.metadata.row_count;
  }
  EXPECT_EQ(total_rows, kFileCount);
}

TEST_F(TableResolutionTest, DetectsHivePartitionColumnWithStringType) {
  write_file(dir_ / "region=A" / "part-0.parquet", 0, 3);
  write_file(dir_ / "region=B" / "part-0.parquet", 3, 3);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  ASSERT_EQ(resolved.partition_columns.size(), 1u);
  EXPECT_EQ(resolved.partition_columns[0].name, "region");
  EXPECT_EQ(resolved.partition_columns[0].type.id, TypeId::String);
  ASSERT_EQ(resolved.schema.field_count(), 3u);
  EXPECT_EQ(resolved.schema.field(2).name, "region");
  EXPECT_EQ(resolved.schema.field(2).type.id, TypeId::String);

  ASSERT_EQ(resolved.files.size(), 2u);
  for (const ResolvedFile& file : resolved.files) {
    ASSERT_EQ(file.partition_values.size(), 1u);
    const std::string& region = std::get<std::string>(file.partition_values[0]);
    EXPECT_TRUE(region == "A" || region == "B");
  }
}

TEST_F(TableResolutionTest, InfersInt64TypeForNumericPartitionValues) {
  write_file(dir_ / "year=2024" / "part-0.parquet", 0, 2);
  write_file(dir_ / "year=2025" / "part-0.parquet", 2, 2);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  ASSERT_EQ(resolved.partition_columns.size(), 1u);
  EXPECT_EQ(resolved.partition_columns[0].name, "year");
  EXPECT_EQ(resolved.partition_columns[0].type.id, TypeId::Int64);

  bool saw_2024 = false;
  bool saw_2025 = false;
  for (const ResolvedFile& file : resolved.files) {
    const std::int64_t year = std::get<std::int64_t>(file.partition_values[0]);
    if (year == 2024) saw_2024 = true;
    if (year == 2025) saw_2025 = true;
  }
  EXPECT_TRUE(saw_2024);
  EXPECT_TRUE(saw_2025);
}

TEST_F(TableResolutionTest, InfersDate32TypeForIsoDatePartitionValues) {
  write_file(dir_ / "event_date=2026-01-01" / "part-0.parquet", 0, 2);
  write_file(dir_ / "event_date=2026-01-02" / "part-0.parquet", 2, 2);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  ASSERT_EQ(resolved.partition_columns.size(), 1u);
  EXPECT_EQ(resolved.partition_columns[0].type.id, TypeId::Date32);
  // 2026-01-01 is a valid, in-range date32 (positive day count since epoch).
  for (const ResolvedFile& file : resolved.files) {
    EXPECT_GT(std::get<std::int64_t>(file.partition_values[0]), 0);
  }
}

TEST_F(TableResolutionTest, MultiLevelPartitioningResolvesInRootToLeafOrder) {
  write_file(dir_ / "region=A" / "year=2024" / "part-0.parquet", 0, 2);
  write_file(dir_ / "region=A" / "year=2025" / "part-0.parquet", 2, 2);
  write_file(dir_ / "region=B" / "year=2024" / "part-0.parquet", 4, 2);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  ASSERT_EQ(resolved.partition_columns.size(), 2u);
  EXPECT_EQ(resolved.partition_columns[0].name, "region");
  EXPECT_EQ(resolved.partition_columns[1].name, "year");
  ASSERT_EQ(resolved.schema.field_count(), 4u);
  EXPECT_EQ(resolved.schema.field(2).name, "region");
  EXPECT_EQ(resolved.schema.field(3).name, "year");
}

TEST_F(TableResolutionTest, RejectsInconsistentPartitioningAcrossFiles) {
  write_file(dir_ / "region=A" / "part-0.parquet", 0, 2);
  write_file(dir_ / "part-1.parquet", 2, 2);  // not partitioned at all -- inconsistent with the above.

  EXPECT_THROW((void)resolve_table(store_, {dir_.string()}), StorageError);
}

TEST_F(TableResolutionTest, RejectsInconsistentPartitionKeysAcrossFiles) {
  write_file(dir_ / "region=A" / "part-0.parquet", 0, 2);
  write_file(dir_ / "country=US" / "part-0.parquet", 2, 2);  // different key at the same level.

  EXPECT_THROW((void)resolve_table(store_, {dir_.string()}), StorageError);
}

TEST_F(TableResolutionTest, RejectsPartitionColumnCollidingWithExistingColumnName) {
  write_file(dir_ / "id=A" / "part-0.parquet", 0, 2);  // "id" already exists as a physical column.

  EXPECT_THROW((void)resolve_table(store_, {dir_.string()}), StorageError);
}

// Regression test for the real bug fixed in parse_partition_value(): a
// partition directory name that's digit-shaped enough to pass
// looks_like_int64() (only a shape check, no magnitude bound) but too large
// for std::int64_t used to make std::stoll() throw a raw, uncaught
// std::out_of_range instead of this file's own StorageError -- inconsistent
// with every other rejection path here. EXPECT_THROW's exact-type match
// means this test would itself fail (a std::out_of_range doesn't convert to
// StorageError) if the fix ever regressed.
TEST_F(TableResolutionTest, RejectsPartitionValueTooLargeForInt64) {
  write_file(dir_ / "id=999999999999999999999999" / "part-0.parquet", 0, 2);  // 24 digits, overflows int64

  EXPECT_THROW((void)resolve_table(store_, {dir_.string()}), StorageError);
}

// infer_partition_type()'s date32 path: "2026-13-40" is digit-shaped enough
// to pass looks_like_iso_date()'s own shape-only check (10 chars, dashes at
// positions 4 and 7), but month 13 / day 40 isn't a real calendar date --
// parse_iso_date() throws SqlError for it, so infer_partition_type() must
// fall back to string_type() rather than let that exception propagate out of
// resolve_table() (or worse, silently store a garbage date32 value).
TEST_F(TableResolutionTest, FallsBackToStringTypeForDigitShapedButInvalidCalendarDate) {
  write_file(dir_ / "event_date=2026-13-40" / "part-0.parquet", 0, 2);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  ASSERT_EQ(resolved.partition_columns.size(), 1u);
  EXPECT_EQ(resolved.partition_columns[0].type.id, TypeId::String);
  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_EQ(std::get<std::string>(resolved.files[0].partition_values[0]), "2026-13-40");
}

// parse_key_value_segment()'s "key starts with a digit" rejection: a
// directory segment shaped like "1=x" isn't key=value-shaped at all by this
// file's own rules, so extract_partition_segments() stops scanning right
// there and this file is treated as having zero partition levels -- not as
// a partition column literally named "1".
TEST_F(TableResolutionTest, IgnoresPartitionSegmentWhoseKeyStartsWithADigit) {
  write_file(dir_ / "1=x" / "part-0.parquet", 0, 2);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  EXPECT_TRUE(resolved.partition_columns.empty());
  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_TRUE(resolved.files[0].partition_values.empty());
}

// parse_key_value_segment()'s "key has a non-alnum/underscore character"
// rejection: "a.b=x" fails the same shape check for the same reason.
TEST_F(TableResolutionTest, IgnoresPartitionSegmentWhoseKeyHasANonAlnumCharacter) {
  write_file(dir_ / "a.b=x" / "part-0.parquet", 0, 2);

  const ResolvedTable resolved = resolve_table(store_, {dir_.string()});
  EXPECT_TRUE(resolved.partition_columns.empty());
  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_TRUE(resolved.files[0].partition_values.empty());
}

}  // namespace
}  // namespace kernellake
