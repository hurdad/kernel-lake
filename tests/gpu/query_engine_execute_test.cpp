// End-to-end test for QueryEngine::execute() against real Parquet files,
// exercising the exact MVP query shape from the spec: parse -> bind ->
// logical plan -> optimize -> physical plan -> Parquet pruning -> GPU
// filter -> GPU grouped aggregation -> Arrow result.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <map>

#include "kernellake/api/query_engine.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

// Days since the Unix epoch, matching Arrow/cudf's date32 representation.
constexpr std::int32_t kDec30_2025 = 20452;
constexpr std::int32_t kDec31_2025 = 20453;
constexpr std::int32_t kJan01_2026 = 20454;
constexpr std::int32_t kJan02_2026 = 20455;
constexpr std::int32_t kJan03_2026 = 20456;

class QueryEngineExecuteTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_engine_execute_test_" +
                     std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();

    arrow::StringBuilder region_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::Date32Builder date_builder;

    // Two rows per region fall before the WHERE cutoff and must be
    // excluded; two rows per region fall on/after it and must be summed.
    const std::vector<std::string> regions = {"A", "A", "A", "B", "B", "B"};
    const std::vector<double> amounts = {10.0, 20.0, 5.0, 100.0, 7.0, 3.0};
    const std::vector<std::int32_t> dates = {kDec30_2025, kJan01_2026, kJan02_2026,
                                              kDec31_2025, kJan01_2026, kJan03_2026};
    for (std::size_t i = 0; i < regions.size(); ++i) {
      ASSERT_TRUE(region_builder.Append(regions[i]).ok());
      ASSERT_TRUE(amount_builder.Append(amounts[i]).ok());
      ASSERT_TRUE(date_builder.Append(dates[i]).ok());
    }
    std::shared_ptr<arrow::Array> region_array, amount_array, date_array;
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    ASSERT_TRUE(date_builder.Finish(&date_array).ok());

    const auto schema = arrow::schema({arrow::field("region", arrow::utf8(), false),
                                        arrow::field("amount", arrow::float64(), false),
                                        arrow::field("event_date", arrow::date32(), false)});
    const auto table = arrow::Table::Make(schema, {region_array, amount_array, date_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/3);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
  QueryEngine engine_{default_config()};
};

TEST_F(QueryEngineExecuteTest, FilterAndGroupedAggregateMatchesExpectedTotals) {
  const QueryResult result =
      engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                       "') WHERE event_date >= DATE '2026-01-01' GROUP BY region");

  ASSERT_EQ(result.rows_returned, 2);
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 2);

  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(total_column, nullptr);

  std::map<std::string, double> totals_by_region;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    totals_by_region[region_column->GetString(i)] = total_column->Value(i);
  }
  ASSERT_EQ(totals_by_region.size(), 2u);
  EXPECT_DOUBLE_EQ(totals_by_region.at("A"), 25.0);  // 20.0 + 5.0
  EXPECT_DOUBLE_EQ(totals_by_region.at("B"), 10.0);  // 7.0 + 3.0

  EXPECT_TRUE(result.elapsed_wall_seconds.has_value());
  EXPECT_TRUE(result.peak_gpu_memory_bytes.has_value());
}

TEST_F(QueryEngineExecuteTest, ScalarAggregateWithNoGroupByMatchesExpectedTotal) {
  const QueryResult result =
      engine_.execute("SELECT SUM(amount) AS total FROM read_parquet('" + path_ + "')");

  ASSERT_EQ(result.rows_returned, 1);
  ASSERT_EQ(result.batches.size(), 1u);
  const auto total_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
  ASSERT_NE(total_column, nullptr);
  EXPECT_DOUBLE_EQ(total_column->Value(0), 145.0);  // 10+20+5+100+7+3
}

TEST_F(QueryEngineExecuteTest, PlainProjectionReturnsAllRows) {
  const QueryResult result = engine_.execute("SELECT region FROM read_parquet('" + path_ + "')");
  EXPECT_EQ(result.rows_returned, 6);
}

}  // namespace
}  // namespace kernellake
