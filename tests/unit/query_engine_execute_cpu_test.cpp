// End-to-end test for the Apache Arrow Acero CPU execution backend
// (QueryEngine::execute_cpu(), reached via execute(sql) when
// engine.backend == "cpu"). Needs no GPU, so it belongs in the CI-eligible
// unit suite -- a first for anything past physical planning. Reuses the
// exact same fixture data/expected values as
// tests/gpu/query_engine_execute_test.cpp so the two backends' results can
// be compared directly for the query shapes both support.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <filesystem>
#include <map>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

constexpr std::int32_t kDec30_2025 = 20452;
constexpr std::int32_t kDec31_2025 = 20453;
constexpr std::int32_t kJan01_2026 = 20454;
constexpr std::int32_t kJan02_2026 = 20455;
constexpr std::int32_t kJan03_2026 = 20456;

EngineConfig cpu_backend_config() {
  EngineConfig config = default_config();
  config.engine.backend = "cpu";
  return config;
}

class QueryEngineExecuteCpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_engine_execute_cpu_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();

    arrow::StringBuilder region_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::Date32Builder date_builder;

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

    regions_path_ = (dir_ / "regions.parquet").string();
    arrow::StringBuilder region_key_builder;
    arrow::StringBuilder region_name_builder;
    ASSERT_TRUE(region_key_builder.Append("A").ok());
    ASSERT_TRUE(region_name_builder.Append("Alpha").ok());
    ASSERT_TRUE(region_key_builder.Append("B").ok());
    ASSERT_TRUE(region_name_builder.Append("Beta").ok());
    std::shared_ptr<arrow::Array> region_key_array, region_name_array;
    ASSERT_TRUE(region_key_builder.Finish(&region_key_array).ok());
    ASSERT_TRUE(region_name_builder.Finish(&region_name_array).ok());
    const auto regions_schema = arrow::schema(
        {arrow::field("region", arrow::utf8(), false), arrow::field("region_name", arrow::utf8(), false)});
    const auto regions_table = arrow::Table::Make(regions_schema, {region_key_array, region_name_array});
    auto regions_sink = arrow::io::FileOutputStream::Open(regions_path_).ValueOrDie();
    const arrow::Status regions_status = parquet::arrow::WriteTable(
        *regions_table, arrow::default_memory_pool(), regions_sink, /*chunk_size=*/2);
    ASSERT_TRUE(regions_status.ok()) << regions_status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
  std::string regions_path_;
  QueryEngine engine_{cpu_backend_config()};
};

TEST_F(QueryEngineExecuteCpuTest, FilterAndGroupedAggregateMatchesExpectedTotals) {
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
}

TEST_F(QueryEngineExecuteCpuTest, ScalarAggregateWithNoGroupByMatchesExpectedTotal) {
  const QueryResult result =
      engine_.execute("SELECT SUM(amount) AS total FROM read_parquet('" + path_ + "')");

  ASSERT_EQ(result.rows_returned, 1);
  ASSERT_EQ(result.batches.size(), 1u);
  const auto total_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
  ASSERT_NE(total_column, nullptr);
  EXPECT_DOUBLE_EQ(total_column->Value(0), 145.0);  // 10+20+5+100+7+3
}

// Regression test: Acero's AggregateNodeOptions can only target an
// already-existing column by FieldRef, not evaluate an expression itself
// -- SUM/AVG/etc. over a computed expression (not a plain column
// reference) used to throw "aggregating by a computed expression is not
// yet supported by the CPU execution backend" unconditionally. This is
// exactly the shape TPC-H Q1/Q6 both need (SUM(l_extendedprice *
// l_discount), SUM(l_extendedprice * (1 - l_discount)), ...), found by
// running the real tools/benchmark_three_way.py three-way comparison.
TEST_F(QueryEngineExecuteCpuTest, GroupedAggregateOverComputedExpressionMatchesExpectedTotals) {
  const QueryResult result = engine_.execute("SELECT region, SUM(amount * 2) AS total FROM read_parquet('" +
                                             path_ + "') GROUP BY region");

  ASSERT_EQ(result.rows_returned, 2);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(total_column, nullptr);

  std::map<std::string, double> totals_by_region;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    totals_by_region[region_column->GetString(i)] = total_column->Value(i);
  }
  ASSERT_EQ(totals_by_region.size(), 2u);
  EXPECT_DOUBLE_EQ(totals_by_region.at("A"), 70.0);   // (10.0 + 20.0 + 5.0) * 2
  EXPECT_DOUBLE_EQ(totals_by_region.at("B"), 220.0);  // (100.0 + 7.0 + 3.0) * 2
}

TEST_F(QueryEngineExecuteCpuTest, ScalarAggregateOverComputedExpressionMatchesExpectedTotal) {
  const QueryResult result =
      engine_.execute("SELECT SUM(amount * 2) AS total FROM read_parquet('" + path_ + "')");

  ASSERT_EQ(result.rows_returned, 1);
  const auto total_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
  ASSERT_NE(total_column, nullptr);
  EXPECT_DOUBLE_EQ(total_column->Value(0), 290.0);  // (10+20+5+100+7+3) * 2
}

TEST_F(QueryEngineExecuteCpuTest, PlainProjectionReturnsAllRows) {
  const QueryResult result = engine_.execute("SELECT region FROM read_parquet('" + path_ + "')");
  EXPECT_EQ(result.rows_returned, 6);
}

TEST_F(QueryEngineExecuteCpuTest, AggregateOrderByProducesDescendingTotals) {
  const QueryResult result = engine_.execute("SELECT SUM(amount) AS total, region FROM read_parquet('" +
                                             path_ + "') GROUP BY region ORDER BY total DESC");

  ASSERT_EQ(result.rows_returned, 2);
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  ASSERT_NE(total_column, nullptr);
  ASSERT_NE(region_column, nullptr);

  EXPECT_EQ(region_column->GetString(0), "B");
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
  EXPECT_EQ(region_column->GetString(1), "A");
  EXPECT_DOUBLE_EQ(total_column->Value(1), 35.0);
}

TEST_F(QueryEngineExecuteCpuTest, CountStarGroupedAndScalarMatchExpectedRowCounts) {
  // Regression test: Arrow Compute's "count"/"hash_count" require a real
  // value-column argument (arity 1/2) and reject COUNT(*)'s empty target,
  // which needs the dedicated arity-0/1 "count_all"/"hash_count_all"
  // functions instead -- caught by exercising this for real against a
  // Parquet file, not just by reading the Arrow Compute headers.
  const QueryResult grouped = engine_.execute("SELECT region, COUNT(*) AS n FROM read_parquet('" + path_ +
                                              "') GROUP BY region ORDER BY region");
  ASSERT_EQ(grouped.batches.size(), 1u);
  const auto grouped_n =
      std::static_pointer_cast<arrow::Int64Array>(grouped.batches.front()->GetColumnByName("n"));
  ASSERT_NE(grouped_n, nullptr);
  ASSERT_EQ(grouped_n->length(), 2);
  EXPECT_EQ(grouped_n->Value(0), 3);  // region A
  EXPECT_EQ(grouped_n->Value(1), 3);  // region B

  const QueryResult scalar = engine_.execute("SELECT COUNT(*) AS n FROM read_parquet('" + path_ + "')");
  ASSERT_EQ(scalar.batches.size(), 1u);
  const auto scalar_n =
      std::static_pointer_cast<arrow::Int64Array>(scalar.batches.front()->GetColumnByName("n"));
  ASSERT_NE(scalar_n, nullptr);
  EXPECT_EQ(scalar_n->Value(0), 6);
}

TEST_F(QueryEngineExecuteCpuTest, LimitTruncatesToRequestedRowCount) {
  const QueryResult result =
      engine_.execute("SELECT region FROM read_parquet('" + path_ + "') ORDER BY amount DESC LIMIT 2");
  ASSERT_EQ(result.batches.size(), 1u);
  EXPECT_EQ(result.batches.front()->num_rows(), 2);
}

TEST_F(QueryEngineExecuteCpuTest, PopulatesCpuTimingAndScanStats) {
  const QueryResult result = engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" +
                                             path_ + "') GROUP BY region");
  ASSERT_TRUE(result.elapsed_wall_seconds.has_value());
  ASSERT_TRUE(result.metadata_inspection_seconds.has_value());
  ASSERT_TRUE(result.cpu_execution_seconds.has_value());
  EXPECT_GE(*result.metadata_inspection_seconds, 0.0);
  EXPECT_GE(*result.cpu_execution_seconds, 0.0);
  EXPECT_GE(*result.elapsed_wall_seconds, *result.cpu_execution_seconds);

  // Not tracked by this backend -- must stay a documented null, not an
  // invented zero (see query_engine_execute_cpu.cpp).
  EXPECT_FALSE(result.gpu_execution_seconds.has_value());
  EXPECT_FALSE(result.device_to_host_seconds.has_value());
  EXPECT_FALSE(result.parquet_decoding_seconds.has_value());
  EXPECT_FALSE(result.peak_gpu_memory_bytes.has_value());

  ASSERT_TRUE(result.files_considered.has_value());
  ASSERT_TRUE(result.row_groups_considered.has_value());
  EXPECT_EQ(*result.files_scanned, 1);
}

TEST_F(QueryEngineExecuteCpuTest, LikeIsNotYetSupportedByCpuBackend) {
  EXPECT_THROW(
      (void)(engine_.execute("SELECT region FROM read_parquet('" + path_ + "') WHERE region LIKE 'A%'")),
      ExecutionError);
}

// Regression test: the CPU execution backend used to reject CASE
// expressions everywhere ("unrecognized expression type in CPU expression
// compiler (LIKE/IN/CASE/DECIMAL are not yet supported...)"), even though
// the GPU backend already supported CASE inside a grouped aggregate
// argument (just not inside a *scalar* one -- a separate, still-open GPU
// gap -- or inside WHERE). Fixed by mapping CaseExpression to Arrow
// Compute's own "case_when" kernel (a struct of per-branch boolean
// conditions built via "make_struct", followed by one value expression per
// condition and an optional trailing ELSE value) in
// compile_expression_cpu() -- the one function shared by every context
// (WHERE, SELECT list, grouped and scalar aggregate arguments), so all of
// them work as soon as this one function does, unlike the GPU backend's
// split between an eager cudf-materializing path and a separate lazy
// cudf::ast path.
TEST_F(QueryEngineExecuteCpuTest, CaseInGroupedAggregateWhereAndScalarAggregateMatchesExpectedTotals) {
  const QueryResult grouped = engine_.execute(
      "SELECT region, SUM(CASE WHEN amount > 15 THEN 1 ELSE 0 END) AS high_count FROM read_parquet('" +
      path_ + "') GROUP BY region");
  ASSERT_EQ(grouped.rows_returned, 2);
  const std::shared_ptr<arrow::RecordBatch>& grouped_batch = grouped.batches.front();
  const auto region_column =
      std::static_pointer_cast<arrow::StringArray>(grouped_batch->GetColumnByName("region"));
  const auto high_count_column =
      std::static_pointer_cast<arrow::Int64Array>(grouped_batch->GetColumnByName("high_count"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(high_count_column, nullptr);
  std::map<std::string, std::int64_t> high_count_by_region;
  for (std::int64_t i = 0; i < grouped_batch->num_rows(); ++i) {
    high_count_by_region[region_column->GetString(i)] = high_count_column->Value(i);
  }
  ASSERT_EQ(high_count_by_region.size(), 2u);
  EXPECT_EQ(high_count_by_region.at("A"), 1);  // only 20.0 > 15
  EXPECT_EQ(high_count_by_region.at("B"), 1);  // only 100.0 > 15

  const QueryResult scalar = engine_.execute(
      "SELECT SUM(CASE WHEN amount > 15 THEN 1 ELSE 0 END) AS high_count FROM read_parquet('" + path_ + "')");
  ASSERT_EQ(scalar.rows_returned, 1);
  const auto scalar_column =
      std::static_pointer_cast<arrow::Int64Array>(scalar.batches.front()->GetColumnByName("high_count"));
  ASSERT_NE(scalar_column, nullptr);
  EXPECT_EQ(scalar_column->Value(0), 2);  // 20.0 and 100.0

  const QueryResult where_result = engine_.execute("SELECT COUNT(*) AS n FROM read_parquet('" + path_ +
                                                   "') WHERE (CASE WHEN amount > 15 THEN 1 ELSE 0 END) = 1");
  ASSERT_EQ(where_result.rows_returned, 1);
  const auto n_column =
      std::static_pointer_cast<arrow::Int64Array>(where_result.batches.front()->GetColumnByName("n"));
  ASSERT_NE(n_column, nullptr);
  EXPECT_EQ(n_column->Value(0), 2);
}

// Regression test: the CPU execution backend used to reject every
// HashJoinNode outright ("physical plan node 'HashJoin' is not yet
// supported by the CPU execution backend"), even though the parser/binder
// already accepted two-table INNER JOIN ... ON queries and the GPU backend
// already executed them correctly -- a real asymmetry found while scoping
// which TPC-H queries beyond Q1/Q6 could run through
// tools/benchmark_three_way.py. Fixed by translating HashJoinNode to
// Acero's own "hashjoin" node (arrow::acero::HashJoinNodeOptions), which
// implements the same two-table INNER equi-join natively.
TEST_F(QueryEngineExecuteCpuTest, TwoTableInnerJoinMatchesExpectedTotals) {
  const QueryResult result =
      engine_.execute("SELECT r.region_name, SUM(s.amount) AS total FROM read_parquet('" + path_ +
                      "') AS s JOIN read_parquet('" + regions_path_ +
                      "') AS r ON s.region = r.region GROUP BY r.region_name");

  ASSERT_EQ(result.rows_returned, 2);
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 2);

  const auto name_column =
      std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region_name"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(name_column, nullptr);
  ASSERT_NE(total_column, nullptr);

  std::map<std::string, double> totals_by_name;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    totals_by_name[name_column->GetString(i)] = total_column->Value(i);
  }
  ASSERT_EQ(totals_by_name.size(), 2u);
  EXPECT_DOUBLE_EQ(totals_by_name.at("Alpha"), 35.0);  // 10+20+5
  EXPECT_DOUBLE_EQ(totals_by_name.at("Beta"), 110.0);  // 100+7+3
}

}  // namespace
}  // namespace kernellake
