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
#include <fstream>
#include <map>
#include <tuple>

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

    // A third table, for a 3-way-join test: joins onto regions.parquet via
    // region_name (not region -- a step further removed from sales.parquet
    // in the chain, so the test genuinely exercises a 3-source join, not
    // just two 2-source joins glued together).
    managers_path_ = (dir_ / "managers.parquet").string();
    arrow::StringBuilder manager_region_name_builder;
    arrow::StringBuilder manager_builder;
    ASSERT_TRUE(manager_region_name_builder.Append("Alpha").ok());
    ASSERT_TRUE(manager_builder.Append("Ann").ok());
    ASSERT_TRUE(manager_region_name_builder.Append("Beta").ok());
    ASSERT_TRUE(manager_builder.Append("Bo").ok());
    std::shared_ptr<arrow::Array> manager_region_name_array, manager_array;
    ASSERT_TRUE(manager_region_name_builder.Finish(&manager_region_name_array).ok());
    ASSERT_TRUE(manager_builder.Finish(&manager_array).ok());
    const auto managers_schema = arrow::schema(
        {arrow::field("region_name", arrow::utf8(), false), arrow::field("manager", arrow::utf8(), false)});
    const auto managers_table =
        arrow::Table::Make(managers_schema, {manager_region_name_array, manager_array});
    auto managers_sink = arrow::io::FileOutputStream::Open(managers_path_).ValueOrDie();
    const arrow::Status managers_status = parquet::arrow::WriteTable(
        *managers_table, arrow::default_memory_pool(), managers_sink, /*chunk_size=*/2);
    ASSERT_TRUE(managers_status.ok()) << managers_status.ToString();

    // Two more tables, both with a column literally named "x" (not the
    // join key) holding *different* values on each side -- regression
    // fixture for the JOIN bare-name-collision bug cluster (aggregate
    // dedup by to_string(), remap_columns() resolving by name against the
    // combined schema, GROUP BY's ungrouped-column check ignoring the
    // table qualifier). Unlike "region" above (shared by construction
    // *because* it's the join key, so both sides always agree on its value
    // for a matched row), "x" differs across sides specifically so a
    // collision produces a visibly wrong, distinguishable result instead
    // of accidentally still looking correct.
    left_dup_path_ = (dir_ / "left_dup.parquet").string();
    arrow::Int64Builder left_id_builder;
    arrow::Int64Builder left_x_builder;
    for (std::int64_t i = 1; i <= 2; ++i) {
      ASSERT_TRUE(left_id_builder.Append(i).ok());
      ASSERT_TRUE(left_x_builder.Append(i * 10).ok());  // 10, 20
    }
    std::shared_ptr<arrow::Array> left_id_array, left_x_array;
    ASSERT_TRUE(left_id_builder.Finish(&left_id_array).ok());
    ASSERT_TRUE(left_x_builder.Finish(&left_x_array).ok());
    const auto left_dup_schema =
        arrow::schema({arrow::field("id", arrow::int64(), false), arrow::field("x", arrow::int64(), false)});
    const auto left_dup_table = arrow::Table::Make(left_dup_schema, {left_id_array, left_x_array});
    auto left_dup_sink = arrow::io::FileOutputStream::Open(left_dup_path_).ValueOrDie();
    const arrow::Status left_dup_status = parquet::arrow::WriteTable(
        *left_dup_table, arrow::default_memory_pool(), left_dup_sink, /*chunk_size=*/2);
    ASSERT_TRUE(left_dup_status.ok()) << left_dup_status.ToString();

    right_dup_path_ = (dir_ / "right_dup.parquet").string();
    arrow::Int64Builder right_id_builder;
    arrow::Int64Builder right_x_builder;
    for (std::int64_t i = 1; i <= 2; ++i) {
      ASSERT_TRUE(right_id_builder.Append(i).ok());
      ASSERT_TRUE(right_x_builder.Append(i * 100).ok());  // 100, 200
    }
    std::shared_ptr<arrow::Array> right_id_array, right_x_array;
    ASSERT_TRUE(right_id_builder.Finish(&right_id_array).ok());
    ASSERT_TRUE(right_x_builder.Finish(&right_x_array).ok());
    const auto right_dup_schema =
        arrow::schema({arrow::field("id", arrow::int64(), false), arrow::field("x", arrow::int64(), false)});
    const auto right_dup_table = arrow::Table::Make(right_dup_schema, {right_id_array, right_x_array});
    auto right_dup_sink = arrow::io::FileOutputStream::Open(right_dup_path_).ValueOrDie();
    const arrow::Status right_dup_status = parquet::arrow::WriteTable(
        *right_dup_table, arrow::default_memory_pool(), right_dup_sink, /*chunk_size=*/2);
    ASSERT_TRUE(right_dup_status.ok()) << right_dup_status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
  std::string regions_path_;
  std::string managers_path_;
  std::string left_dup_path_;
  std::string right_dup_path_;
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

// Regression test: build_logical_plan() used to reject a SELECT item that
// *combines* multiple aggregates arithmetically (e.g. a ratio), throwing
// "SELECT item '...' is neither an aggregate nor a GROUP BY column" even
// though it bound successfully -- exactly TPC-H Q14's shape. Fixed by
// rewrite_aggregate_refs() in logical_planner.cpp; see
// LogicalPlanner.AggregateArithmeticCombiningTwoAggregatesBuildsBothSlots
// in tests/unit/logical_plan_test.cpp for the structural version of this
// same check.
TEST_F(QueryEngineExecuteCpuTest, ScalarAggregateArithmeticCombiningTwoAggregatesMatchesExpectedRatio) {
  const QueryResult result = engine_.execute(
      "SELECT 100.0 * SUM(amount) / SUM(amount * 2) AS ratio FROM read_parquet('" + path_ + "')");

  ASSERT_EQ(result.rows_returned, 1);
  const auto ratio_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("ratio"));
  ASSERT_NE(ratio_column, nullptr);
  EXPECT_DOUBLE_EQ(ratio_column->Value(0), 50.0);  // 100 * 145 / 290
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

// Regression test: the CPU execution backend used to reject LIKE/NOT LIKE
// in every context ("unrecognized expression type in CPU expression
// compiler (LIKE/IN/CASE/DECIMAL are not yet supported...)"). Fixed by
// mapping LikeExpression to Arrow Compute's own "match_like" kernel in
// compile_expression_cpu() -- the same shared function the CASE fix
// landed on, so WHERE, SELECT list, and CASE branches (see
// CaseInGroupedAggregateWhereAndScalarAggregateMatchesExpectedTotals
// above, which predates this fix and only covers CASE) all gain LIKE
// support from this one change.
TEST_F(QueryEngineExecuteCpuTest, LikeAndNotLikeInWhereMatchExpectedRows) {
  const QueryResult like_result =
      engine_.execute("SELECT region FROM read_parquet('" + path_ + "') WHERE region LIKE 'A%'");
  EXPECT_EQ(like_result.rows_returned, 3);  // all 3 "A" rows

  const QueryResult not_like_result =
      engine_.execute("SELECT region FROM read_parquet('" + path_ + "') WHERE region NOT LIKE 'A%'");
  EXPECT_EQ(not_like_result.rows_returned, 3);  // all 3 "B" rows

  const QueryResult case_result = engine_.execute(
      "SELECT SUM(CASE WHEN region LIKE 'A%' THEN 1 ELSE 0 END) AS a_count FROM read_parquet('" + path_ +
      "')");
  ASSERT_EQ(case_result.batches.size(), 1u);
  const auto a_count_column =
      std::static_pointer_cast<arrow::Int64Array>(case_result.batches.front()->GetColumnByName("a_count"));
  ASSERT_NE(a_count_column, nullptr);
  EXPECT_EQ(a_count_column->Value(0), 3);
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

// EXTRACT(YEAR FROM ...) as a GROUP BY key -- exactly TPC-H Q7/Q9's shape,
// and the CPU-backend counterpart of
// QueryEngineExecuteTest.ExtractYearAsGroupByKeyMatchesExpectedTotals
// (tests/gpu/query_engine_execute_test.cpp), same fixture/expected values.
// A computed GROUP BY key (EXTRACT- or CASE-derived) used to be flatly
// rejected by acero_query_executor.cpp's HashAggregateNode translation
// ("GROUP BY by a computed expression is not yet supported by the CPU
// execution backend") -- fixed by projecting it under its own logical
// name via the same AggregateInputPlan machinery aggregate arguments
// already used, rather than resolve_aggregate_target's throwaway
// synthetic name (which would have silently renamed this key's actual
// output column -- see require_plain_column_index's own comment).
TEST_F(QueryEngineExecuteCpuTest, ExtractYearAsGroupByKeyMatchesExpectedTotals) {
  const QueryResult result =
      engine_.execute("SELECT EXTRACT(YEAR FROM event_date) AS y, SUM(amount) AS total FROM read_parquet('" +
                      path_ + "') GROUP BY y ORDER BY y");

  ASSERT_EQ(result.rows_returned, 2);
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto year_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("y"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(year_column, nullptr);
  ASSERT_NE(total_column, nullptr);

  std::map<std::int64_t, double> total_by_year;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    total_by_year[year_column->Value(i)] = total_column->Value(i);
  }
  ASSERT_EQ(total_by_year.size(), 2u);
  EXPECT_DOUBLE_EQ(total_by_year.at(2025), 110.0);
  EXPECT_DOUBLE_EQ(total_by_year.at(2026), 35.0);
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

// Regression test: a 3+-way JOIN chain used to be rejected outright at
// parse time ("KernelLake supports at most two read_parquet(...)
// sources"), even though the underlying hsql SQL parser already builds a
// correct left-deep join tree for it. Fixed by generalizing AstJoinClause/
// BoundJoin to a chain and building a left-deep chain of LogicalJoin nodes
// in build_logical_plan() -- see docs/ARCHITECTURE.md's "N-way joins"
// section. This end-to-end test exercises the whole pipeline (parse ->
// bind -> logical plan -> physical plan -> Acero execution) for a real
// 3-source join: sales -> regions (via region) -> managers (via
// region_name, a column that doesn't even exist on sales.parquet, so this
// genuinely requires the third source's schema to be resolved against the
// *combined* [sales, regions] schema, not just regions.parquet alone).
TEST_F(QueryEngineExecuteCpuTest, ThreeTableInnerJoinMatchesExpectedTotals) {
  const QueryResult result = engine_.execute(
      "SELECT m.manager, SUM(s.amount) AS total FROM read_parquet('" + path_ + "') AS s JOIN read_parquet('" +
      regions_path_ + "') AS r ON s.region = r.region JOIN read_parquet('" + managers_path_ +
      "') AS m ON r.region_name = m.region_name GROUP BY m.manager");

  ASSERT_EQ(result.rows_returned, 2);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto manager_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("manager"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(manager_column, nullptr);
  ASSERT_NE(total_column, nullptr);

  std::map<std::string, double> totals_by_manager;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    totals_by_manager[manager_column->GetString(i)] = total_column->Value(i);
  }
  ASSERT_EQ(totals_by_manager.size(), 2u);
  EXPECT_DOUBLE_EQ(totals_by_manager.at("Ann"), 35.0);  // Alpha: 10+20+5
  EXPECT_DOUBLE_EQ(totals_by_manager.at("Bo"), 110.0);  // Beta: 100+7+3
}

// Regression test for find_scan_boundary()'s join recursion
// (physical_planner.cpp): a WHERE filter above a 3+-way JOIN chain sits on
// top of the *outer* HashJoinNode, whose left child is itself another
// HashJoinNode (the inner sales-regions join), not a ParquetScanNode --
// find_scan_boundary() must recurse through that nested join correctly to
// find the combined schema/column-map the WHERE predicate needs remapping
// against. No existing test combines a WHERE filter with a JOIN chain this
// deep (ThreeTableInnerJoinMatchesExpectedTotals above has no WHERE at all;
// InSubqueryWorksAlongsideARealJoin below has WHERE but only a single JOIN).
TEST_F(QueryEngineExecuteCpuTest, ThreeTableInnerJoinWithWhereFilterMatchesExpectedTotals) {
  const QueryResult result = engine_.execute(
      "SELECT m.manager, SUM(s.amount) AS total FROM read_parquet('" + path_ + "') AS s JOIN read_parquet('" +
      regions_path_ + "') AS r ON s.region = r.region JOIN read_parquet('" + managers_path_ +
      "') AS m ON r.region_name = m.region_name WHERE s.amount > 5 GROUP BY m.manager");

  ASSERT_EQ(result.rows_returned, 2);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto manager_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("manager"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(manager_column, nullptr);
  ASSERT_NE(total_column, nullptr);

  std::map<std::string, double> totals_by_manager;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    totals_by_manager[manager_column->GetString(i)] = total_column->Value(i);
  }
  ASSERT_EQ(totals_by_manager.size(), 2u);
  EXPECT_DOUBLE_EQ(totals_by_manager.at("Ann"), 30.0);  // Alpha: 10+20 (amount=5 filtered out by WHERE)
  EXPECT_DOUBLE_EQ(totals_by_manager.at("Bo"), 107.0);  // Beta: 100+7 (amount=3 filtered out by WHERE)
}

// Regression test for physical_planner.cpp's remap_columns(): it used to
// resolve a ColumnExpression above a JOIN by re-looking-up its bare *name*
// against the combined narrowed schema (Schema::find_field(), first match
// only) instead of trusting the binder-resolved column_index() already on
// the expression -- so `l.x`/`r.x` (same name, opposite sides, genuinely
// different values here) both silently resolved to the left side's "x".
TEST_F(QueryEngineExecuteCpuTest, JoinSelectsSameNamedColumnFromBothSidesWithoutCollision) {
  // No ORDER BY here: ORDER BY on a plain (non-aggregate) query only binds
  // against the pre-projection source schema, not the SELECT list's own
  // output aliases (see binder.cpp's ORDER BY handling) -- a separate,
  // pre-existing gap, not something this test is about. l.id is selected
  // instead so rows can be matched up regardless of output order.
  const QueryResult result =
      engine_.execute("SELECT l.id AS row_id, l.x AS lx, r.x AS rx FROM read_parquet('" + left_dup_path_ +
                      "') AS l JOIN read_parquet('" + right_dup_path_ + "') AS r ON l.id = r.id");

  ASSERT_EQ(result.rows_returned, 2);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto id_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("row_id"));
  const auto lx_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("lx"));
  const auto rx_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("rx"));
  ASSERT_NE(id_column, nullptr);
  ASSERT_NE(lx_column, nullptr);
  ASSERT_NE(rx_column, nullptr);

  std::map<std::int64_t, std::pair<std::int64_t, std::int64_t>> by_id;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    by_id[id_column->Value(i)] = {lx_column->Value(i), rx_column->Value(i)};
  }
  ASSERT_EQ(by_id.size(), 2u);
  EXPECT_EQ(by_id.at(1).first, 10);
  EXPECT_EQ(by_id.at(1).second, 100);
  EXPECT_EQ(by_id.at(2).first, 20);
  EXPECT_EQ(by_id.at(2).second, 200);
}

// Regression test for logical_planner.cpp's register_aggregate()/
// rewrite_aggregate_refs(): both used to key their dedup maps by
// Expression::to_string(), under which SUM(l.x) and SUM(r.x) render to the
// identical string "SUM(x)" once bound -- so the second aggregate silently
// reused the first one's LogicalAggregate slot instead of getting its own.
TEST_F(QueryEngineExecuteCpuTest, JoinSumsSameNamedColumnFromBothSidesWithoutCollision) {
  const QueryResult result =
      engine_.execute("SELECT SUM(l.x) AS l_total, SUM(r.x) AS r_total FROM read_parquet('" + left_dup_path_ +
                      "') AS l JOIN read_parquet('" + right_dup_path_ + "') AS r ON l.id = r.id");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto l_total_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("l_total"));
  const auto r_total_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("r_total"));
  ASSERT_NE(l_total_column, nullptr);
  ASSERT_NE(r_total_column, nullptr);

  EXPECT_EQ(l_total_column->Value(0), 30);   // 10 + 20
  EXPECT_EQ(r_total_column->Value(0), 300);  // 100 + 200
}

// Regression test for binder.cpp's references_ungrouped_column(): it used
// to compare AstColumnRef::name against GROUP BY's bare column names,
// ignoring the table qualifier entirely -- so `GROUP BY l.x` was wrongly
// treated as covering an ungrouped `SELECT r.x`, silently accepting a
// query that should be rejected (r.x is neither grouped nor aggregated).
TEST_F(QueryEngineExecuteCpuTest, RejectsGroupByOnOneJoinSideWithSameNamedUngroupedColumnFromTheOther) {
  EXPECT_THROW((void)(engine_.execute("SELECT r.x, COUNT(*) AS cnt FROM read_parquet('" + left_dup_path_ +
                                      "') AS l JOIN read_parquet('" + right_dup_path_ +
                                      "') AS r ON l.id = r.id GROUP BY l.x")),
               BindingError);
}

// Regression test for physical_planner.cpp's references_scan_schema():
// it used to treat *any* LogicalFilter as scan-schema-referencing
// unconditionally, which was safe before HAVING existed but breaks once
// the optimizer's redundant-projection-removal elides the aggregate
// path's final re-projection (exactly what happens here, since the
// SELECT list already matches the aggregate's own output order) --
// leaving LogicalSort sitting directly on the HAVING LogicalFilter, whose
// ColumnExpression indices describe the *aggregate's* output schema, not
// the scan's. The real symptom was an Acero execution-time error ("No
// match for FieldRef.FieldPath(2)"), only reachable via this exact
// combination: a JOIN (so the final projection can become a genuine
// identity projection over the combined schema), GROUP BY + HAVING (a
// LogicalFilter directly on a LogicalAggregate), and ORDER BY (a
// LogicalSort that ends up directly on that filter once the redundant
// projection is elided).
TEST_F(QueryEngineExecuteCpuTest, JoinGroupByHavingOrderByDoesNotMisremapSortKeyIndices) {
  const QueryResult result = engine_.execute(
      "SELECT r.region_name, SUM(s.amount) AS total FROM read_parquet('" + path_ +
      "') AS s JOIN read_parquet('" + regions_path_ +
      "') AS r ON s.region = r.region GROUP BY r.region_name HAVING SUM(s.amount) > 50 ORDER BY total DESC");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto name_column =
      std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region_name"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(name_column, nullptr);
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(name_column->GetString(0), "Beta");
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
}

// The GROUP BY + HAVING + LIMIT-with-no-ORDER-BY shape (same bug class as
// JoinGroupByHavingOrderByDoesNotMisremapSortKeyIndices above, one node
// combination further: LogicalLimit instead of LogicalSort sitting directly
// on the HAVING LogicalFilter) is verified at the physical-plan level in
// physical_planner_test.cpp's
// JoinGroupByHavingLimitRemapsCorrectlyWithNoOrderBy, not here -- see that
// test's own comment for why: Acero's own "fetch" node unconditionally
// rejects LIMIT on top of any unordered input (HashAggregateNode's output
// carries no ordering guarantee), so *no* `GROUP BY ... LIMIT` query without
// an ORDER BY can execute end to end on this CPU backend today, regardless
// of HAVING/JOIN -- a separate, pre-existing Acero-integration limitation,
// not a physical_planner.cpp remapping bug, and out of scope to fix here.

// TPC-H Q11's own shape, end to end: a literal HAVING threshold filters
// out groups below it. region A totals 35.0, region B totals 110.0 --
// only B survives a > 50 threshold.
TEST_F(QueryEngineExecuteCpuTest, HavingWithLiteralThresholdFiltersGroups) {
  const QueryResult result = engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" +
                                             path_ + "') GROUP BY region HAVING SUM(amount) > 50");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(region_column->GetString(0), "B");
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
}

// Regression test: finish_logical_plan() rewrites HAVING's own aggregate
// references (register_aggregate()) in the same pre-LogicalAggregate phase
// as the SELECT list, specifically so a HAVING aggregate that never appears
// in the SELECT list itself still grows LogicalAggregate's own `aggregates`
// list before it's constructed -- otherwise LogicalAggregate would already
// be built from a stale, smaller list by the time HAVING tried to rewrite
// its own reference. Same split as HavingWithLiteralThresholdFiltersGroups
// (region A totals 35.0, region B totals 110.0), but `total` is never
// selected here.
TEST_F(QueryEngineExecuteCpuTest, HavingReferencesAggregateNotInSelectListFiltersGroupsCorrectly) {
  const QueryResult result = engine_.execute("SELECT region FROM read_parquet('" + path_ +
                                             "') GROUP BY region HAVING SUM(amount) > 50");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  ASSERT_NE(region_column, nullptr);
  EXPECT_EQ(region_column->GetString(0), "B");
}

// QueryEngine::evaluate_scalar_subquery(): a real non-correlated scalar
// subquery computes the HAVING threshold instead of a literal -- exactly
// TPC-H Q11's own `SUM(...) * 0.0001` shape. 145.0 (grand total) * 0.3 =
// 43.5, so region A (35.0) is excluded and region B (110.0) survives, the
// same split as the literal-threshold test above but driven by a nested
// bind->plan->execute cycle instead of a constant.
TEST_F(QueryEngineExecuteCpuTest, HavingWithScalarSubqueryThresholdFiltersGroups) {
  const QueryResult result = engine_.execute(
      "SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
      "') GROUP BY region HAVING SUM(amount) > (SELECT SUM(amount) * 0.3 FROM read_parquet('" + path_ +
      "'))");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(region_column->GetString(0), "B");
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
}

// sql::resolve_subqueries() recurses through an AstBetween's lower/upper
// bounds -- a subquery sitting directly as HAVING's own comparison operand
// (the test above) never exercises that recursion. Same threshold/split as
// HavingWithScalarSubqueryThresholdFiltersGroups (43.5, excluding region A's
// 35.0 and including region B's 110.0), just reached through a BETWEEN
// instead of a bare comparison.
TEST_F(QueryEngineExecuteCpuTest, HavingSubqueryNestedInsideBetweenFiltersGroups) {
  const QueryResult result =
      engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                      "') GROUP BY region HAVING SUM(amount) BETWEEN "
                      "(SELECT SUM(amount) * 0.3 FROM read_parquet('" +
                      path_ + "')) AND 200");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(region_column->GetString(0), "B");
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
}

// Same recursion check, but for an AstCase branch instead of an AstBetween
// bound -- resolve_subqueries() walks both when_then results and the else
// branch.
TEST_F(QueryEngineExecuteCpuTest, HavingSubqueryNestedInsideCaseFiltersGroups) {
  const QueryResult result =
      engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                      "') GROUP BY region HAVING SUM(amount) > CASE WHEN SUM(amount) > 0 THEN "
                      "(SELECT SUM(amount) * 0.3 FROM read_parquet('" +
                      path_ + "')) ELSE 0 END");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(region_column->GetString(0), "B");
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
}

// A HAVING subquery's own subquery: subquery1 (a grouped aggregate that
// itself has a HAVING referencing subquery2) must resolve subquery2
// first, collapse to a genuine single row (region A, 35.0 -- the only
// region whose total is below subquery2's 72.5 threshold), and only then
// hand that one value up to the outer HAVING as its own threshold.
// Regression coverage for evaluate_scalar_subquery()'s own recursive
// resolved.having != nullptr branch, which a flat single-level subquery
// test can't exercise.
TEST_F(QueryEngineExecuteCpuTest, NestedHavingSubqueryResolvesInnerSubqueryFirst) {
  const QueryResult result = engine_.execute(
      "SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
      "') GROUP BY region HAVING SUM(amount) > "
      "(SELECT SUM(amount) AS x FROM read_parquet('" +
      path_ + "') GROUP BY region HAVING SUM(amount) < (SELECT SUM(amount) * 0.5 FROM read_parquet('" +
      path_ + "')))");

  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(region_column->GetString(0), "B");
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
}

TEST_F(QueryEngineExecuteCpuTest, HavingSubqueryReturningZeroRowsThrowsExecutionError) {
  EXPECT_THROW((void)(engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                                      "') GROUP BY region HAVING SUM(amount) > "
                                      "(SELECT amount FROM read_parquet('" +
                                      path_ + "') WHERE region = 'nonexistent')")),
               ExecutionError);
}

TEST_F(QueryEngineExecuteCpuTest, HavingSubqueryReturningMultipleRowsThrowsExecutionError) {
  EXPECT_THROW((void)(engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                                      "') GROUP BY region HAVING SUM(amount) > "
                                      "(SELECT amount FROM read_parquet('" +
                                      path_ + "'))")),
               ExecutionError);
}

TEST_F(QueryEngineExecuteCpuTest, HavingSubqueryReturningMultipleColumnsThrowsExecutionError) {
  // region = 'A' AND amount = 10 matches exactly one row (10.0, "A") --
  // isolates the column-count check from the row-count check, since this
  // subquery genuinely returns one row, just with two columns.
  EXPECT_THROW((void)(engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                                      "') GROUP BY region HAVING SUM(amount) > "
                                      "(SELECT region, amount FROM read_parquet('" +
                                      path_ + "') WHERE region = 'A' AND amount = 10)")),
               ExecutionError);
}

// The subquery is bound independently against its own FROM clause's
// schema only -- it has no access to the outer query's tables/aliases, so
// referencing a column that exists on the outer query but not on the
// subquery's own source must fail exactly like any other unknown-column
// reference, not silently resolve against the outer scope (there is no
// correlated-subquery support to accidentally provide that).
TEST_F(QueryEngineExecuteCpuTest, HavingSubqueryCannotReferenceOuterQueryColumns) {
  EXPECT_THROW((void)(engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                                      "') GROUP BY region HAVING SUM(amount) > "
                                      "(SELECT COUNT(*) FROM read_parquet('" +
                                      regions_path_ + "') WHERE amount > 5)")),
               BindingError);
}

// TPC-H Q18's own shape: `x IN (SELECT ...)` in WHERE, resolved into a
// literal list (an OR-chain of equalities) before binding. Only region B
// has a row with amount > 50 (100.0), so the subquery resolves to {"B"}
// and the outer query returns exactly the 3 region-B rows.
TEST_F(QueryEngineExecuteCpuTest, InSubqueryFiltersToMatchingRows) {
  const QueryResult result = engine_.execute("SELECT region FROM read_parquet('" + path_ +
                                             "') WHERE region IN (SELECT region FROM read_parquet('" + path_ +
                                             "') WHERE amount > 50)");
  EXPECT_EQ(result.rows_returned, 3);
  const auto region_column =
      std::static_pointer_cast<arrow::StringArray>(result.batches.front()->GetColumnByName("region"));
  ASSERT_NE(region_column, nullptr);
  for (std::int64_t i = 0; i < region_column->length(); ++i) {
    EXPECT_EQ(region_column->GetString(i), "B");
  }
}

// The IN-subquery is one conjunct in a larger AND-chain alongside a real
// predicate on the outer table -- exactly Q18's own WHERE shape
// (`o_orderkey IN (...) AND c_custkey = o_custkey AND ...`). Region B's
// rows are 100.0/7.0/3.0; ANDing `amount > 5` narrows to just 100.0/7.0.
TEST_F(QueryEngineExecuteCpuTest, InSubqueryAndedWithOtherWhereConjunctsMatchesExpectedRows) {
  const QueryResult result = engine_.execute("SELECT amount FROM read_parquet('" + path_ +
                                             "') WHERE region IN (SELECT region FROM read_parquet('" + path_ +
                                             "') WHERE amount > 50) AND amount > 5");
  EXPECT_EQ(result.rows_returned, 2);
  const auto amount_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("amount"));
  ASSERT_NE(amount_column, nullptr);
  std::vector<double> amounts;
  for (std::int64_t i = 0; i < amount_column->length(); ++i) {
    amounts.push_back(amount_column->Value(i));
  }
  std::sort(amounts.begin(), amounts.end());
  EXPECT_DOUBLE_EQ(amounts[0], 7.0);
  EXPECT_DOUBLE_EQ(amounts[1], 100.0);
}

// `x IN (SELECT ... /* zero rows */)` is always false -- standard SQL
// semantics for an empty set, handled by sql::resolve_in_subqueries()
// itself (not an error) -- see that function's own doc comment.
TEST_F(QueryEngineExecuteCpuTest, InSubqueryWithEmptyResultMatchesNoRows) {
  const QueryResult result = engine_.execute("SELECT region FROM read_parquet('" + path_ +
                                             "') WHERE region IN (SELECT region FROM read_parquet('" + path_ +
                                             "') WHERE amount > 1000)");
  EXPECT_EQ(result.rows_returned, 0);
}

// `x NOT IN (SELECT ... /* zero rows */)` is always true, the mirror image
// of the empty-IN case above.
TEST_F(QueryEngineExecuteCpuTest, NotInSubqueryWithEmptyResultMatchesAllRows) {
  const QueryResult result = engine_.execute("SELECT COUNT(*) AS n FROM read_parquet('" + path_ +
                                             "') WHERE region NOT IN (SELECT region FROM read_parquet('" +
                                             path_ + "') WHERE amount > 1000)");
  const auto n_column =
      std::static_pointer_cast<arrow::Int64Array>(result.batches.front()->GetColumnByName("n"));
  ASSERT_NE(n_column, nullptr);
  EXPECT_EQ(n_column->Value(0), 6);
}

// Real end-to-end coverage of the exact Q18 shape: an IN-subquery in
// WHERE alongside a real JOIN, not just a single-table query.
TEST_F(QueryEngineExecuteCpuTest, InSubqueryWorksAlongsideARealJoin) {
  const QueryResult result = engine_.execute("SELECT r.region_name, s.amount FROM read_parquet('" + path_ +
                                             "') AS s JOIN read_parquet('" + regions_path_ +
                                             "') AS r ON s.region = r.region WHERE s.region IN "
                                             "(SELECT region FROM read_parquet('" +
                                             path_ + "') WHERE amount > 50)");
  EXPECT_EQ(result.rows_returned, 3);
  const auto name_column =
      std::static_pointer_cast<arrow::StringArray>(result.batches.front()->GetColumnByName("region_name"));
  ASSERT_NE(name_column, nullptr);
  for (std::int64_t i = 0; i < name_column->length(); ++i) {
    EXPECT_EQ(name_column->GetString(i), "Beta");
  }
}

TEST_F(QueryEngineExecuteCpuTest, InSubqueryReturningMultipleColumnsThrowsExecutionError) {
  EXPECT_THROW((void)(engine_.execute("SELECT region FROM read_parquet('" + path_ +
                                      "') WHERE region IN (SELECT region, amount FROM read_parquet('" +
                                      path_ + "'))")),
               ExecutionError);
}

// Regression coverage: MIN/MAX/AVG only ever exercised SUM/COUNT on this
// backend before this test -- a real gap, since translate_aggregate()'s
// Min/Max/Avg branches (acero_query_executor.cpp) are separate Arrow
// Compute function names ("min"/"max"/"mean", "hash_min"/"hash_max"/
// "hash_mean"), not parameter variations of SUM's own path.
TEST_F(QueryEngineExecuteCpuTest, GroupedMinMaxAvgMatchExpectedValues) {
  const QueryResult result = engine_.execute(
      "SELECT region, MIN(amount) AS lo, MAX(amount) AS hi, AVG(amount) AS avg "
      "FROM read_parquet('" +
      path_ + "') GROUP BY region");
  ASSERT_EQ(result.rows_returned, 2);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto lo_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("lo"));
  const auto hi_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("hi"));
  const auto avg_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("avg"));
  ASSERT_NE(region_column, nullptr);
  ASSERT_NE(lo_column, nullptr);
  ASSERT_NE(hi_column, nullptr);
  ASSERT_NE(avg_column, nullptr);

  std::map<std::string, std::tuple<double, double, double>> by_region;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    by_region[region_column->GetString(i)] = {lo_column->Value(i), hi_column->Value(i), avg_column->Value(i)};
  }
  ASSERT_EQ(by_region.size(), 2u);
  EXPECT_DOUBLE_EQ(std::get<0>(by_region.at("A")), 5.0);
  EXPECT_DOUBLE_EQ(std::get<1>(by_region.at("A")), 20.0);
  EXPECT_DOUBLE_EQ(std::get<2>(by_region.at("A")), 35.0 / 3.0);
  EXPECT_DOUBLE_EQ(std::get<0>(by_region.at("B")), 3.0);
  EXPECT_DOUBLE_EQ(std::get<1>(by_region.at("B")), 100.0);
  EXPECT_DOUBLE_EQ(std::get<2>(by_region.at("B")), 110.0 / 3.0);
}

TEST_F(QueryEngineExecuteCpuTest, ScalarMinMaxAvgMatchExpectedValues) {
  const QueryResult result = engine_.execute(
      "SELECT MIN(amount) AS lo, MAX(amount) AS hi, AVG(amount) AS avg FROM read_parquet('" + path_ + "')");
  ASSERT_EQ(result.rows_returned, 1);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto lo_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("lo"));
  const auto hi_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("hi"));
  const auto avg_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("avg"));
  ASSERT_NE(lo_column, nullptr);
  ASSERT_NE(hi_column, nullptr);
  ASSERT_NE(avg_column, nullptr);
  EXPECT_DOUBLE_EQ(lo_column->Value(0), 3.0);
  EXPECT_DOUBLE_EQ(hi_column->Value(0), 100.0);
  EXPECT_DOUBLE_EQ(avg_column->Value(0), 145.0 / 6.0);
}

// Regression coverage: CAST(...) was essentially 0%-covered on this backend
// end to end before this test -- every amount in the fixture is already a
// whole number, so truncate-vs-round ambiguity (this project's CAST
// truncates; see docs/ARCHITECTURE.md for why this differs from DuckDB's
// rounding) doesn't affect this result. Mirrors
// QueryEngineExecuteTest.CastConvertsAmountToInteger (tests/gpu/
// query_engine_execute_test.cpp), the GPU-backend counterpart.
TEST_F(QueryEngineExecuteCpuTest, CastConvertsAmountToInteger) {
  const QueryResult result = engine_.execute(
      "SELECT CAST(amount AS BIGINT) AS amount_int FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto column =
      std::static_pointer_cast<arrow::Int64Array>(result.batches.front()->GetColumnByName("amount_int"));
  ASSERT_NE(column, nullptr);
  std::vector<std::int64_t> values;
  for (std::int64_t i = 0; i < result.batches.front()->num_rows(); ++i) values.push_back(column->Value(i));
  std::sort(values.begin(), values.end());
  EXPECT_EQ(values, (std::vector<std::int64_t>{3, 7, 100}));
}

// Regression coverage: BETWEEN and the unary operators (NOT/Negate/IsNull/
// IsNotNull) were only ever referenced in parser/binder/optimizer unit
// tests, never actually executed end to end on this backend.
TEST_F(QueryEngineExecuteCpuTest, BetweenAndUnaryOperatorsMatchExpectedRows) {
  const QueryResult between_result =
      engine_.execute("SELECT amount FROM read_parquet('" + path_ + "') WHERE amount BETWEEN 5 AND 20");
  EXPECT_EQ(between_result.rows_returned, 4);  // 10.0, 20.0, 5.0, 7.0

  const QueryResult not_result =
      engine_.execute("SELECT amount FROM read_parquet('" + path_ + "') WHERE NOT (region = 'A')");
  EXPECT_EQ(not_result.rows_returned, 3);  // all 3 region B rows

  const QueryResult negate_result =
      engine_.execute("SELECT amount FROM read_parquet('" + path_ + "') WHERE -amount < -50");
  EXPECT_EQ(negate_result.rows_returned, 1);  // only amount == 100.0
}

// Regression coverage: Hive-partition-column materialization on this
// backend was never exercised end to end before this test (the GPU
// backend already has ParquetScanOperatorTest.MaterializesPartitionColumnsPerFragment
// covering its own, independently-implemented equivalent).
TEST_F(QueryEngineExecuteCpuTest, MaterializesPartitionColumnsAcrossFragments) {
  const fs::path partition_a = dir_ / "cpu_partitioned" / "region=A";
  const fs::path partition_b = dir_ / "cpu_partitioned" / "region=B";
  fs::create_directories(partition_a);
  fs::create_directories(partition_b);

  auto write_amount_file = [](const std::string& file_path, double value) {
    arrow::DoubleBuilder amount_builder;
    ASSERT_TRUE(amount_builder.Append(value).ok());
    std::shared_ptr<arrow::Array> amount_array;
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    const auto schema = arrow::schema({arrow::field("amount", arrow::float64(), false)});
    const auto table = arrow::Table::Make(schema, {amount_array});
    auto sink = arrow::io::FileOutputStream::Open(file_path).ValueOrDie();
    const arrow::Status status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink,
                                                            /*chunk_size=*/1);
    ASSERT_TRUE(status.ok()) << status.ToString();
  };
  write_amount_file((partition_a / "part.parquet").string(), 42.0);
  write_amount_file((partition_b / "part.parquet").string(), 99.0);

  const QueryResult result = engine_.execute("SELECT region, amount FROM read_parquet('" +
                                             (dir_ / "cpu_partitioned").string() + "')");
  ASSERT_EQ(result.rows_returned, 2);
  std::map<std::string, double> amount_by_region;
  for (const std::shared_ptr<arrow::RecordBatch>& batch : result.batches) {
    const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
    const auto amount_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("amount"));
    ASSERT_NE(region_column, nullptr);
    ASSERT_NE(amount_column, nullptr);
    for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
      amount_by_region[region_column->GetString(i)] = amount_column->Value(i);
    }
  }
  ASSERT_EQ(amount_by_region.size(), 2u);
  EXPECT_DOUBLE_EQ(amount_by_region.at("A"), 42.0);
  EXPECT_DOUBLE_EQ(amount_by_region.at("B"), 99.0);
}

// Regression coverage: no CPU-backend test previously forced a real
// StorageError translation (corrupt/malformed Parquet content) -- confirms
// open_fragment_reader()'s error path surfaces a clean exception rather
// than crashing or hanging.
TEST_F(QueryEngineExecuteCpuTest, CorruptParquetFileSurfacesCleanError) {
  const std::string corrupt_path = (dir_ / "corrupt.parquet").string();
  {
    std::ofstream corrupt_file(corrupt_path, std::ios::binary);
    corrupt_file << "not a real parquet file";
  }
  EXPECT_THROW((void)(engine_.execute("SELECT * FROM read_parquet('" + corrupt_path + "')")), std::exception);
}

// Documents a known, acknowledged, out-of-scope limitation rather than a
// regression: AggregateInputPlan::claimed_names's own comment
// (acero_query_executor.cpp) explains that two same-named GROUP BY keys
// from opposite JOIN sides, selected *together*, get a synthetic name for
// the second one that doesn't match its own HashAggregateNode::
// output_schema() field name -- a real GetColumnByName() mismatch
// downstream. Fixing this needs Field-level qualification, explicitly
// out of scope for that comment; this test exists so the behavior has a
// regression anchor instead of being silently untested.
TEST_F(QueryEngineExecuteCpuTest, GroupByOnSameNamedColumnFromBothJoinSidesTogetherIsAKnownLimitation) {
  EXPECT_THROW((void)(engine_.execute("SELECT l.x, r.x, COUNT(*) AS cnt FROM read_parquet('" +
                                      left_dup_path_ + "') AS l JOIN read_parquet('" + right_dup_path_ +
                                      "') AS r ON l.id = r.id GROUP BY l.x, r.x")),
               std::exception);
}

}  // namespace
}  // namespace kernellake
