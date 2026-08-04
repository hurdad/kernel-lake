// End-to-end test for QueryEngine::execute() against real Parquet files,
// exercising the exact MVP query shape from the spec: parse -> bind ->
// logical plan -> optimize -> physical plan -> Parquet pruning -> GPU
// filter -> GPU grouped aggregation -> Arrow result.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <filesystem>
#include <map>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/memory/rmm_environment.hpp"

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

// Regression test for a real bug: a bare `COUNT(*)` with no other column
// referenced anywhere in the query (no WHERE/GROUP BY/join) legitimately
// produces an empty required_columns() from the optimizer's column-pruning
// pass, but a cudf::table built from zero selected columns has no column
// to derive num_rows() from -- every chunk was silently treated as empty
// and this returned 0 instead of the real row count. Every pre-existing
// COUNT(*) test in this file references another column via GROUP BY/WHERE/
// a join key, so this exact shape had no test coverage before. See
// docs/ARCHITECTURE.md's "Ubuntu 26.04 baseline" section and
// docs/ROADMAP.md for the full root-cause writeup; fixed in
// src/io/physical_planner.cpp's convert_scan().
TEST_F(QueryEngineExecuteTest, BareCountStarWithNoOtherColumnReferenceMatchesRealRowCount) {
  const QueryResult result = engine_.execute("SELECT COUNT(*) AS n FROM read_parquet('" + path_ + "')");

  ASSERT_EQ(result.rows_returned, 1);
  ASSERT_EQ(result.batches.size(), 1u);
  const auto n_column =
      std::static_pointer_cast<arrow::Int64Array>(result.batches.front()->GetColumnByName("n"));
  ASSERT_NE(n_column, nullptr);
  EXPECT_EQ(n_column->Value(0), 6);
}

TEST_F(QueryEngineExecuteTest, PlainProjectionReturnsAllRows) {
  const QueryResult result = engine_.execute("SELECT region FROM read_parquet('" + path_ + "')");
  EXPECT_EQ(result.rows_returned, 6);
}

TEST_F(QueryEngineExecuteTest, AggregateOrderByProducesDescendingTotals) {
  // region A: 10+20+5=35, region B: 100+7+3=110 (see SetUp's regions/
  // amounts). SELECT order [total, region] differs from HashAggregate's
  // natural [region, total] output, so the reprojection survives the
  // optimizer's redundant-projection removal -- exercising the exact
  // shape that surfaced the LogicalProjection/Sort remap bugs fixed in
  // physical_planner.cpp (see PhysicalPlannerTest in
  // tests/unit/physical_planner_test.cpp for the structural version of
  // this same check).
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

TEST_F(QueryEngineExecuteTest, LikeFiltersToMatchingRegionRows) {
  // region A rows: 10+20+5=35 (see SetUp's regions/amounts).
  const QueryResult result = engine_.execute("SELECT SUM(amount) AS total FROM read_parquet('" + path_ +
                                             "') WHERE region LIKE 'A%'");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto total_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
  ASSERT_NE(total_column, nullptr);
  EXPECT_DOUBLE_EQ(total_column->Value(0), 35.0);
}

TEST_F(QueryEngineExecuteTest, NotLikeFiltersToNonMatchingRegionRows) {
  // This specific shape (a scalar aggregate with no GROUP BY to also
  // reference "region") is what originally surfaced the missing
  // LikeExpression case in the optimizer's required-columns collector --
  // without it, "region" would be silently pruned from the scan entirely.
  const QueryResult result = engine_.execute("SELECT SUM(amount) AS total FROM read_parquet('" + path_ +
                                             "') WHERE region NOT LIKE 'A%'");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto total_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
  ASSERT_NE(total_column, nullptr);
  EXPECT_DOUBLE_EQ(total_column->Value(0), 110.0);
}

TEST_F(QueryEngineExecuteTest, InDesugarsToEquivalentOrChain) {
  const QueryResult result =
      engine_.execute("SELECT COUNT(*) AS n FROM read_parquet('" + path_ + "') WHERE region IN ('A')");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto n_column =
      std::static_pointer_cast<arrow::Int64Array>(result.batches.front()->GetColumnByName("n"));
  ASSERT_NE(n_column, nullptr);
  EXPECT_EQ(n_column->Value(0), 3);
}

TEST_F(QueryEngineExecuteTest, CaseWithGroupByAliasBucketsRows) {
  // amounts > 15: 20, 100 (2 rows); <= 15: 10, 5, 7, 3 (4 rows). "bucket"
  // is not a base-table column -- GROUP BY only resolves it by matching
  // this query's own SELECT-list alias for the CASE expression, and the
  // CASE's own STRING literal branches ('high'/'low') must not be routed
  // through cudf::ast::compute_column (it cannot produce STRING output at
  // all, even for a pure literal) -- see docs/ARCHITECTURE.md.
  const QueryResult result = engine_.execute(
      "SELECT CASE WHEN amount > 15 THEN 'high' ELSE 'low' END AS bucket, COUNT(*) AS n FROM read_parquet('" +
      path_ + "') GROUP BY bucket ORDER BY bucket");

  ASSERT_EQ(result.rows_returned, 2);
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  const auto bucket_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("bucket"));
  const auto n_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("n"));
  ASSERT_NE(bucket_column, nullptr);
  ASSERT_NE(n_column, nullptr);

  std::map<std::string, std::int64_t> counts_by_bucket;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    counts_by_bucket[bucket_column->GetString(i)] = n_column->Value(i);
  }
  ASSERT_EQ(counts_by_bucket.size(), 2u);
  EXPECT_EQ(counts_by_bucket.at("high"), 2);
  EXPECT_EQ(counts_by_bucket.at("low"), 4);
}

// Regression test: ScalarAggregateOperator (the no-GROUP-BY aggregate path)
// used to compile a non-plain-column aggregate argument via the plain
// cudf::ast ExpressionCompiler directly, which has no CASE support at all
// ("unrecognized expression type in GPU expression compiler") -- unlike
// HashAggregateOperator (the GROUP-BY path, exercised by
// CaseWithGroupByAliasBucketsRows above), which already had the CASE-aware
// compile_expr()/materialize() machinery. This is exactly TPC-H Q14's
// shape (a scalar SUM(CASE WHEN ... THEN ... ELSE ... END), no GROUP BY).
// Fixed by giving ScalarAggregateOperator the identical CompiledExpr/
// CompiledCase machinery HashAggregateOperator and ProjectionOperator
// already have. Note this does NOT cover CASE inside WHERE -- confirmed
// still unsupported by FilterOperator (a separate, still-open gap; neither
// Q12 nor Q14's own WHERE clause needs it, so it's out of scope here).
TEST_F(QueryEngineExecuteTest, CaseInScalarAggregateMatchesExpectedTotal) {
  const QueryResult result = engine_.execute(
      "SELECT SUM(CASE WHEN amount > 15 THEN 1 ELSE 0 END) AS high_count FROM read_parquet('" + path_ + "')");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto high_count_column =
      std::static_pointer_cast<arrow::Int64Array>(result.batches.front()->GetColumnByName("high_count"));
  ASSERT_NE(high_count_column, nullptr);
  EXPECT_EQ(high_count_column->Value(0), 2);  // 20.0 and 100.0
}

TEST_F(QueryEngineExecuteTest, CastConvertsAmountToInteger) {
  // Every amount in SetUp is already a whole number, so truncate-vs-round
  // ambiguity (our CAST truncates; see docs/ARCHITECTURE.md for why this
  // differs from DuckDB's rounding behavior) doesn't affect this result.
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

// --- Phase 0: split execution path (explain() + execute(physical, rmm)) ---
// exercised the same way a long-lived caller (e.g. a future Flight SQL
// server) would use it: one RmmEnvironment built once, reused across the
// plan-then-execute split, instead of QueryEngine::execute(sql)'s own
// build-one-per-call convenience wrapper. See docs/ARCHITECTURE.md's
// Concurrency notes and the split's own doc comments in query_engine.hpp.

TEST_F(QueryEngineExecuteTest, SplitExecutionPathMatchesConvenienceWrapper) {
  const std::string sql = "SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                          "') WHERE event_date >= DATE '2026-01-01' GROUP BY region";
  const QueryResult direct = engine_.execute(sql);

  const PhysicalPlanPtr physical = engine_.explain(sql);
  RmmEnvironment rmm_environment(default_config());
  const QueryResult split = engine_.execute(physical, rmm_environment);

  ASSERT_EQ(direct.batches.size(), 1u);
  ASSERT_EQ(split.batches.size(), 1u);
  EXPECT_EQ(direct.rows_returned, split.rows_returned);
  EXPECT_TRUE(direct.batches.front()->Equals(*split.batches.front()));
}

TEST_F(QueryEngineExecuteTest, ConvenienceWrapperPopulatesRealTimingFields) {
  const QueryResult result = engine_.execute("SELECT region, SUM(amount) AS total FROM read_parquet('" +
                                             path_ + "') GROUP BY region");
  ASSERT_TRUE(result.elapsed_wall_seconds.has_value());
  ASSERT_TRUE(result.metadata_inspection_seconds.has_value());
  ASSERT_TRUE(result.gpu_execution_seconds.has_value());
  ASSERT_TRUE(result.device_to_host_seconds.has_value());
  ASSERT_TRUE(result.parquet_decoding_seconds.has_value());

  EXPECT_GE(*result.metadata_inspection_seconds, 0.0);
  EXPECT_GE(*result.gpu_execution_seconds, 0.0);
  EXPECT_GE(*result.device_to_host_seconds, 0.0);
  EXPECT_GE(*result.parquet_decoding_seconds, 0.0);
  // gpu_execution_seconds covers only operator-tree construction + the
  // open/next/close pull loop; elapsed_wall_seconds additionally covers
  // parsing/binding/planning/metadata inspection above it, so it must be
  // at least as large.
  EXPECT_GE(*result.elapsed_wall_seconds, *result.gpu_execution_seconds);
  // No natural host-to-device boundary exists in this architecture yet
  // (see query_engine_execute_gpu.cpp) -- must stay a documented null, not
  // an invented zero.
  EXPECT_FALSE(result.host_to_device_seconds.has_value());
}

// --- Phase 1: cross-backend parity (GPU vs. the Acero CPU backend) ---
// Both backends translate the exact same PhysicalPlanPtr tree
// (src/execution/operator_builder.cpp vs. src/execution_cpu/
// acero_query_executor.cpp); this is the genuine two-backend correctness
// check the CPU backend's own plan called for, not just "it runs" on
// either side alone.

// Compares column values only, not full RecordBatch/schema equality:
// cudf's aggregate kernels happen not to allocate a null-mask buffer when a
// particular result contains no nulls, so the GPU path's Arrow schema often
// reports a field as "not null" incidentally (a property of that one
// result, not a documented guarantee); Arrow Acero's hash aggregate kernels
// always allocate one, reporting "nullable" instead. Neither backend
// promises a specific nullability for aggregate outputs, so requiring
// schema equality here would fail on an accidental implementation detail
// unrelated to correctness -- values are the actual contract being tested.
void expect_double_columns_match(const arrow::RecordBatch& gpu_batch, const arrow::RecordBatch& cpu_batch,
                                 const std::string& column_name) {
  const auto gpu_column =
      std::static_pointer_cast<arrow::DoubleArray>(gpu_batch.GetColumnByName(column_name));
  const auto cpu_column =
      std::static_pointer_cast<arrow::DoubleArray>(cpu_batch.GetColumnByName(column_name));
  ASSERT_NE(gpu_column, nullptr);
  ASSERT_NE(cpu_column, nullptr);
  ASSERT_EQ(gpu_column->length(), cpu_column->length());
  for (std::int64_t i = 0; i < gpu_column->length(); ++i) {
    EXPECT_DOUBLE_EQ(gpu_column->Value(i), cpu_column->Value(i)) << "row " << i;
  }
}

TEST_F(QueryEngineExecuteTest, CpuBackendMatchesGpuBackendForFilterAndGroupedAggregate) {
  const std::string sql = "SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ +
                          "') WHERE event_date >= DATE '2026-01-01' GROUP BY region ORDER BY region";
  const QueryResult gpu_result = engine_.execute(sql);

  EngineConfig cpu_config = default_config();
  cpu_config.engine.backend = "cpu";
  const QueryEngine cpu_engine(cpu_config);
  const QueryResult cpu_result = cpu_engine.execute(sql);

  ASSERT_EQ(gpu_result.batches.size(), 1u);
  ASSERT_EQ(cpu_result.batches.size(), 1u);
  EXPECT_EQ(gpu_result.rows_returned, cpu_result.rows_returned);

  const arrow::RecordBatch& gpu_batch = *gpu_result.batches.front();
  const arrow::RecordBatch& cpu_batch = *cpu_result.batches.front();
  const auto gpu_region = std::static_pointer_cast<arrow::StringArray>(gpu_batch.GetColumnByName("region"));
  const auto cpu_region = std::static_pointer_cast<arrow::StringArray>(cpu_batch.GetColumnByName("region"));
  ASSERT_NE(gpu_region, nullptr);
  ASSERT_NE(cpu_region, nullptr);
  ASSERT_EQ(gpu_region->length(), cpu_region->length());
  for (std::int64_t i = 0; i < gpu_region->length(); ++i) {
    EXPECT_EQ(gpu_region->GetString(i), cpu_region->GetString(i)) << "row " << i;
  }
  expect_double_columns_match(gpu_batch, cpu_batch, "total");
}

TEST_F(QueryEngineExecuteTest, CpuBackendMatchesGpuBackendForCountStarAndLimit) {
  const std::string sql = "SELECT region, COUNT(*) AS n FROM read_parquet('" + path_ +
                          "') GROUP BY region ORDER BY n DESC LIMIT 1";
  const QueryResult gpu_result = engine_.execute(sql);

  EngineConfig cpu_config = default_config();
  cpu_config.engine.backend = "cpu";
  const QueryEngine cpu_engine(cpu_config);
  const QueryResult cpu_result = cpu_engine.execute(sql);

  ASSERT_EQ(gpu_result.batches.size(), 1u);
  ASSERT_EQ(cpu_result.batches.size(), 1u);
  const auto gpu_n =
      std::static_pointer_cast<arrow::Int64Array>(gpu_result.batches.front()->GetColumnByName("n"));
  const auto cpu_n =
      std::static_pointer_cast<arrow::Int64Array>(cpu_result.batches.front()->GetColumnByName("n"));
  ASSERT_NE(gpu_n, nullptr);
  ASSERT_NE(cpu_n, nullptr);
  ASSERT_EQ(gpu_n->length(), 1);
  ASSERT_EQ(cpu_n->length(), 1);
  EXPECT_EQ(gpu_n->Value(0), cpu_n->Value(0));
}

TEST_F(QueryEngineExecuteTest, SplitExecutionPathLeavesMetadataInspectionSecondsNull) {
  // The split entry point never does planning itself (see its doc comment
  // in query_engine.hpp) -- it cannot honestly report metadata_inspection_
  // seconds, so this must stay nullopt rather than 0.0.
  const std::string sql = "SELECT region FROM read_parquet('" + path_ + "')";
  const PhysicalPlanPtr physical = engine_.explain(sql);
  RmmEnvironment rmm_environment(default_config());
  const QueryResult result = engine_.execute(physical, rmm_environment);
  EXPECT_FALSE(result.metadata_inspection_seconds.has_value());
  ASSERT_TRUE(result.gpu_execution_seconds.has_value());
  EXPECT_GE(*result.gpu_execution_seconds, 0.0);
}

}  // namespace
}  // namespace kernellake
