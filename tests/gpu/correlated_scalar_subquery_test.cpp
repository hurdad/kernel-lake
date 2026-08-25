// End-to-end coverage for a correlated *scalar* subquery in WHERE (TPC-H
// Q17/Q2/Q20's shape) -- through the whole parse ->
// sql::rewrite_correlated_scalar_subqueries() -> bind -> plan -> optimize
// -> physical plan -> execution pipeline, on both backends. Unlike
// EXISTS/NOT EXISTS (semi_anti_join_test.cpp's own SemiAntiJoinOperator
// coverage), this decorrelates into an ordinary INNER JOIN against a
// synthesized, GROUP BY-aggregated derived table -- no new operator of
// its own, so this file exercises the *rewrite* and the JOIN-against-a-
// derived-table plumbing (join_subplans, in query_engine.cpp/
// logical_planner.cpp), reusing HashJoinOperator/HashAggregateOperator
// unchanged.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/api/query_engine.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

// TPC-H Q17's own shape: a single-column correlation
// (`l2.part_id = p.part_id`) against a single, non-joined inner FROM
// source -- the case that needs strip_table_qualifier()
// (subquery_resolver.cpp) since the synthesized derived table keeps that
// single source unjoined.
class CorrelatedScalarSubqueryQ17ShapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_corr_scalar_q17_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    lineitem_path_ = (dir_ / "lineitem.parquet").string();

    // part_id=1: qty {5, 10, 45}, avg=20, half-avg=10 -- only qty=5 is
    // below it. part_id=2: qty {1, 2, 27}, avg=10, half-avg=5 -- qty=1
    // and qty=2 are both below it.
    arrow::Int64Builder part_id_builder;
    arrow::Int64Builder qty_builder;
    const std::vector<std::int64_t> part_ids = {1, 1, 1, 2, 2, 2};
    const std::vector<std::int64_t> qtys = {5, 10, 45, 1, 2, 27};
    for (std::size_t i = 0; i < part_ids.size(); ++i) {
      ASSERT_TRUE(part_id_builder.Append(part_ids[i]).ok());
      ASSERT_TRUE(qty_builder.Append(qtys[i]).ok());
    }
    std::shared_ptr<arrow::Array> part_id_array, qty_array;
    ASSERT_TRUE(part_id_builder.Finish(&part_id_array).ok());
    ASSERT_TRUE(qty_builder.Finish(&qty_array).ok());
    const auto schema = arrow::schema(
        {arrow::field("part_id", arrow::int64(), false), arrow::field("qty", arrow::int64(), false)});
    const auto table = arrow::Table::Make(schema, {part_id_array, qty_array});
    auto sink = arrow::io::FileOutputStream::Open(lineitem_path_).ValueOrDie();
    ASSERT_TRUE(
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/6).ok());
  }

  void TearDown() override { fs::remove_all(dir_); }

  [[nodiscard]] std::string sql() const {
    return "SELECT l.part_id, l.qty FROM read_parquet('" + lineitem_path_ +
           "') AS l WHERE l.qty < (SELECT 0.5 * AVG(l2.qty) FROM read_parquet('" + lineitem_path_ +
           "') AS l2 WHERE l2.part_id = l.part_id) ORDER BY l.part_id, l.qty";
  }

  fs::path dir_;
  std::string lineitem_path_;
};

TEST_F(CorrelatedScalarSubqueryQ17ShapeTest, FiltersRowsBelowHalfTheGroupAverage) {
  QueryEngine engine(default_config());
  const QueryResult result = engine.execute(sql());

  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 3);
  const auto part_id_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("part_id"));
  const auto qty_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("qty"));
  ASSERT_NE(part_id_column, nullptr);
  ASSERT_NE(qty_column, nullptr);
  EXPECT_EQ(part_id_column->Value(0), 1);
  EXPECT_EQ(qty_column->Value(0), 5);
  EXPECT_EQ(part_id_column->Value(1), 2);
  EXPECT_EQ(qty_column->Value(1), 1);
  EXPECT_EQ(part_id_column->Value(2), 2);
  EXPECT_EQ(qty_column->Value(2), 2);
}

TEST_F(CorrelatedScalarSubqueryQ17ShapeTest, CpuBackendMatchesGpuBackend) {
  EngineConfig cpu_config = default_config();
  cpu_config.engine.backend = "cpu";
  const QueryEngine cpu_engine(cpu_config);
  const QueryResult cpu_result = cpu_engine.execute(sql());

  QueryEngine gpu_engine(default_config());
  const QueryResult gpu_result = gpu_engine.execute(sql());

  ASSERT_EQ(cpu_result.batches.size(), 1u);
  ASSERT_EQ(gpu_result.batches.size(), 1u);
  const arrow::RecordBatch& cpu_batch = *cpu_result.batches.front();
  const arrow::RecordBatch& gpu_batch = *gpu_result.batches.front();
  ASSERT_EQ(cpu_batch.num_rows(), gpu_batch.num_rows());
  const auto cpu_qty = std::static_pointer_cast<arrow::Int64Array>(cpu_batch.GetColumnByName("qty"));
  const auto gpu_qty = std::static_pointer_cast<arrow::Int64Array>(gpu_batch.GetColumnByName("qty"));
  ASSERT_NE(cpu_qty, nullptr);
  ASSERT_NE(gpu_qty, nullptr);
  for (std::int64_t i = 0; i < cpu_batch.num_rows(); ++i) {
    EXPECT_EQ(cpu_qty->Value(i), gpu_qty->Value(i)) << "row " << i;
  }
}

// TPC-H Q20's own shape: a *two-column* correlation
// (`l.part_id = ps.part_id AND l.supplier_id = ps.supplier_id`) -- the
// first column becomes the synthesized join step's own ON-clause key,
// the second an extra outer WHERE conjunct (the same "one key in ON, the
// rest as a post-join WHERE filter" idiom this project's Q9 already uses
// for a real two-column JOIN condition). Data is chosen so that grouping
// by `part_id` *alone* (ignoring `supplier_id`) gives a different, wrong
// answer for (part_id=1, supplier_id=100): seeing lineitem's other
// supplier_id=200 row (qty=100) inflates that group's average enough to
// flip the comparison the wrong way.
class CorrelatedScalarSubqueryQ20ShapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_corr_scalar_q20_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    lineitem_path_ = (dir_ / "lineitem.parquet").string();
    partsupp_path_ = (dir_ / "partsupp.parquet").string();

    {
      arrow::Int64Builder part_id_builder;
      arrow::Int64Builder supplier_id_builder;
      arrow::Int64Builder qty_builder;
      const std::vector<std::int64_t> part_ids = {1, 1, 2, 2};
      const std::vector<std::int64_t> supplier_ids = {100, 200, 100, 200};
      const std::vector<std::int64_t> qtys = {10, 100, 4, 60};
      for (std::size_t i = 0; i < part_ids.size(); ++i) {
        ASSERT_TRUE(part_id_builder.Append(part_ids[i]).ok());
        ASSERT_TRUE(supplier_id_builder.Append(supplier_ids[i]).ok());
        ASSERT_TRUE(qty_builder.Append(qtys[i]).ok());
      }
      std::shared_ptr<arrow::Array> part_id_array, supplier_id_array, qty_array;
      ASSERT_TRUE(part_id_builder.Finish(&part_id_array).ok());
      ASSERT_TRUE(supplier_id_builder.Finish(&supplier_id_array).ok());
      ASSERT_TRUE(qty_builder.Finish(&qty_array).ok());
      const auto schema = arrow::schema({arrow::field("part_id", arrow::int64(), false),
                                         arrow::field("supplier_id", arrow::int64(), false),
                                         arrow::field("qty", arrow::int64(), false)});
      const auto table = arrow::Table::Make(schema, {part_id_array, supplier_id_array, qty_array});
      auto sink = arrow::io::FileOutputStream::Open(lineitem_path_).ValueOrDie();
      ASSERT_TRUE(
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/4).ok());
    }
    {
      arrow::Int64Builder part_id_builder;
      arrow::Int64Builder supplier_id_builder;
      arrow::Int64Builder avail_builder;
      const std::vector<std::int64_t> part_ids = {1, 1, 2, 2};
      const std::vector<std::int64_t> supplier_ids = {100, 200, 100, 200};
      const std::vector<std::int64_t> avails = {6, 200, 1, 100};
      for (std::size_t i = 0; i < part_ids.size(); ++i) {
        ASSERT_TRUE(part_id_builder.Append(part_ids[i]).ok());
        ASSERT_TRUE(supplier_id_builder.Append(supplier_ids[i]).ok());
        ASSERT_TRUE(avail_builder.Append(avails[i]).ok());
      }
      std::shared_ptr<arrow::Array> part_id_array, supplier_id_array, avail_array;
      ASSERT_TRUE(part_id_builder.Finish(&part_id_array).ok());
      ASSERT_TRUE(supplier_id_builder.Finish(&supplier_id_array).ok());
      ASSERT_TRUE(avail_builder.Finish(&avail_array).ok());
      const auto schema = arrow::schema({arrow::field("part_id", arrow::int64(), false),
                                         arrow::field("supplier_id", arrow::int64(), false),
                                         arrow::field("avail", arrow::int64(), false)});
      const auto table = arrow::Table::Make(schema, {part_id_array, supplier_id_array, avail_array});
      auto sink = arrow::io::FileOutputStream::Open(partsupp_path_).ValueOrDie();
      ASSERT_TRUE(
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/4).ok());
    }
  }

  void TearDown() override { fs::remove_all(dir_); }

  [[nodiscard]] std::string sql() const {
    return "SELECT ps.part_id, ps.supplier_id FROM read_parquet('" + partsupp_path_ +
           "') AS ps WHERE ps.avail > (SELECT 0.5 * SUM(l.qty) FROM read_parquet('" + lineitem_path_ +
           "') AS l WHERE l.part_id = ps.part_id AND l.supplier_id = ps.supplier_id) ORDER BY "
           "ps.part_id, ps.supplier_id";
  }

  fs::path dir_;
  std::string lineitem_path_;
  std::string partsupp_path_;
};

TEST_F(CorrelatedScalarSubqueryQ20ShapeTest, TwoColumnCorrelationMatchesPerSupplierNotPerPartAverage) {
  QueryEngine engine(default_config());
  const QueryResult result = engine.execute(sql());

  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  // (1,100): avail=6 > 0.5*10=5 -- passes. (1,200): avail=200 > 0.5*100=50
  // -- passes. (2,100): avail=1 > 0.5*4=2 -- fails. (2,200): avail=100 >
  // 0.5*60=30 -- passes. If supplier_id were ignored (grouped by part_id
  // alone), (1,100) would instead see part 1's *combined* qty (10+100=110,
  // half=55) and wrongly fail (6 > 55 is false).
  ASSERT_EQ(batch->num_rows(), 3);
  const auto part_id_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("part_id"));
  const auto supplier_id_column =
      std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("supplier_id"));
  ASSERT_NE(part_id_column, nullptr);
  ASSERT_NE(supplier_id_column, nullptr);
  EXPECT_EQ(part_id_column->Value(0), 1);
  EXPECT_EQ(supplier_id_column->Value(0), 100);
  EXPECT_EQ(part_id_column->Value(1), 1);
  EXPECT_EQ(supplier_id_column->Value(1), 200);
  EXPECT_EQ(part_id_column->Value(2), 2);
  EXPECT_EQ(supplier_id_column->Value(2), 200);
}

TEST_F(CorrelatedScalarSubqueryQ20ShapeTest, CpuBackendMatchesGpuBackend) {
  EngineConfig cpu_config = default_config();
  cpu_config.engine.backend = "cpu";
  const QueryEngine cpu_engine(cpu_config);
  const QueryResult cpu_result = cpu_engine.execute(sql());

  QueryEngine gpu_engine(default_config());
  const QueryResult gpu_result = gpu_engine.execute(sql());

  ASSERT_EQ(cpu_result.batches.size(), 1u);
  ASSERT_EQ(gpu_result.batches.size(), 1u);
  const arrow::RecordBatch& cpu_batch = *cpu_result.batches.front();
  const arrow::RecordBatch& gpu_batch = *gpu_result.batches.front();
  ASSERT_EQ(cpu_batch.num_rows(), gpu_batch.num_rows());
  const auto cpu_part_id = std::static_pointer_cast<arrow::Int64Array>(cpu_batch.GetColumnByName("part_id"));
  const auto gpu_part_id = std::static_pointer_cast<arrow::Int64Array>(gpu_batch.GetColumnByName("part_id"));
  ASSERT_NE(cpu_part_id, nullptr);
  ASSERT_NE(gpu_part_id, nullptr);
  for (std::int64_t i = 0; i < cpu_batch.num_rows(); ++i) {
    EXPECT_EQ(cpu_part_id->Value(i), gpu_part_id->Value(i)) << "row " << i;
  }
}

// TPC-H Q2's own shape: the correlated subquery's own FROM is a
// multi-way JOIN chain (not a single source), and it keeps its own
// additional (non-correlation) WHERE conjunct -- exercises try_decorrelate()
// moving `inner.join` onto the derived table as-is and leaving a
// remaining conjunct on the derived table's own WHERE clause, both paths
// CorrelatedScalarSubqueryQ17ShapeTest's single-source shape never
// reaches.
class CorrelatedScalarSubqueryMultiWayJoinShapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_corr_scalar_multijoin_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    partsupp_path_ = (dir_ / "partsupp.parquet").string();
    supplier_path_ = (dir_ / "supplier.parquet").string();

    // partsupp: part_id 1 has two suppliers (cost 10, 20); part_id 2 has
    // one supplier (cost 5) in region 'EU' and one excluded by the
    // supplier-region filter below (cost 1, region 'US').
    {
      arrow::Int64Builder part_id_builder;
      arrow::Int64Builder supplier_id_builder;
      arrow::DoubleBuilder cost_builder;
      const std::vector<std::int64_t> part_ids = {1, 1, 2, 2};
      const std::vector<std::int64_t> supplier_ids = {10, 20, 30, 40};
      const std::vector<double> costs = {10.0, 20.0, 5.0, 1.0};
      for (std::size_t i = 0; i < part_ids.size(); ++i) {
        ASSERT_TRUE(part_id_builder.Append(part_ids[i]).ok());
        ASSERT_TRUE(supplier_id_builder.Append(supplier_ids[i]).ok());
        ASSERT_TRUE(cost_builder.Append(costs[i]).ok());
      }
      std::shared_ptr<arrow::Array> part_id_array, supplier_id_array, cost_array;
      ASSERT_TRUE(part_id_builder.Finish(&part_id_array).ok());
      ASSERT_TRUE(supplier_id_builder.Finish(&supplier_id_array).ok());
      ASSERT_TRUE(cost_builder.Finish(&cost_array).ok());
      const auto schema = arrow::schema({arrow::field("part_id", arrow::int64(), false),
                                         arrow::field("supplier_id", arrow::int64(), false),
                                         arrow::field("cost", arrow::float64(), false)});
      const auto table = arrow::Table::Make(schema, {part_id_array, supplier_id_array, cost_array});
      auto sink = arrow::io::FileOutputStream::Open(partsupp_path_).ValueOrDie();
      ASSERT_TRUE(
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/4).ok());
    }
    {
      arrow::Int64Builder supplier_id_builder;
      arrow::StringBuilder region_builder;
      const std::vector<std::int64_t> supplier_ids = {10, 20, 30, 40};
      const std::vector<std::string> regions = {"EU", "EU", "EU", "US"};
      for (std::size_t i = 0; i < supplier_ids.size(); ++i) {
        ASSERT_TRUE(supplier_id_builder.Append(supplier_ids[i]).ok());
        ASSERT_TRUE(region_builder.Append(regions[i]).ok());
      }
      std::shared_ptr<arrow::Array> supplier_id_array, region_array;
      ASSERT_TRUE(supplier_id_builder.Finish(&supplier_id_array).ok());
      ASSERT_TRUE(region_builder.Finish(&region_array).ok());
      const auto schema = arrow::schema(
          {arrow::field("supplier_id", arrow::int64(), false), arrow::field("region", arrow::utf8(), false)});
      const auto table = arrow::Table::Make(schema, {supplier_id_array, region_array});
      auto sink = arrow::io::FileOutputStream::Open(supplier_path_).ValueOrDie();
      ASSERT_TRUE(
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/4).ok());
    }
  }

  void TearDown() override { fs::remove_all(dir_); }

  [[nodiscard]] std::string sql() const {
    return "SELECT ps.part_id, ps.supplier_id FROM read_parquet('" + partsupp_path_ +
           "') AS ps WHERE ps.cost = (SELECT MIN(ps2.cost) FROM read_parquet('" + partsupp_path_ +
           "') AS ps2 JOIN read_parquet('" + supplier_path_ +
           "') AS s2 ON s2.supplier_id = ps2.supplier_id WHERE ps2.part_id = ps.part_id AND s2.region = "
           "'EU') ORDER BY ps.part_id, ps.supplier_id";
  }

  fs::path dir_;
  std::string partsupp_path_;
  std::string supplier_path_;
};

TEST_F(CorrelatedScalarSubqueryMultiWayJoinShapeTest, KeepsOnlyTheCheapestEuSupplierPerPart) {
  QueryEngine engine(default_config());
  const QueryResult result = engine.execute(sql());

  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  // part 1: cheapest EU supplier is 10 (cost 10). part 2: supplier 40 is
  // excluded by the region filter (US), leaving only supplier 30 (cost
  // 5) as both the EU minimum and the only surviving row.
  ASSERT_EQ(batch->num_rows(), 2);
  const auto part_id_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("part_id"));
  const auto supplier_id_column =
      std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("supplier_id"));
  ASSERT_NE(part_id_column, nullptr);
  ASSERT_NE(supplier_id_column, nullptr);
  EXPECT_EQ(part_id_column->Value(0), 1);
  EXPECT_EQ(supplier_id_column->Value(0), 10);
  EXPECT_EQ(part_id_column->Value(1), 2);
  EXPECT_EQ(supplier_id_column->Value(1), 30);
}

TEST_F(CorrelatedScalarSubqueryMultiWayJoinShapeTest, CpuBackendMatchesGpuBackend) {
  EngineConfig cpu_config = default_config();
  cpu_config.engine.backend = "cpu";
  const QueryEngine cpu_engine(cpu_config);
  const QueryResult cpu_result = cpu_engine.execute(sql());

  QueryEngine gpu_engine(default_config());
  const QueryResult gpu_result = gpu_engine.execute(sql());

  ASSERT_EQ(cpu_result.batches.size(), 1u);
  ASSERT_EQ(gpu_result.batches.size(), 1u);
  const arrow::RecordBatch& cpu_batch = *cpu_result.batches.front();
  const arrow::RecordBatch& gpu_batch = *gpu_result.batches.front();
  ASSERT_EQ(cpu_batch.num_rows(), gpu_batch.num_rows());
  const auto cpu_supplier_id =
      std::static_pointer_cast<arrow::Int64Array>(cpu_batch.GetColumnByName("supplier_id"));
  const auto gpu_supplier_id =
      std::static_pointer_cast<arrow::Int64Array>(gpu_batch.GetColumnByName("supplier_id"));
  ASSERT_NE(cpu_supplier_id, nullptr);
  ASSERT_NE(gpu_supplier_id, nullptr);
  for (std::int64_t i = 0; i < cpu_batch.num_rows(); ++i) {
    EXPECT_EQ(cpu_supplier_id->Value(i), gpu_supplier_id->Value(i)) << "row " << i;
  }
}

}  // namespace
}  // namespace kernellake
