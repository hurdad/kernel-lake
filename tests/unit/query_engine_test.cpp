#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class QueryEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_engine_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();

    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::StringBuilder region_builder;
    for (int64_t i = 0; i < 10; ++i) {
      ASSERT_TRUE(id_builder.Append(i).ok());
      ASSERT_TRUE(amount_builder.Append(static_cast<double>(i)).ok());
      ASSERT_TRUE(region_builder.Append(i < 5 ? "A" : "B").ok());
    }
    std::shared_ptr<arrow::Array> id_array, amount_array, region_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false),
                                       arrow::field("amount", arrow::float64(), false),
                                       arrow::field("region", arrow::utf8(), false)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array, region_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
  QueryEngine engine_{default_config()};
};

TEST_F(QueryEngineTest, ExplainLogicalBindsAgainstRealParquetSchema) {
  const LogicalPlanPtr plan = engine_.explain_logical(
      "SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ + "') GROUP BY region");
  // The reprojection the planner adds on top of LogicalAggregate happens to
  // be an identity here (SELECT order already matches [group_by...,
  // aggregates...]) -- but identity projections are no longer elided at
  // this (logical, pre-column-pruning) stage, since doing so used to make
  // annotate_scan() silently lose track of which columns a query's actual
  // output needs (see optimizer.cpp's rewrite_plan() LogicalProjection
  // comment). The equivalent optimization now happens later, in
  // physical_planner.cpp, once column pruning has already run -- so
  // LogicalProjection always survives here.
  EXPECT_EQ(plan->node_name(), "LogicalProjection");
  ASSERT_EQ(plan->output_schema().field_count(), 2u);
  EXPECT_EQ(plan->output_schema().field(0).name, "region");
  EXPECT_EQ(plan->output_schema().field(1).name, "total");
}

TEST_F(QueryEngineTest, ExplainProducesFullPhysicalPlanWithPruning) {
  const PhysicalPlanPtr plan =
      engine_.explain("SELECT id FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("ArrowResult"), std::string::npos);
  EXPECT_NE(text.find("ParquetScan"), std::string::npos);
  EXPECT_NE(text.find("row_groups: 1/2"), std::string::npos);
}

TEST_F(QueryEngineTest, ExplainRejectsUnknownColumnWithBindingError) {
  EXPECT_THROW((void)(engine_.explain_logical("SELECT nonexistent FROM read_parquet('" + path_ + "')")),
               BindingError);
}

#ifdef KERNELLAKE_WITH_CUDA
// GPU-enabled build: execute() actually runs the query -- correctness is
// covered in depth by tests/gpu/query_engine_execute_test.cpp, but a smoke
// test belongs here too since this suite runs on every build.
TEST_F(QueryEngineTest, ExecuteReturnsRowsOnGpuBuild) {
  const QueryResult result = engine_.execute("SELECT id FROM read_parquet('" + path_ + "')");
  EXPECT_EQ(result.rows_returned, 10);
}
#else
// CPU-only build: no GPU execution layer is linked in, so execute() must
// say so clearly rather than silently doing nothing or crashing.
TEST_F(QueryEngineTest, ExecuteThrowsClearExecutionError) {
  EXPECT_THROW((void)(engine_.execute("SELECT id FROM read_parquet('" + path_ + "')")), ExecutionError);
}
#endif

}  // namespace
}  // namespace kernellake
