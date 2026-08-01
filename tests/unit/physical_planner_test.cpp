#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/optimizer/optimizer.hpp"
#include "kernellake/planner/logical_planner.hpp"
#include "kernellake/sql/parser.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class PhysicalPlannerTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_physical_planner_test_" +
                     std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();
    write_two_row_group_file(path_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  // Row group 0: id 0-4, region "A". Row group 1: id 5-9, region "B".
  static void write_two_row_group_file(const std::string& path) {
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
    auto sink = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  Schema sales_schema() {
    return Schema({Field{"id", int64_type(false)}, Field{"amount", float64_type(false)},
                    Field{"region", string_type(false)}});
  }

  PhysicalPlanPtr plan_for(const std::string& sql) {
    const auto stmt = sql::parse_sql(sql);
    const BoundQuery bound = bind_query(stmt, sales_schema());
    LogicalPlanPtr logical = build_logical_plan(bound, sales_schema());
    logical = optimize(std::move(logical));
    return build_physical_plan(logical, store_);
  }

  static const PhysicalPlanNode* find_leaf(const PhysicalPlanNode* node) {
    while (!node->children().empty()) node = node->children()[0].get();
    return node;
  }

  fs::path dir_;
  std::string path_;
  LocalObjectStore store_;
};

TEST_F(PhysicalPlannerTest, BuildsScanWithNarrowedColumnsAndSchema) {
  const PhysicalPlanPtr plan =
      plan_for("SELECT region FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const auto* projection = dynamic_cast<const ProjectionNode*>(result->child().get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const FilterNode*>(projection->child().get());
  ASSERT_NE(filter, nullptr);
  const auto* scan = dynamic_cast<const ParquetScanNode*>(filter->child().get());
  ASSERT_NE(scan, nullptr);

  // Only "region" is needed (the WHERE and SELECT only ever touch region).
  ASSERT_EQ(scan->columns().size(), 1u);
  EXPECT_EQ(scan->columns()[0], "region");
  ASSERT_EQ(scan->output_schema().field_count(), 1u);
  EXPECT_EQ(scan->output_schema().field(0).name, "region");
}

TEST_F(PhysicalPlannerTest, ExpressionsAboveScanUseNarrowedColumnIndicesNotOriginalOnes) {
  // "region" is field index 2 in sales_schema(), but column pruning narrows
  // the physical scan to just this one column, putting it at index 0 in the
  // batches the scan operator will actually produce. Every ColumnExpression
  // above the scan (here, the filter predicate and the projection item) must
  // be rewritten to that narrowed index -- otherwise the execution layer
  // would index into a column that doesn't exist in the pruned batch.
  const PhysicalPlanPtr plan =
      plan_for("SELECT region FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const auto* projection = dynamic_cast<const ProjectionNode*>(result->child().get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const FilterNode*>(projection->child().get());
  ASSERT_NE(filter, nullptr);

  const auto* comparison = dynamic_cast<const BinaryExpression*>(filter->predicate().get());
  ASSERT_NE(comparison, nullptr);
  const auto* predicate_column = dynamic_cast<const ColumnExpression*>(comparison->left().get());
  ASSERT_NE(predicate_column, nullptr);
  EXPECT_EQ(predicate_column->column_index(), 0u);

  ASSERT_EQ(projection->items().size(), 1u);
  const auto* projection_column = dynamic_cast<const ColumnExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(projection_column, nullptr);
  EXPECT_EQ(projection_column->column_index(), 0u);
}

TEST_F(PhysicalPlannerTest, AggregateArgumentUsesNarrowedColumnIndexWhenEarlierColumnsAreDropped) {
  // "amount" is field index 1 in sales_schema(); selecting only SUM(amount)
  // drops "id" (index 0), so the narrowed scan schema puts "amount" at
  // index 0 -- while the binder-assigned index on the AggregateExpression's
  // argument is still 1 until the physical planner remaps it.
  const PhysicalPlanPtr plan = plan_for("SELECT SUM(amount) AS total FROM read_parquet('" + path_ + "')");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const auto* scalar = dynamic_cast<const ScalarAggregateNode*>(result->child().get());
  ASSERT_NE(scalar, nullptr);
  ASSERT_EQ(scalar->aggregates().size(), 1u);
  const auto* aggregate = dynamic_cast<const AggregateExpression*>(scalar->aggregates()[0].expr.get());
  ASSERT_NE(aggregate, nullptr);
  const auto* argument_column = dynamic_cast<const ColumnExpression*>(aggregate->argument().get());
  ASSERT_NE(argument_column, nullptr);
  EXPECT_EQ(argument_column->column_index(), 0u);
}

TEST_F(PhysicalPlannerTest, PruningSkipsWholeRowGroupAtPhysicalLevel) {
  const PhysicalPlanPtr plan =
      plan_for("SELECT id FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  const auto* scan = dynamic_cast<const ParquetScanNode*>(find_leaf(plan.get()));
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->files_scanned(), 1u);
  ASSERT_EQ(scan->fragments().size(), 1u);
  const PhysicalFileFragment& fragment = scan->fragments()[0];
  EXPECT_EQ(fragment.selected_row_groups, (std::vector<int>{1}));
  EXPECT_EQ(fragment.skipped_row_groups, (std::vector<int>{0}));
  EXPECT_FALSE(fragment.pruning_reasons.empty());
}

TEST_F(PhysicalPlannerTest, ScalarAggregateForNoGroupBy) {
  const PhysicalPlanPtr plan = plan_for("SELECT SUM(amount) AS total FROM read_parquet('" + path_ + "')");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  // Aggregate reprojection may or may not survive redundant-projection
  // removal; walk down to find the aggregate node either way.
  const PhysicalPlanNode* node = result->child().get();
  while (dynamic_cast<const ScalarAggregateNode*>(node) == nullptr && !node->children().empty()) {
    node = node->children()[0].get();
  }
  const auto* scalar = dynamic_cast<const ScalarAggregateNode*>(node);
  ASSERT_NE(scalar, nullptr);
  ASSERT_EQ(scalar->aggregates().size(), 1u);
  EXPECT_EQ(scalar->aggregates()[0].name, "total");
}

TEST_F(PhysicalPlannerTest, HashAggregateForGroupBy) {
  const PhysicalPlanPtr plan = plan_for(
      "SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ + "') GROUP BY region");
  const PhysicalPlanNode* node = plan.get();
  while (dynamic_cast<const HashAggregateNode*>(node) == nullptr && !node->children().empty()) {
    node = node->children()[0].get();
  }
  const auto* hash_agg = dynamic_cast<const HashAggregateNode*>(node);
  ASSERT_NE(hash_agg, nullptr);
  ASSERT_EQ(hash_agg->group_by().size(), 1u);
  ASSERT_EQ(hash_agg->aggregates().size(), 1u);
}

TEST_F(PhysicalPlannerTest, RejectsOrderByWithClearError) {
  const auto stmt =
      sql::parse_sql("SELECT id FROM read_parquet('" + path_ + "') ORDER BY id");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  LogicalPlanPtr logical = optimize(build_logical_plan(bound, sales_schema()));
  EXPECT_THROW(build_physical_plan(logical, store_), PlanningError);
}

TEST_F(PhysicalPlannerTest, ExplainShowsFilesAndRowGroupCounts) {
  const PhysicalPlanPtr plan =
      plan_for("SELECT id FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("ArrowResult"), std::string::npos);
  EXPECT_NE(text.find("ParquetScan"), std::string::npos);
  EXPECT_NE(text.find("files: 1/1"), std::string::npos);
  EXPECT_NE(text.find("row_groups: 1/2"), std::string::npos);
}

}  // namespace
}  // namespace kernellake
