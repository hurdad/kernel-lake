#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <filesystem>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
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
  // Selecting the single column that's also all the scan was pruned down to
  // is a genuine identity projection relative to the (already-narrowed)
  // scan schema, so it's elided here (see
  // PhysicalPlannerTest.ElidesIdentityProjectionAfterColumnPruning) --
  // result's child is the Filter directly, no ProjectionNode in between.
  const auto* filter = dynamic_cast<const FilterNode*>(result->child().get());
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
  // batches the scan operator will actually produce. The filter predicate
  // above the scan must be rewritten to that narrowed index -- otherwise
  // the execution layer would index into a column that doesn't exist in
  // the pruned batch. (The projection that used to sit here too has been
  // elided -- selecting the scan's one pruned column is a genuine identity
  // projection relative to it; see
  // SurvivingPlainProjectionRemapsAgainstTheNarrowedScanSchema below for a
  // projection-remapping test using a shape where it survives.)
  const PhysicalPlanPtr plan =
      plan_for("SELECT region FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const auto* filter = dynamic_cast<const FilterNode*>(result->child().get());
  ASSERT_NE(filter, nullptr);

  const auto* comparison = dynamic_cast<const BinaryExpression*>(filter->predicate().get());
  ASSERT_NE(comparison, nullptr);
  const auto* predicate_column = dynamic_cast<const ColumnExpression*>(comparison->left().get());
  ASSERT_NE(predicate_column, nullptr);
  EXPECT_EQ(predicate_column->column_index(), 0u);
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
  const PhysicalPlanPtr plan = plan_for("SELECT id FROM read_parquet('" + path_ + "') WHERE region = 'B'");
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
  const PhysicalPlanPtr plan =
      plan_for("SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ + "') GROUP BY region");
  const PhysicalPlanNode* node = plan.get();
  while (dynamic_cast<const HashAggregateNode*>(node) == nullptr && !node->children().empty()) {
    node = node->children()[0].get();
  }
  const auto* hash_agg = dynamic_cast<const HashAggregateNode*>(node);
  ASSERT_NE(hash_agg, nullptr);
  ASSERT_EQ(hash_agg->group_by().size(), 1u);
  ASSERT_EQ(hash_agg->aggregates().size(), 1u);
}

TEST_F(PhysicalPlannerTest, BuildsSortNodeForOrderBy) {
  const auto stmt = sql::parse_sql("SELECT amount FROM read_parquet('" + path_ + "') ORDER BY amount DESC");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  LogicalPlanPtr logical = optimize(build_logical_plan(bound, sales_schema()));
  const PhysicalPlanPtr plan = build_physical_plan(logical, store_);

  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const PhysicalPlanNode* node = result->child().get();
  while (dynamic_cast<const SortNode*>(node) == nullptr && !node->children().empty()) {
    node = node->children()[0].get();
  }
  const auto* sort = dynamic_cast<const SortNode*>(node);
  ASSERT_NE(sort, nullptr);
  ASSERT_EQ(sort->keys().size(), 1u);
  EXPECT_FALSE(sort->keys()[0].ascending);
  const auto* key_column = dynamic_cast<const ColumnExpression*>(sort->keys()[0].expr.get());
  ASSERT_NE(key_column, nullptr);
  // "amount" is field index 1 in sales_schema(), but it's the only column
  // selected/ordered by, so column pruning narrows the scan to just it,
  // putting it at index 0 in the batches the scan actually produces -- the
  // sort key's index must have been remapped to match, not left at 1.
  EXPECT_EQ(key_column->column_index(), 0u);
}

TEST_F(PhysicalPlannerTest, SurvivingAggregateReprojectionIsNotRemappedAgainstTheScan) {
  // SELECT order [total, region] differs from HashAggregate's natural
  // [group_by..., aggregates...] = [region, total] output order, so the
  // optimizer's redundant-projection-removal rule can't remove the
  // reprojection here (unlike most other tests in this file, where it
  // happens to). This is the exact shape that exposed a real bug: the
  // reprojection's items reference the *aggregate's* output schema
  // (region@0, total@1) already correctly, by construction in
  // logical_planner.cpp -- remapping them against the scan (which has no
  // "total" column at all) must NOT happen.
  const PhysicalPlanPtr plan =
      plan_for("SELECT SUM(amount) AS total, region FROM read_parquet('" + path_ + "') GROUP BY region");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const auto* projection = dynamic_cast<const ProjectionNode*>(result->child().get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 2u);

  // logical_planner.cpp's reprojection always builds a plain
  // ColumnExpression referencing the aggregate's output position, even for
  // an aggregate-derived column like "total" -- never re-wraps the
  // original AggregateExpression. HashAggregate's own output order is
  // [group_by..., aggregates...] = [region@0, total@1].
  const auto* total_column = dynamic_cast<const ColumnExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(total_column->column_index(), 1u);

  const auto* region_column = dynamic_cast<const ColumnExpression*>(projection->items()[1].expr.get());
  ASSERT_NE(region_column, nullptr);
  EXPECT_EQ(region_column->column_index(), 0u);
}

TEST_F(PhysicalPlannerTest, SurvivingPlainProjectionRemapsAgainstTheNarrowedScanSchema) {
  // convert_scan() narrows the physical scan to required_columns() while
  // preserving the scan's *original* field order (id, amount, region --
  // see that function's own comment); pruned to {amount, region} (id is
  // never referenced) that's [amount@0, region@1]. Selecting them back out
  // in the *reverse* of that ("region, amount") means this projection is a
  // genuine reordering, not an identity relative to the scan, so unlike
  // the single-column tests above it survives and DOES need remapping.
  const PhysicalPlanPtr plan =
      plan_for("SELECT region, amount FROM read_parquet('" + path_ + "') WHERE amount > 0");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const auto* projection = dynamic_cast<const ProjectionNode*>(result->child().get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 2u);

  // Scan is narrowed to [amount, region] (original relative order
  // preserved), so the reordering projection maps region -> scan index 1,
  // amount -> index 0.
  const auto* region_column = dynamic_cast<const ColumnExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(region_column, nullptr);
  EXPECT_EQ(region_column->column_index(), 1u);
  const auto* amount_column = dynamic_cast<const ColumnExpression*>(projection->items()[1].expr.get());
  ASSERT_NE(amount_column, nullptr);
  EXPECT_EQ(amount_column->column_index(), 0u);
}

TEST_F(PhysicalPlannerTest, AggregateOrderBySurvivesWhenReprojectionIsNotRemoved) {
  // Combines both fixes: a surviving aggregate reprojection AND a Sort
  // directly above it (not above LogicalAggregate), which is the shape
  // that broke before the LogicalProjection discriminator fix in
  // physical_planner.cpp.
  const PhysicalPlanPtr plan = plan_for("SELECT SUM(amount) AS total, region FROM read_parquet('" + path_ +
                                        "') GROUP BY region ORDER BY total");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  const auto* sort = dynamic_cast<const SortNode*>(result->child().get());
  ASSERT_NE(sort, nullptr);
  ASSERT_EQ(sort->keys().size(), 1u);
  const auto* key_column = dynamic_cast<const ColumnExpression*>(sort->keys()[0].expr.get());
  ASSERT_NE(key_column, nullptr);
  // "total" is the reprojection's output position 0.
  EXPECT_EQ(key_column->column_index(), 0u);
  const auto* projection = dynamic_cast<const ProjectionNode*>(sort->child().get());
  ASSERT_NE(projection, nullptr);
}

// Regression test for a real bug: `SELECT id, amount` selects sales_schema()
// in its exact original order (id=0, amount=1), which used to make the
// whole LogicalProjection look like a no-op identity projection and get
// elided in optimizer.cpp's rewrite_plan() -- before annotate_scan()'s
// column-pruning pass ever ran over the rewritten tree. With the projection
// gone, the WHERE clause's Filter was the only thing left annotate_scan
// could see, so the scan's required columns silently narrowed to just
// "id", dropping "amount" from the query's actual output with no error at
// all. Confirmed for real against a running kernellake-server (both CPU and
// GPU backends): `SELECT id, amount FROM t WHERE id < 3` returned rows
// containing only the `id` value. Fixed by moving the identity-projection
// elision from optimizer.cpp (before pruning) to physical_planner.cpp
// (after pruning) -- see ElidesIdentityProjectionAfterColumnPruning below
// for the case where eliding is still safe.
TEST_F(PhysicalPlannerTest, RegressionKeepsEveryColumnWhenSelectListMatchesSchemaOrder) {
  const PhysicalPlanPtr plan = plan_for("SELECT id, amount FROM read_parquet('" + path_ + "') WHERE id < 3");
  const auto* scan = dynamic_cast<const ParquetScanNode*>(find_leaf(plan.get()));
  ASSERT_NE(scan, nullptr);
  const std::vector<std::string>& columns = scan->columns();
  ASSERT_EQ(columns.size(), 2u);
  EXPECT_NE(std::find(columns.begin(), columns.end(), "id"), columns.end());
  EXPECT_NE(std::find(columns.begin(), columns.end(), "amount"), columns.end());
}

// Companion test: convert_scan() narrows the scan to required_columns()
// while preserving the scan's *original* field order (see
// physical_planner.cpp's own comment on that -- required_columns() itself
// is stored alphabetically sorted, but that's just an internal detail of
// how it's collected, not the physical scan's actual output order). So
// selecting every column in the table's original schema order (with
// nothing to prune, since all three are used) makes this projection a
// genuine identity relative to the scan's own output schema. This really
// is safe to elide, confirming the optimization itself (skip a
// pass-through-only ProjectionNode) still works, just moved to
// physical_planner.cpp so it runs after column pruning instead of before.
TEST_F(PhysicalPlannerTest, ElidesIdentityProjectionAfterColumnPruning) {
  const PhysicalPlanPtr plan = plan_for("SELECT id, amount, region FROM read_parquet('" + path_ + "')");
  const auto* result = dynamic_cast<const ArrowResultNode*>(plan.get());
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(dynamic_cast<const ProjectionNode*>(result->child().get()), nullptr);
  EXPECT_NE(dynamic_cast<const ParquetScanNode*>(result->child().get()), nullptr);
}

TEST_F(PhysicalPlannerTest, ExplainShowsFilesAndRowGroupCounts) {
  const PhysicalPlanPtr plan = plan_for("SELECT id FROM read_parquet('" + path_ + "') WHERE region = 'B'");
  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("ArrowResult"), std::string::npos);
  EXPECT_NE(text.find("ParquetScan"), std::string::npos);
  EXPECT_NE(text.find("files: 1/1"), std::string::npos);
  EXPECT_NE(text.find("row_groups: 1/2"), std::string::npos);
}

// A key/single-value table with `row_count` rows, one row group -- used by
// the build-side-selection tests below, where the two joined tables' row
// counts (not their column shapes) are what matters.
void write_key_value_file(const std::string& path, const std::string& value_column, int64_t row_count) {
  arrow::Int64Builder key_builder;
  arrow::DoubleBuilder value_builder;
  for (int64_t i = 0; i < row_count; ++i) {
    ASSERT_TRUE(key_builder.Append(i % 3).ok());
    ASSERT_TRUE(value_builder.Append(static_cast<double>(i)).ok());
  }
  std::shared_ptr<arrow::Array> key_array, value_array;
  ASSERT_TRUE(key_builder.Finish(&key_array).ok());
  ASSERT_TRUE(value_builder.Finish(&value_array).ok());
  const auto schema = arrow::schema(
      {arrow::field("jkey", arrow::int64(), false), arrow::field(value_column, arrow::float64(), false)});
  const auto table = arrow::Table::Make(schema, {key_array, value_array});
  auto sink = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  const arrow::Status status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink,
                                                          /*chunk_size=*/row_count);
  ASSERT_TRUE(status.ok()) << status.ToString();
}

class PhysicalPlannerJoinBuildSideTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_physical_planner_join_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    small_path_ = (dir_ / "small.parquet").string();
    large_path_ = (dir_ / "large.parquet").string();
    write_key_value_file(small_path_, "sval", 3);
    write_key_value_file(large_path_, "lval", 20);
  }

  void TearDown() override { fs::remove_all(dir_); }

  PhysicalPlanPtr plan_for_join(const std::string& sql) {
    const sql::AstSelectStatement stmt = sql::parse_sql(sql);
    std::vector<Schema> join_schemas;
    for (const std::string& path :
         {stmt.join->first.paths.front(), stmt.join->steps.front().source.paths.front()}) {
      const FileMetadata metadata = inspect_parquet_file(store_, Uri{path});
      join_schemas.push_back(metadata.schema);
    }
    const BoundQuery bound = bind_query(stmt, join_schemas);
    LogicalPlanPtr logical = build_logical_plan(bound, join_schemas);
    logical = optimize(std::move(logical));
    return build_physical_plan(logical, store_);
  }

  static const HashJoinNode* find_hash_join(const PhysicalPlanNode* node) {
    if (const auto* join = dynamic_cast<const HashJoinNode*>(node)) {
      return join;
    }
    for (const PhysicalPlanPtr& child : node->children()) {
      if (const HashJoinNode* found = find_hash_join(child.get())) {
        return found;
      }
    }
    return nullptr;
  }

  fs::path dir_;
  std::string small_path_;
  std::string large_path_;
  LocalObjectStore store_;
};

TEST_F(PhysicalPlannerJoinBuildSideTest, BuildsOnSmallerSideEvenWhenItIsWrittenFirstInTheQuery) {
  // small (3 rows) is written *first*/left in the FROM clause, large (20
  // rows) second/right -- without the size-aware swap, HashJoinOperator
  // would build its hash table on `large` (the right/build side by
  // convention), the opposite of what performs well. The fix should swap
  // so `right()` ends up holding the smaller table regardless of clause
  // order.
  const PhysicalPlanPtr plan =
      plan_for_join("SELECT sval, lval FROM read_parquet('" + small_path_ + "') AS s JOIN read_parquet('" +
                    large_path_ + "') AS l ON s.jkey = l.jkey");
  const HashJoinNode* join = find_hash_join(plan.get());
  ASSERT_NE(join, nullptr);
  EXPECT_NE(join->right()->output_schema().find_field("sval"), std::nullopt)
      << "right() should be the small table";
  EXPECT_EQ(join->right()->output_schema().find_field("lval"), std::nullopt);
}

TEST_F(PhysicalPlannerJoinBuildSideTest, DoesNotSwapWhenTheSmallerSideIsAlreadyOnTheRight) {
  // Same two tables, opposite clause order: large first/left, small
  // second/right -- already the preferred shape, so no swap should occur.
  const PhysicalPlanPtr plan =
      plan_for_join("SELECT lval, sval FROM read_parquet('" + large_path_ + "') AS l JOIN read_parquet('" +
                    small_path_ + "') AS s ON l.jkey = s.jkey");
  const HashJoinNode* join = find_hash_join(plan.get());
  ASSERT_NE(join, nullptr);
  EXPECT_NE(join->right()->output_schema().find_field("sval"), std::nullopt)
      << "right() should still be the small table";
  EXPECT_EQ(join->right()->output_schema().find_field("lval"), std::nullopt);
}

}  // namespace
}  // namespace kernellake
