#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/planner/logical_planner.hpp"
#include "kernellake/sql/parser.hpp"

namespace kernellake {
namespace {

Schema sales_schema() {
  return Schema({
      Field{"region", string_type(false)},
      Field{"amount", float64_type(false)},
      Field{"event_date", date32_type(false)},
  });
}

Schema lineitem_schema() {
  return Schema({
      Field{"l_extendedprice", float64_type(false)},
      Field{"l_discount", float64_type(false)},
      Field{"l_shipdate", date32_type(false)},
      Field{"l_quantity", int32_type(false)},
  });
}

TEST(LogicalPlanner, BuildsGeneralMvpQueryShape) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total_amount, COUNT(*) AS order_count "
      "FROM read_parquet('/data/sales/*.parquet') "
      "WHERE event_date >= DATE '2026-01-01' "
      "GROUP BY region "
      "LIMIT 100");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  ASSERT_EQ(plan->node_name(), "LogicalLimit");
  const auto* limit = dynamic_cast<const LogicalLimit*>(plan.get());
  ASSERT_NE(limit, nullptr);
  EXPECT_EQ(limit->limit(), 100);

  const auto* projection = dynamic_cast<const LogicalProjection*>(limit->child().get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 3u);
  EXPECT_EQ(projection->items()[0].name, "region");
  EXPECT_EQ(projection->items()[1].name, "total_amount");
  EXPECT_EQ(projection->items()[2].name, "order_count");

  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  ASSERT_EQ(aggregate->group_by().size(), 1u);
  EXPECT_EQ(aggregate->group_by()[0].name, "region");
  ASSERT_EQ(aggregate->aggregates().size(), 2u);

  const auto* filter = dynamic_cast<const LogicalFilter*>(aggregate->children()[0].get());
  ASSERT_NE(filter, nullptr);

  const auto* scan = dynamic_cast<const LogicalScan*>(filter->children()[0].get());
  ASSERT_NE(scan, nullptr);
  EXPECT_EQ(scan->source_paths()[0], "/data/sales/*.parquet");
}

TEST(LogicalPlanner, ReordersAggregateOutputToMatchSelectList) {
  // Aggregate listed before the grouped column, unlike LogicalAggregate's
  // fixed internal [group_by..., aggregates...] output order.
  const auto stmt = sql::parse_sql(
      "SELECT SUM(amount) AS total, region FROM read_parquet('/x.parquet') GROUP BY region");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 2u);
  EXPECT_EQ(projection->items()[0].name, "total");
  EXPECT_EQ(projection->items()[1].name, "region");
  // Verify the projection's column expressions actually evaluate the right
  // underlying aggregate schema position, not just carry the right name.
  const auto* total_col = dynamic_cast<const ColumnExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(total_col, nullptr);
  EXPECT_EQ(total_col->column_index(), 1u);  // aggregates come after the 1 group-by column
  const auto* region_col =
      dynamic_cast<const ColumnExpression*>(projection->items()[1].expr.get());
  ASSERT_NE(region_col, nullptr);
  EXPECT_EQ(region_col->column_index(), 0u);
}

TEST(LogicalPlanner, BuildsTpchQ6ScalarAggregateShape) {
  const auto stmt = sql::parse_sql(
      "SELECT SUM(l_extendedprice * l_discount) AS revenue "
      "FROM read_parquet('/data/tpch/lineitem/*.parquet') "
      "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
      "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24");
  const Schema schema = lineitem_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  EXPECT_EQ(aggregate->group_by().size(), 0u);
  ASSERT_EQ(aggregate->aggregates().size(), 1u);
  EXPECT_EQ(aggregate->aggregates()[0].name, "revenue");
}

TEST(LogicalPlanner, RejectsOrderByAfterGroupBy) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) FROM read_parquet('/x.parquet') GROUP BY region "
      "ORDER BY region");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  EXPECT_THROW(build_logical_plan(bound, schema), PlanningError);
}

TEST(LogicalPlanner, NonAggregateOrderByPlacedBeforeProjection) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') ORDER BY amount DESC");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* sort = dynamic_cast<const LogicalSort*>(projection->children()[0].get());
  ASSERT_NE(sort, nullptr);
  ASSERT_EQ(sort->keys().size(), 1u);
  EXPECT_FALSE(sort->keys()[0].ascending);
}

TEST(LogicalPlanner, ExplainTextContainsExpectedNodesAndAttributes) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total_amount "
      "FROM read_parquet('/data/sales/*.parquet') "
      "WHERE event_date >= DATE '2026-01-01' "
      "GROUP BY region");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("LogicalProjection"), std::string::npos);
  EXPECT_NE(text.find("LogicalAggregate"), std::string::npos);
  EXPECT_NE(text.find("group_by: [region]"), std::string::npos);
  EXPECT_NE(text.find("LogicalFilter"), std::string::npos);
  EXPECT_NE(text.find("LogicalScan"), std::string::npos);
  EXPECT_NE(text.find("/data/sales/*.parquet"), std::string::npos);
}

TEST(LogicalPlanner, ExplainJsonIsValidAndContainsNodeNames) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/data/sales/*.parquet')");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const std::string json_text = explain_json(*plan);
  EXPECT_NE(json_text.find("\"node\": \"LogicalProjection\""), std::string::npos);
  EXPECT_NE(json_text.find("\"LogicalScan\""), std::string::npos);
}

}  // namespace
}  // namespace kernellake
