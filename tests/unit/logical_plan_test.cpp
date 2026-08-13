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

// Three small schemas for N-way-join tests: a (a_key, a_val), b (b_key,
// a_key -- joins onto a), c (c_key, b_key -- joins onto b).
Schema table_a_schema() {
  return Schema({Field{"a_key", int64_type(false)}, Field{"a_val", string_type(false)}});
}
Schema table_b_schema() {
  return Schema({Field{"b_key", int64_type(false)}, Field{"a_key", int64_type(false)}});
}
Schema table_c_schema() {
  return Schema({Field{"c_val", string_type(false)}, Field{"b_key", int64_type(false)}});
}

// Two schemas sharing a bare column name ("amount") -- for a JOIN test
// where SUM(a.amount) and SUM(b.amount) must land in two distinct
// LogicalAggregate slots, not be deduplicated by a naive bare-name check.
Schema amount_schema_a() {
  return Schema({Field{"join_key", int64_type(false)}, Field{"amount", float64_type(false)}});
}
Schema amount_schema_b() {
  return Schema({Field{"join_key", int64_type(false)}, Field{"amount", float64_type(false)}});
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
  const auto stmt =
      sql::parse_sql("SELECT SUM(amount) AS total, region FROM read_parquet('/x.parquet') GROUP BY region");
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
  const auto* region_col = dynamic_cast<const ColumnExpression*>(projection->items()[1].expr.get());
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

// Regression test: a 3+-way JOIN chain used to be rejected outright at
// parse time; this pins down that build_logical_plan() constructs the
// expected left-deep chain of two LogicalJoin nodes for 3 sources
// (LogicalJoin(LogicalJoin(Scan(a), Scan(b)), Scan(c))), not some other
// shape -- see docs/ARCHITECTURE.md's "N-way joins" section.
TEST(LogicalPlanner, BuildsLeftDeepJoinChainForThreeTableJoin) {
  const auto stmt = sql::parse_sql(
      "SELECT a.a_val, c.c_val FROM read_parquet('/a.parquet') AS a "
      "JOIN read_parquet('/b.parquet') AS b ON a.a_key = b.a_key "
      "JOIN read_parquet('/c.parquet') AS c ON b.b_key = c.b_key");
  const std::vector<Schema> schemas = {table_a_schema(), table_b_schema(), table_c_schema()};
  const BoundQuery bound = bind_query(stmt, schemas);
  const LogicalPlanPtr plan = build_logical_plan(bound, schemas);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* outer_join = dynamic_cast<const LogicalJoin*>(projection->children()[0].get());
  ASSERT_NE(outer_join, nullptr);
  const auto* outer_right_scan = dynamic_cast<const LogicalScan*>(outer_join->right().get());
  ASSERT_NE(outer_right_scan, nullptr);
  EXPECT_EQ(outer_right_scan->output_schema().field_count(), table_c_schema().field_count());

  const auto* inner_join = dynamic_cast<const LogicalJoin*>(outer_join->left().get());
  ASSERT_NE(inner_join, nullptr);
  const auto* inner_left_scan = dynamic_cast<const LogicalScan*>(inner_join->left().get());
  ASSERT_NE(inner_left_scan, nullptr);
  EXPECT_EQ(inner_left_scan->output_schema().field_count(), table_a_schema().field_count());
  const auto* inner_right_scan = dynamic_cast<const LogicalScan*>(inner_join->right().get());
  ASSERT_NE(inner_right_scan, nullptr);
  EXPECT_EQ(inner_right_scan->output_schema().field_count(), table_b_schema().field_count());

  // Inner join key: a.a_key (index 0) = b.a_key (index 1 in table_b_schema()).
  EXPECT_EQ(inner_join->left_key_index(), 0u);
  EXPECT_EQ(inner_join->right_key_index(), 1u);
  // Outer join key: b.b_key -- index 2 in the combined [a, b] schema so
  // far (a contributes 2 fields, b_key is b's own index 0) = c.b_key
  // (index 1 in table_c_schema()).
  EXPECT_EQ(outer_join->left_key_index(), 2u);
  EXPECT_EQ(outer_join->right_key_index(), 1u);
}

// Regression test: build_logical_plan() used to only recognize a SELECT
// item as valid in an aggregate query if it *was* (at the top level)
// exactly an AggregateExpression or exactly matched a GROUP BY key by
// to_string() -- anything else threw "SELECT item '...' is neither an
// aggregate nor a GROUP BY column", even though the binder had already
// bound it successfully. TPC-H Q14's own SELECT item is exactly this
// shape: `100.00 * SUM(CASE WHEN ... THEN ... ELSE 0 END) / SUM(...)`, a
// BinaryExpression combining two AggregateExpression subtrees arithmetically,
// neither one bare. Fixed by rewrite_aggregate_refs(), which recursively
// finds every distinct aggregate subtree (registering each once) and
// rebuilds the surrounding expression with ColumnExpression references to
// their LogicalAggregate output slots -- found and fixed while adding Q14.
TEST(LogicalPlanner, AggregateArithmeticCombiningTwoAggregatesBuildsBothSlots) {
  const auto stmt = sql::parse_sql(
      "SELECT 100.0 * SUM(CASE WHEN l_quantity > 10 THEN l_extendedprice * (1 - l_discount) ELSE 0 END) "
      "/ SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue "
      "FROM read_parquet('/data/tpch/lineitem/*.parquet')");
  const Schema schema = lineitem_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  EXPECT_EQ(projection->items()[0].name, "promo_revenue");
  const auto* division = dynamic_cast<const BinaryExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(division, nullptr);
  EXPECT_EQ(division->op(), BinaryOperator::Divide);

  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  EXPECT_EQ(aggregate->group_by().size(), 0u);
  // Two distinct aggregates: the CASE-wrapped SUM and the plain SUM --
  // neither is the SELECT item itself, both had to be found by recursing
  // into the multiplication/division tree.
  ASSERT_EQ(aggregate->aggregates().size(), 2u);
}

// register_aggregate() dedups by structural_key(): the same aggregate
// expression referenced twice in the SELECT list (here, under two different
// aliases) must only occupy one LogicalAggregate slot, keeping the *first*
// name it was registered under.
TEST(LogicalPlanner, DuplicateAggregateCallsShareOneAggregateSlot) {
  const auto stmt =
      sql::parse_sql("SELECT SUM(amount) AS a, SUM(amount) AS b FROM read_parquet('/x.parquet')");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  ASSERT_EQ(aggregate->aggregates().size(), 1u);
  EXPECT_EQ(aggregate->aggregates()[0].name, "a");

  ASSERT_EQ(projection->items().size(), 2u);
  const auto* a_col = dynamic_cast<const ColumnExpression*>(projection->items()[0].expr.get());
  const auto* b_col = dynamic_cast<const ColumnExpression*>(projection->items()[1].expr.get());
  ASSERT_NE(a_col, nullptr);
  ASSERT_NE(b_col, nullptr);
  EXPECT_EQ(a_col->column_index(), b_col->column_index());
}

// The dedup above is keyed by structural_key(), not a bare column name --
// SUM(a.amount) and SUM(b.amount) reference "amount" on opposite JOIN
// sides (different column_index()) and must get two distinct slots.
TEST(LogicalPlanner, AggregatesOverSameNamedColumnsFromDifferentJoinSidesGetDistinctSlots) {
  const auto stmt = sql::parse_sql(
      "SELECT SUM(a.amount) + SUM(b.amount) AS total FROM read_parquet('/a.parquet') AS a "
      "JOIN read_parquet('/b.parquet') AS b ON a.join_key = b.join_key");
  const std::vector<Schema> schemas = {amount_schema_a(), amount_schema_b()};
  const BoundQuery bound = bind_query(stmt, schemas);
  const LogicalPlanPtr plan = build_logical_plan(bound, schemas);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  EXPECT_EQ(aggregate->aggregates().size(), 2u);
}

// rewrite_aggregate_refs() recursing into an aggregate wrapped in a
// BinaryExpression is already covered by
// AggregateArithmeticCombiningTwoAggregatesBuildsBothSlots above; these
// three cover the remaining wrapper kinds it also has a dedicated case
// for -- only CASE-wrapping had test coverage before.
TEST(LogicalPlanner, RewriteAggregateRefsHandlesAggregateWrappedInBetween) {
  const auto stmt = sql::parse_sql(
      "SELECT SUM(l_extendedprice) BETWEEN 1 AND SUM(l_discount) AS in_range "
      "FROM read_parquet('/x.parquet')");
  const Schema schema = lineitem_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  const auto* between = dynamic_cast<const BetweenExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(between, nullptr);
  EXPECT_NE(dynamic_cast<const ColumnExpression*>(between->value().get()), nullptr);
  EXPECT_NE(dynamic_cast<const ColumnExpression*>(between->upper().get()), nullptr);

  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  ASSERT_EQ(aggregate->aggregates().size(), 2u);
}

TEST(LogicalPlanner, RewriteAggregateRefsHandlesAggregateWrappedInLike) {
  const auto stmt = sql::parse_sql("SELECT MIN(region) LIKE 'A%' AS has_a FROM read_parquet('/x.parquet')");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  const auto* like = dynamic_cast<const LikeExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(like, nullptr);
  EXPECT_NE(dynamic_cast<const ColumnExpression*>(like->value().get()), nullptr);

  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  ASSERT_EQ(aggregate->aggregates().size(), 1u);
}

TEST(LogicalPlanner, RewriteAggregateRefsHandlesAggregateWrappedInExtract) {
  const auto stmt =
      sql::parse_sql("SELECT EXTRACT(YEAR FROM MIN(event_date)) AS min_year FROM read_parquet('/x.parquet')");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  const auto* extract = dynamic_cast<const ExtractExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(extract, nullptr);
  EXPECT_NE(dynamic_cast<const ColumnExpression*>(extract->operand().get()), nullptr);

  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  ASSERT_EQ(aggregate->aggregates().size(), 1u);
}

TEST(LogicalPlanner, AggregateOrderByReferencesSelectListOutputName) {
  // "region" here means the SELECT-list output column (also happens to be
  // the GROUP BY key's own name in this case), not a base-table column --
  // see binder.cpp's ORDER BY handling for why that distinction matters.
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') GROUP BY region "
      "ORDER BY region DESC");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const auto* sort = dynamic_cast<const LogicalSort*>(plan.get());
  ASSERT_NE(sort, nullptr);
  ASSERT_EQ(sort->keys().size(), 1u);
  EXPECT_FALSE(sort->keys()[0].ascending);
  // Sort sits directly on the final LogicalProjection, whose output schema
  // the sort key's column index must match -- see the physical planner's
  // "keys_reference_scan_schema" discriminator, which relies on exactly
  // this structural shape.
  const auto* projection = dynamic_cast<const LogicalProjection*>(sort->children()[0].get());
  ASSERT_NE(projection, nullptr);
}

TEST(LogicalPlanner, RejectsAggregateOrderByOnNonOutputExpression) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') GROUP BY region "
      "ORDER BY amount");
  const Schema schema = sales_schema();
  // "amount" is not a SELECT-list output name (only "region" and "total"
  // are) -- ORDER BY after GROUP BY is scoped to output names, not
  // arbitrary re-derived expressions (see binder.cpp).
  EXPECT_THROW((void)(bind_query(stmt, schema)), BindingError);
}

TEST(LogicalPlanner, NonAggregateOrderByPlacedBeforeProjection) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') ORDER BY amount DESC");
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
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/data/sales/*.parquet')");
  const Schema schema = sales_schema();
  const BoundQuery bound = bind_query(stmt, schema);
  const LogicalPlanPtr plan = build_logical_plan(bound, schema);

  const std::string json_text = explain_json(*plan);
  EXPECT_NE(json_text.find("\"node\": \"LogicalProjection\""), std::string::npos);
  EXPECT_NE(json_text.find("\"LogicalScan\""), std::string::npos);
}

}  // namespace
}  // namespace kernellake
