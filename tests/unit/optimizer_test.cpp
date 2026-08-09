#include <gtest/gtest.h>

#include <algorithm>

#include "kernellake/optimizer/optimizer.hpp"
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

LogicalPlanPtr plan_for(const std::string& sql, const Schema& schema) {
  const auto stmt = sql::parse_sql(sql);
  const BoundQuery bound = bind_query(stmt, schema);
  return build_logical_plan(bound, schema);
}

LogicalPlanPtr plan_for_join(const std::string& sql, const std::vector<Schema>& schemas) {
  const auto stmt = sql::parse_sql(sql);
  const BoundQuery bound = bind_query(stmt, schemas);
  return build_logical_plan(bound, schemas);
}

const LogicalScan* find_scan(const LogicalPlanPtr& node) {
  if (const auto* scan = dynamic_cast<const LogicalScan*>(node.get())) return scan;
  for (const LogicalPlanPtr& child : node->children()) {
    if (const LogicalScan* found = find_scan(child)) return found;
  }
  return nullptr;
}

// Finds the LogicalScan whose schema has a field named `column_name` --
// for predicate-pushdown tests below, where a query joins several tables
// and each assertion needs to check a *specific* table's own scan, not
// just "some scan somewhere" (find_scan() above only ever finds the
// first, e.g. for a single-table query).
const LogicalScan* find_scan_with_column(const LogicalPlanPtr& node, const std::string& column_name) {
  if (const auto* scan = dynamic_cast<const LogicalScan*>(node.get())) {
    return scan->output_schema().find_field(column_name) ? scan : nullptr;
  }
  for (const LogicalPlanPtr& child : node->children()) {
    if (const LogicalScan* found = find_scan_with_column(child, column_name)) return found;
  }
  return nullptr;
}

Schema customer_schema() {
  return Schema({
      Field{"c_custkey", int64_type(false)},
      Field{"c_mktsegment", string_type(false)},
  });
}

Schema orders_schema() {
  return Schema({
      Field{"o_orderkey", int64_type(false)},
      Field{"o_custkey", int64_type(false)},
      Field{"o_orderdate", date32_type(false)},
  });
}

Schema q3_lineitem_schema() {
  return Schema({
      Field{"l_orderkey", int64_type(false)},
      Field{"l_shipdate", date32_type(false)},
  });
}

// These three tests select only a strict subset of a two-column schema;
// keeping them isolated to a subset (rather than the full schema) has no
// bearing on the optimizer itself anymore -- see
// Optimizer.KeepsIdentityProjectionForColumnPruningToSeeLater below for why.
Schema two_column_schema() {
  return Schema({Field{"a", int64_type(false)}, Field{"b", int64_type(false)}});
}

TEST(Optimizer, ConstantFoldsArithmetic) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 1 + 2", two_column_schema());
  plan = optimize(std::move(plan));

  // Plan shape: Projection -> Filter -> Scan.
  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* real_filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(real_filter, nullptr);
  EXPECT_EQ(real_filter->predicate()->to_string(), "(a > 3)");
}

// Regression test: constant folding used to round-trip int64 literals
// through double, which silently loses precision past 2^53 -- this literal
// pair (both well past 2^53 but their sum still fits in int64) would fold
// to the wrong value under the old double-based path.
TEST(Optimizer, ConstantFoldsLargeInt64ArithmeticExactly) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 4611686018427387904 + 1",
                       two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* real_filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(real_filter, nullptr);
  EXPECT_EQ(real_filter->predicate()->to_string(), "(a > 4611686018427387905)");
}

// Regression test: static_cast<int64_t>(double) on a magnitude past
// INT64_MAX/MIN is undefined behavior -- the old double-based folding path
// had no overflow check at all before that cast. An overflowing int64 sum
// must be left unfolded (evaluated at runtime instead), not crash or fold
// to a wrapped/garbage value.
TEST(Optimizer, DoesNotFoldOverflowingInt64Arithmetic) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 9223372036854775807 + 1",
                       two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* real_filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(real_filter, nullptr);
  EXPECT_EQ(real_filter->predicate()->to_string(), "(a > (9223372036854775807 + 1))");
}

// Regression test: constant folding used to skip CASE branches entirely
// (simplify_expression had no case for CaseExpression, falling through to
// its default "nothing to simplify" branch) -- a foldable THEN/ELSE
// expression must still be folded.
TEST(Optimizer, ConstantFoldsInsideCaseBranches) {
  auto plan =
      plan_for("SELECT CASE WHEN a > 1 THEN 1 + 2 ELSE 3 + 4 END AS bucket FROM read_parquet('/x.parquet')",
               two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  EXPECT_EQ(projection->items()[0].expr->to_string(), "CASE WHEN (a > 1) THEN 3 ELSE 7 END");
}

TEST(Optimizer, RemovesFilterThatFoldsToTrue) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE 1 = 1", two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  // The filter should have been removed entirely, leaving Projection -> Scan.
  EXPECT_NE(dynamic_cast<const LogicalScan*>(projection->children()[0].get()), nullptr);
}

TEST(Optimizer, AnnotatesAlwaysFalseFilterWithZeroRows) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE 1 = 2", two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  ASSERT_TRUE(filter->estimated_rows.has_value());
  EXPECT_EQ(*filter->estimated_rows, 0u);
}

TEST(Optimizer, CombinesAdjacentFilters) {
  // Two WHERE conditions ANDed together already produce a single Filter from
  // the binder/planner; simulate an adjacent-filter scenario by nesting one
  // manually to prove the combine rule collapses it back to one node.
  Schema schema({Field{"a", int64_type(false)}, Field{"b", int64_type(false)}});
  auto inner_scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  auto a_col = std::make_shared<ColumnExpression>("a", 0, int64_type(false));
  auto one = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  auto inner_pred =
      std::make_shared<BinaryExpression>(BinaryOperator::Greater, a_col, one, boolean_type(false));
  auto inner_filter = std::make_shared<LogicalFilter>(inner_scan, inner_pred);

  auto b_col = std::make_shared<ColumnExpression>("b", 1, int64_type(false));
  auto two = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(2));
  auto outer_pred = std::make_shared<BinaryExpression>(BinaryOperator::Less, b_col, two, boolean_type(false));
  auto outer_filter = std::make_shared<LogicalFilter>(inner_filter, outer_pred);

  LogicalPlanPtr optimized = optimize(outer_filter);
  const auto* combined = dynamic_cast<const LogicalFilter*>(optimized.get());
  ASSERT_NE(combined, nullptr);
  EXPECT_NE(dynamic_cast<const LogicalScan*>(combined->children()[0].get()), nullptr);
  EXPECT_EQ(combined->predicate()->to_string(), "((b < 2) AND (a > 1))");
}

TEST(Optimizer, SimplifiesBetweenIntoComparisons) {
  auto plan =
      plan_for("SELECT l_discount FROM read_parquet('/x.parquet') WHERE l_discount BETWEEN 0.05 AND 0.07",
               lineitem_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  const auto* binary = dynamic_cast<const BinaryExpression*>(filter->predicate().get());
  ASSERT_NE(binary, nullptr);
  EXPECT_EQ(binary->op(), BinaryOperator::And);
}

// Regression test: an identity projection (its items exactly reproduce the
// child's own schema, in the same order) used to be elided right here in the
// optimizer, before annotate_scan()'s column-pruning pass ever ran over the
// rewritten tree -- silently discarding the only record of which columns the
// query's actual output needed. A Filter/Sort/Aggregate sitting under the
// elided projection would then only mark its own referenced columns as
// required, and the scan would drop every other selected column with no
// error at all: `SELECT id, amount FROM t WHERE id < 3` (id, amount are the
// table's only two columns, selected in schema order -- exactly this
// pattern, and also exactly what `SELECT * ... WHERE ...` desugars to)
// returned only the `id` column, confirmed for real on both backends. Fixed
// by keeping the identity projection in the logical tree so annotate_scan
// can see it like any other projection; the equivalent optimization (skip
// materializing a pass-through-only ProjectionNode at execution time) now
// happens later, in physical_planner.cpp, once column pruning has already
// run -- see PhysicalPlannerTest.ElidesIdentityProjectionAfterColumnPruning
// and .RegressionKeepsEveryColumnWhenSelectListMatchesSchemaOrder.
TEST(Optimizer, KeepsIdentityProjectionForColumnPruningToSeeLater) {
  Schema schema({Field{"a", int64_type(false)}});
  auto scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  auto column = std::make_shared<ColumnExpression>("a", 0, int64_type(false));
  auto projection = std::make_shared<LogicalProjection>(scan, std::vector<NamedExpression>{{column, "a"}});

  LogicalPlanPtr optimized = optimize(projection);
  ASSERT_NE(dynamic_cast<const LogicalProjection*>(optimized.get()), nullptr);
}

TEST(Optimizer, PushesLimitThroughProjectionToScan) {
  auto plan = plan_for("SELECT region FROM read_parquet('/x.parquet') LIMIT 10", sales_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* limit = dynamic_cast<const LogicalLimit*>(projection->children()[0].get());
  ASSERT_NE(limit, nullptr);
  EXPECT_EQ(limit->limit(), 10);
  EXPECT_NE(dynamic_cast<const LogicalScan*>(limit->children()[0].get()), nullptr);
}

TEST(Optimizer, DoesNotPushLimitPastFilter) {
  auto plan =
      plan_for("SELECT region FROM read_parquet('/x.parquet') WHERE amount > 0 LIMIT 10", sales_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* limit = dynamic_cast<const LogicalLimit*>(projection->children()[0].get());
  ASSERT_NE(limit, nullptr);
  EXPECT_NE(dynamic_cast<const LogicalFilter*>(limit->children()[0].get()), nullptr);
}

TEST(Optimizer, IdentifiesRequiredScanColumnsForTpchQ6) {
  auto plan = plan_for(
      "SELECT SUM(l_extendedprice * l_discount) AS revenue "
      "FROM read_parquet('/x.parquet') "
      "WHERE l_shipdate >= DATE '1994-01-01' AND l_discount BETWEEN 0.05 AND 0.07 "
      "AND l_quantity < 24",
      lineitem_schema());
  plan = optimize(std::move(plan));

  const LogicalScan* scan = find_scan(plan);
  ASSERT_NE(scan, nullptr);
  const std::vector<std::string>& required = scan->required_columns();
  // l_shipdate, l_discount, l_quantity (from WHERE) + l_extendedprice,
  // l_discount (from SUM argument) => 4 distinct columns, not all 4 schema
  // columns trivially (this also happens to be all of them here, but the
  // point is the set is computed, not hard-coded to "all columns").
  EXPECT_EQ(required.size(), 4u);
  for (const char* expected : {"l_extendedprice", "l_discount", "l_shipdate", "l_quantity"}) {
    EXPECT_NE(std::find(required.begin(), required.end(), expected), required.end());
  }
}

TEST(Optimizer, DoesNotTreatAggregateReprojectionColumnsAsScanColumns) {
  auto plan = plan_for(
      "SELECT region, SUM(amount) AS total_amount FROM read_parquet('/x.parquet') "
      "GROUP BY region",
      sales_schema());
  plan = optimize(std::move(plan));

  const LogicalScan* scan = find_scan(plan);
  ASSERT_NE(scan, nullptr);
  // Must be {region, amount}, never "total_amount" (that name only exists in
  // the aggregate's output schema, not the scan's).
  EXPECT_EQ(scan->required_columns().size(), 2u);
  EXPECT_EQ(std::find(scan->required_columns().begin(), scan->required_columns().end(), "total_amount"),
            scan->required_columns().end());
}

TEST(Optimizer, IdentifiesPushablePredicatesForPruning) {
  auto plan = plan_for(
      "SELECT region FROM read_parquet('/x.parquet') "
      "WHERE event_date >= DATE '2026-01-01' AND amount > 0",
      sales_schema());
  plan = optimize(std::move(plan));

  const LogicalScan* scan = find_scan(plan);
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->pushable_predicates().size(), 2u);
  EXPECT_EQ(scan->pushable_predicates()[0].column_name, "event_date");
  EXPECT_EQ(scan->pushable_predicates()[0].op, BinaryOperator::GreaterEqual);
  EXPECT_EQ(scan->pushable_predicates()[1].column_name, "amount");
}

// Regression test for a real SF100 GPU OOM on TPC-H Q3: predicate pushdown
// used to stop at a join (see push_predicate_through_join's own comment in
// optimizer.cpp), so a WHERE clause referencing only one side of a
// customer/orders JOIN sat above the join instead of filtering the base
// table first -- meaning the join's build side materialized the entire
// unfiltered table. Confirmed via a real local A/B (peak GPU memory 1.28
// GiB -> 0.42 GiB, 3.07x, for the equivalent 3-way query at TPC-H SF1).
TEST(Optimizer, PushesSingleSidedPredicateThroughTwoWayJoin) {
  auto plan = plan_for_join(
      "SELECT o_orderkey FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.o_custkey = c.c_custkey "
      "WHERE c_mktsegment = 'BUILDING' AND o_orderdate < DATE '1995-03-15'",
      std::vector<Schema>{orders_schema(), customer_schema()});
  plan = optimize(std::move(plan));

  // No Filter should remain directly above the join: both WHERE conjuncts
  // reference exactly one side each, so both should have been pushed all
  // the way down to their own base scan.
  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  EXPECT_NE(dynamic_cast<const LogicalJoin*>(projection->children()[0].get()), nullptr);

  // Reads back via pushable_predicates rather than looking for a Filter
  // node directly: annotate_scan (already run inside optimize()) collapses
  // a Filter sitting directly on a scan into the scan's own
  // pushable_predicates list, same as any other single-table WHERE clause.
  const LogicalScan* orders_scan = find_scan_with_column(plan, "o_orderdate");
  ASSERT_NE(orders_scan, nullptr);
  ASSERT_EQ(orders_scan->pushable_predicates().size(), 1u);
  EXPECT_EQ(orders_scan->pushable_predicates()[0].column_name, "o_orderdate");
  EXPECT_EQ(orders_scan->pushable_predicates()[0].op, BinaryOperator::Less);

  const LogicalScan* customer_scan = find_scan_with_column(plan, "c_mktsegment");
  ASSERT_NE(customer_scan, nullptr);
  ASSERT_EQ(customer_scan->pushable_predicates().size(), 1u);
  EXPECT_EQ(customer_scan->pushable_predicates()[0].column_name, "c_mktsegment");
  EXPECT_EQ(customer_scan->pushable_predicates()[0].op, BinaryOperator::Equal);
}

// The full TPC-H Q3 shape: a 3-way join chain with one single-table
// predicate per table. Confirms pushdown recurses through the whole
// left-deep chain (customer JOIN orders JOIN lineitem), not just one join
// level -- each of the three base scans should end up with its own
// predicate, and no Filter should remain above any join.
TEST(Optimizer, PushesPredicatesThroughThreeWayJoinChain) {
  auto plan = plan_for_join(
      "SELECT l_orderkey FROM read_parquet('/c.parquet') AS c "
      "JOIN read_parquet('/o.parquet') AS o ON c.c_custkey = o.o_custkey "
      "JOIN read_parquet('/l.parquet') AS l ON o.o_orderkey = l.l_orderkey "
      "WHERE c_mktsegment = 'BUILDING' AND o_orderdate < DATE '1995-03-15' "
      "AND l_shipdate > DATE '1995-03-15'",
      std::vector<Schema>{customer_schema(), orders_schema(), q3_lineitem_schema()});
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* outer_join = dynamic_cast<const LogicalJoin*>(projection->children()[0].get());
  ASSERT_NE(outer_join, nullptr);
  const auto* inner_join = dynamic_cast<const LogicalJoin*>(outer_join->children()[0].get());
  ASSERT_NE(inner_join, nullptr);
  // Neither join has a Filter sitting directly on top of it (checked via
  // the parent already being a Join/Projection above, not a Filter).

  const LogicalScan* customer_scan = find_scan_with_column(plan, "c_mktsegment");
  ASSERT_NE(customer_scan, nullptr);
  ASSERT_EQ(customer_scan->pushable_predicates().size(), 1u);
  EXPECT_EQ(customer_scan->pushable_predicates()[0].column_name, "c_mktsegment");

  const LogicalScan* orders_scan = find_scan_with_column(plan, "o_orderdate");
  ASSERT_NE(orders_scan, nullptr);
  ASSERT_EQ(orders_scan->pushable_predicates().size(), 1u);
  EXPECT_EQ(orders_scan->pushable_predicates()[0].column_name, "o_orderdate");

  const LogicalScan* lineitem_scan = find_scan_with_column(plan, "l_shipdate");
  ASSERT_NE(lineitem_scan, nullptr);
  ASSERT_EQ(lineitem_scan->pushable_predicates().size(), 1u);
  EXPECT_EQ(lineitem_scan->pushable_predicates()[0].column_name, "l_shipdate");
}

// A predicate mixing columns from both sides of a join (e.g. a non-equi
// comparison beyond the ON condition) can't be pushed to either side --
// must stay exactly where it was, directly above the join, rather than be
// dropped or incorrectly pushed to one side.
TEST(Optimizer, DoesNotPushCrossSidePredicateThroughJoin) {
  auto plan = plan_for_join(
      "SELECT o_orderkey FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.o_custkey = c.c_custkey "
      "WHERE o_orderkey > c_custkey",
      std::vector<Schema>{orders_schema(), customer_schema()});
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  EXPECT_NE(dynamic_cast<const LogicalJoin*>(filter->children()[0].get()), nullptr);
}

}  // namespace
}  // namespace kernellake
