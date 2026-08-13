#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <tuple>

#include "kernellake/common/errors.hpp"
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

// Only Add is exercised above (ConstantFoldsArithmetic/
// ConstantFoldsLargeInt64ArithmeticExactly/DoesNotFoldOverflowingInt64Arithmetic)
// -- Subtract/Multiply/Divide share the same fold_arithmetic_int64() switch
// but had no test of their own.
TEST(Optimizer, ConstantFoldsSubtractMultiplyDivide) {
  auto subtract_plan =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 10 - 3", two_column_schema());
  subtract_plan = optimize(std::move(subtract_plan));
  const auto* subtract_projection = dynamic_cast<const LogicalProjection*>(subtract_plan.get());
  ASSERT_NE(subtract_projection, nullptr);
  const auto* subtract_filter = dynamic_cast<const LogicalFilter*>(subtract_projection->children()[0].get());
  ASSERT_NE(subtract_filter, nullptr);
  EXPECT_EQ(subtract_filter->predicate()->to_string(), "(a > 7)");

  auto multiply_plan =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 3 * 4", two_column_schema());
  multiply_plan = optimize(std::move(multiply_plan));
  const auto* multiply_projection = dynamic_cast<const LogicalProjection*>(multiply_plan.get());
  ASSERT_NE(multiply_projection, nullptr);
  const auto* multiply_filter = dynamic_cast<const LogicalFilter*>(multiply_projection->children()[0].get());
  ASSERT_NE(multiply_filter, nullptr);
  EXPECT_EQ(multiply_filter->predicate()->to_string(), "(a > 12)");

  auto divide_plan =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 20 / 4", two_column_schema());
  divide_plan = optimize(std::move(divide_plan));
  const auto* divide_projection = dynamic_cast<const LogicalProjection*>(divide_plan.get());
  ASSERT_NE(divide_projection, nullptr);
  const auto* divide_filter = dynamic_cast<const LogicalFilter*>(divide_projection->children()[0].get());
  ASSERT_NE(divide_filter, nullptr);
  EXPECT_EQ(divide_filter->predicate()->to_string(), "(a > 5)");
}

TEST(Optimizer, DoesNotFoldDivisionByZero) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 20 / 0", two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  EXPECT_EQ(filter->predicate()->to_string(), "(a > (20 / 0))");
}

// INT64_MIN / -1 overflows int64 (its magnitude exceeds INT64_MAX) and is
// undefined behavior for the raw '/' operator -- fold_arithmetic_int64()
// has a dedicated guard for exactly this case, same as divide-by-zero.
// Built directly (bypassing SQL text) since INT64_MIN has no representable
// positive counterpart to negate through the parser's own literal grammar.
TEST(Optimizer, DoesNotFoldInt64MinDividedByNegativeOne) {
  Schema schema({Field{"a", int64_type(false)}});
  auto scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  auto min_literal = std::make_shared<LiteralExpression>(
      LiteralExpression::make_int64(std::numeric_limits<std::int64_t>::min()));
  auto minus_one = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(-1));
  auto divide =
      std::make_shared<BinaryExpression>(BinaryOperator::Divide, min_literal, minus_one, int64_type(false));
  auto filter = std::make_shared<LogicalFilter>(scan, divide);

  LogicalPlanPtr optimized = optimize(filter);
  const auto* result_filter = dynamic_cast<const LogicalFilter*>(optimized.get());
  ASSERT_NE(result_filter, nullptr);
  const auto* result_binary = dynamic_cast<const BinaryExpression*>(result_filter->predicate().get());
  ASSERT_NE(result_binary, nullptr);
  EXPECT_EQ(result_binary->op(), BinaryOperator::Divide);
}

// Only Greater is exercised elsewhere (e.g. ConstantFoldsArithmetic) --
// nested inside CASE branches (whose own folding is already proven by
// ConstantFoldsInsideCaseBranches below) so the folded boolean literal is
// directly observable without a further AND/OR/removal pass collapsing it
// away, the way it would if used as a bare WHERE predicate.
TEST(Optimizer, ConstantFoldsRemainingComparisonOperators) {
  auto plan = plan_for(
      "SELECT CASE WHEN 1 = 1 THEN 1 WHEN 2 <> 2 THEN 2 WHEN 1 < 2 THEN 3 WHEN 2 <= 2 THEN 4 "
      "WHEN 3 >= 4 THEN 5 ELSE 6 END AS bucket FROM read_parquet('/x.parquet')",
      two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  EXPECT_EQ(projection->items()[0].expr->to_string(),
            "CASE WHEN TRUE THEN 1 WHEN FALSE THEN 2 WHEN TRUE THEN 3 WHEN TRUE THEN 4 WHEN FALSE THEN 5 "
            "ELSE 6 END");
}

TEST(Optimizer, FoldsNotOfLiteralAndEliminatesDoubleNegation) {
  auto plan =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE NOT (1 = 2) AND a > 0", two_column_schema());
  plan = optimize(std::move(plan));
  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  EXPECT_EQ(filter->predicate()->to_string(), "(a > 0)");

  // NOT (NOT (a > 0)) simplifies straight to (a > 0), without ever passing
  // through a boolean literal along the way -- built directly since there's
  // no SQL syntax for a bare double-NOT of a non-literal expression that
  // survives binding unchanged.
  Schema schema({Field{"a", int64_type(false)}});
  auto scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  auto a_col = std::make_shared<ColumnExpression>("a", 0, int64_type(false));
  auto zero = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(0));
  auto comparison =
      std::make_shared<BinaryExpression>(BinaryOperator::Greater, a_col, zero, boolean_type(false));
  auto inner_not = std::make_shared<UnaryExpression>(UnaryOperator::Not, comparison, boolean_type(false));
  auto outer_not = std::make_shared<UnaryExpression>(UnaryOperator::Not, inner_not, boolean_type(false));
  auto double_negation_filter = std::make_shared<LogicalFilter>(scan, outer_not);

  LogicalPlanPtr optimized = optimize(double_negation_filter);
  const auto* result_filter = dynamic_cast<const LogicalFilter*>(optimized.get());
  ASSERT_NE(result_filter, nullptr);
  EXPECT_EQ(result_filter->predicate()->to_string(), "(a > 0)");
}

TEST(Optimizer, AndShortCircuitsOnLiteralBooleanOperand) {
  auto true_left =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE (1 = 1) AND a > 0", two_column_schema());
  true_left = optimize(std::move(true_left));
  const auto* true_left_projection = dynamic_cast<const LogicalProjection*>(true_left.get());
  ASSERT_NE(true_left_projection, nullptr);
  const auto* true_left_filter =
      dynamic_cast<const LogicalFilter*>(true_left_projection->children()[0].get());
  ASSERT_NE(true_left_filter, nullptr);
  EXPECT_EQ(true_left_filter->predicate()->to_string(), "(a > 0)");

  auto true_right =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 0 AND (1 = 1)", two_column_schema());
  true_right = optimize(std::move(true_right));
  const auto* true_right_projection = dynamic_cast<const LogicalProjection*>(true_right.get());
  ASSERT_NE(true_right_projection, nullptr);
  const auto* true_right_filter =
      dynamic_cast<const LogicalFilter*>(true_right_projection->children()[0].get());
  ASSERT_NE(true_right_filter, nullptr);
  EXPECT_EQ(true_right_filter->predicate()->to_string(), "(a > 0)");

  // AND with a literal FALSE operand collapses to FALSE outright, discarding
  // the other operand entirely -- surfaces as the always-false/zero-rows
  // annotation, same as AnnotatesAlwaysFalseFilterWithZeroRows above.
  auto false_left =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE (1 = 2) AND a > 0", two_column_schema());
  false_left = optimize(std::move(false_left));
  const auto* false_left_projection = dynamic_cast<const LogicalProjection*>(false_left.get());
  ASSERT_NE(false_left_projection, nullptr);
  const auto* false_left_filter =
      dynamic_cast<const LogicalFilter*>(false_left_projection->children()[0].get());
  ASSERT_NE(false_left_filter, nullptr);
  ASSERT_TRUE(false_left_filter->estimated_rows.has_value());
  EXPECT_EQ(*false_left_filter->estimated_rows, 0u);

  auto false_right =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 0 AND (1 = 2)", two_column_schema());
  false_right = optimize(std::move(false_right));
  const auto* false_right_projection = dynamic_cast<const LogicalProjection*>(false_right.get());
  ASSERT_NE(false_right_projection, nullptr);
  const auto* false_right_filter =
      dynamic_cast<const LogicalFilter*>(false_right_projection->children()[0].get());
  ASSERT_NE(false_right_filter, nullptr);
  ASSERT_TRUE(false_right_filter->estimated_rows.has_value());
  EXPECT_EQ(*false_right_filter->estimated_rows, 0u);
}

TEST(Optimizer, OrShortCircuitsOnLiteralBooleanOperand) {
  // OR with a literal TRUE operand collapses to TRUE outright -- the filter
  // is removed entirely, same as RemovesFilterThatFoldsToTrue above.
  auto true_left =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE (1 = 1) OR a > 0", two_column_schema());
  true_left = optimize(std::move(true_left));
  const auto* true_left_projection = dynamic_cast<const LogicalProjection*>(true_left.get());
  ASSERT_NE(true_left_projection, nullptr);
  EXPECT_NE(dynamic_cast<const LogicalScan*>(true_left_projection->children()[0].get()), nullptr);

  auto true_right =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 0 OR (1 = 1)", two_column_schema());
  true_right = optimize(std::move(true_right));
  const auto* true_right_projection = dynamic_cast<const LogicalProjection*>(true_right.get());
  ASSERT_NE(true_right_projection, nullptr);
  EXPECT_NE(dynamic_cast<const LogicalScan*>(true_right_projection->children()[0].get()), nullptr);

  // OR with a literal FALSE operand collapses to the other operand.
  auto false_left =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE (1 = 2) OR a > 0", two_column_schema());
  false_left = optimize(std::move(false_left));
  const auto* false_left_projection = dynamic_cast<const LogicalProjection*>(false_left.get());
  ASSERT_NE(false_left_projection, nullptr);
  const auto* false_left_filter =
      dynamic_cast<const LogicalFilter*>(false_left_projection->children()[0].get());
  ASSERT_NE(false_left_filter, nullptr);
  EXPECT_EQ(false_left_filter->predicate()->to_string(), "(a > 0)");

  auto false_right =
      plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 0 OR (1 = 2)", two_column_schema());
  false_right = optimize(std::move(false_right));
  const auto* false_right_projection = dynamic_cast<const LogicalProjection*>(false_right.get());
  ASSERT_NE(false_right_projection, nullptr);
  const auto* false_right_filter =
      dynamic_cast<const LogicalFilter*>(false_right_projection->children()[0].get());
  ASSERT_NE(false_right_filter, nullptr);
  EXPECT_EQ(false_right_filter->predicate()->to_string(), "(a > 0)");
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

// simplify_expression() has a dedicated case for CastExpression (recurses
// into its operand), CaseExpression (tested just above), AggregateExpression,
// LikeExpression and ExtractExpression -- only CaseExpression had a test.
TEST(Optimizer, ConstantFoldsInsideCastOperand) {
  auto plan =
      plan_for("SELECT CAST(1 + 2 AS BIGINT) AS x FROM read_parquet('/x.parquet')", two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  const auto* cast = dynamic_cast<const CastExpression*>(projection->items()[0].expr.get());
  ASSERT_NE(cast, nullptr);
  const auto* literal = dynamic_cast<const LiteralExpression*>(cast->operand().get());
  ASSERT_NE(literal, nullptr);
  ASSERT_TRUE(std::holds_alternative<std::int64_t>(literal->value()));
  EXPECT_EQ(std::get<std::int64_t>(literal->value()), 3);
}

TEST(Optimizer, ConstantFoldsInsideAggregateArgument) {
  auto plan =
      plan_for("SELECT SUM(amount * (1 + 2)) AS total FROM read_parquet('/x.parquet')", sales_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(projection->children()[0].get());
  ASSERT_NE(aggregate, nullptr);
  ASSERT_EQ(aggregate->aggregates().size(), 1u);
  const auto* sum = dynamic_cast<const AggregateExpression*>(aggregate->aggregates()[0].expr.get());
  ASSERT_NE(sum, nullptr);
  const auto* product = dynamic_cast<const BinaryExpression*>(sum->argument().get());
  ASSERT_NE(product, nullptr);
  // amount * (1 + 2): the int64 literal side gets implicitly CAST to
  // FLOAT64 to match `amount` (see binder.cpp's cast_if_needed) -- the
  // folded literal ends up underneath that CAST, not as the CAST's sibling
  // directly.
  const auto* cast = dynamic_cast<const CastExpression*>(product->right().get());
  ASSERT_NE(cast, nullptr);
  const auto* literal = dynamic_cast<const LiteralExpression*>(cast->operand().get());
  ASSERT_NE(literal, nullptr);
  ASSERT_TRUE(std::holds_alternative<std::int64_t>(literal->value()));
  EXPECT_EQ(std::get<std::int64_t>(literal->value()), 3);
}

TEST(Optimizer, ConstantFoldsInsideLikeValueOperand) {
  auto plan = plan_for(
      "SELECT (CASE WHEN 1 = 1 THEN region ELSE 'X' END) LIKE 'A%' AS matches FROM "
      "read_parquet('/x.parquet')",
      sales_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  EXPECT_EQ(projection->items()[0].expr->to_string(), "CASE WHEN TRUE THEN region ELSE 'X' END LIKE 'A%'");
}

TEST(Optimizer, ConstantFoldsInsideExtractOperand) {
  auto plan = plan_for(
      "SELECT EXTRACT(YEAR FROM CASE WHEN 1 = 1 THEN event_date ELSE event_date END) AS y "
      "FROM read_parquet('/x.parquet')",
      sales_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  ASSERT_EQ(projection->items().size(), 1u);
  EXPECT_EQ(projection->items()[0].expr->to_string(),
            "EXTRACT(YEAR FROM CASE WHEN TRUE THEN event_date ELSE event_date END)");
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

// Regression test: `amount` is FLOAT64, so the integer literal `0` in
// `amount > 0` gets implicitly wrapped in a CastExpression by the binder to
// match its type (promote_numeric/cast_if_needed in binder.cpp).
// collect_pushable_predicates() used to store that still-CAST-wrapped
// expression into PushablePredicate::literal despite unwrap_cast() already
// having found the real LiteralExpression underneath -- violating that
// field's documented "always a LiteralExpression" invariant. Both real
// consumers (parquet_pruning.cpp, iceberg/partition_pruning.cpp)
// dynamic_cast .literal to LiteralExpression and silently skip min/max
// -stats pruning on a null result, with no error -- so this test asserts
// on .literal's actual runtime type and value directly, which the older,
// more general IdentifiesPushablePredicatesForPruning test above never did
// (why this bug shipped unnoticed).
TEST(Optimizer, PushablePredicateLiteralIsUnwrappedFromImplicitCast) {
  auto plan = plan_for("SELECT region FROM read_parquet('/x.parquet') WHERE amount > 0", sales_schema());
  plan = optimize(std::move(plan));

  const LogicalScan* scan = find_scan(plan);
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->pushable_predicates().size(), 1u);
  const PushablePredicate& predicate = scan->pushable_predicates()[0];
  EXPECT_EQ(predicate.column_name, "amount");

  const auto* literal = dynamic_cast<const LiteralExpression*>(predicate.literal.get());
  ASSERT_NE(literal, nullptr) << "literal is " << predicate.literal->to_string()
                              << " -- still CAST-wrapped instead of unwrapped";
  ASSERT_TRUE(std::holds_alternative<double>(literal->value()) ||
              std::holds_alternative<std::int64_t>(literal->value()));
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

// annotate_scan()'s LogicalFilter branch special-cases a filter sitting
// directly on a LogicalAggregate (a HAVING filter): its predicate's
// ColumnExpression indices describe the aggregate's own output schema
// ([region, total] here), not the scan's -- neither collecting its columns
// into the scan's required_columns (which would misindex, e.g. treating
// scan index 1 as "the HAVING threshold column") nor treating it as a
// pushable file-level predicate (it runs after aggregation, over rows that
// don't exist at scan time) is correct. No optimizer test used HAVING at
// all before this.
// literal_as_double()/fold_arithmetic()/fold_numeric_comparison()
// (optimizer.cpp) are the double-literal folding counterparts of
// fold_arithmetic_int64()/fold_integer_comparison() -- simplify_expression()
// only reaches them when at least one operand's LiteralStorage isn't
// int64, a shape no prior optimizer test used (every existing folding test
// above uses pure int64 literals). Built directly (bypassing SQL/the
// binder) so a raw int64-storage-under-a-double-typed-literal mix --
// literal_as_double()'s own int64 branch, only reachable this way, real
// make_*() factories always pair storage/type correctly -- is exercised
// alongside the double-storage branch in the same expression tree.
TEST(Optimizer, ConstantFoldsMixedIntAndDoubleArithmetic) {
  Schema schema({Field{"a", int64_type(false)}});
  auto scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  auto int_literal =
      std::make_shared<LiteralExpression>(LiteralExpression(std::int64_t{2}, float64_type(false)));
  auto double_literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(2.5));
  auto sum = std::make_shared<BinaryExpression>(BinaryOperator::Add, int_literal, double_literal,
                                                float64_type(false));
  auto filter = std::make_shared<LogicalFilter>(scan, sum);

  LogicalPlanPtr optimized = optimize(filter);
  const auto* result_filter = dynamic_cast<const LogicalFilter*>(optimized.get());
  ASSERT_NE(result_filter, nullptr);
  const auto* literal = dynamic_cast<const LiteralExpression*>(result_filter->predicate().get());
  ASSERT_NE(literal, nullptr);
  ASSERT_TRUE(std::holds_alternative<double>(literal->value()));
  EXPECT_DOUBLE_EQ(std::get<double>(literal->value()), 4.5);
}

TEST(Optimizer, ConstantFoldsEveryDoubleArithmeticOperator) {
  Schema schema({Field{"a", int64_type(false)}});
  auto scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  const std::vector<std::tuple<BinaryOperator, double, double, double>> cases = {
      {BinaryOperator::Add, 1.5, 2.5, 4.0},
      {BinaryOperator::Subtract, 5.5, 2.5, 3.0},
      {BinaryOperator::Multiply, 2.5, 4.0, 10.0},
      {BinaryOperator::Divide, 9.0, 2.0, 4.5},
  };
  for (const auto& [op, left, right, expected] : cases) {
    auto left_literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(left));
    auto right_literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(right));
    auto binary = std::make_shared<BinaryExpression>(op, left_literal, right_literal, float64_type(false));
    auto filter = std::make_shared<LogicalFilter>(scan, binary);

    LogicalPlanPtr optimized = optimize(filter);
    const auto* result_filter = dynamic_cast<const LogicalFilter*>(optimized.get());
    ASSERT_NE(result_filter, nullptr);
    const auto* literal = dynamic_cast<const LiteralExpression*>(result_filter->predicate().get());
    ASSERT_NE(literal, nullptr) << "op index " << static_cast<int>(op);
    ASSERT_TRUE(std::holds_alternative<double>(literal->value()));
    EXPECT_DOUBLE_EQ(std::get<double>(literal->value()), expected) << "op index " << static_cast<int>(op);
  }
}

TEST(Optimizer, DoesNotFoldDoubleDivisionByZero) {
  Schema schema({Field{"a", int64_type(false)}});
  auto scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  auto left_literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(1.0));
  auto right_literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(0.0));
  auto divide = std::make_shared<BinaryExpression>(BinaryOperator::Divide, left_literal, right_literal,
                                                   float64_type(false));
  auto filter = std::make_shared<LogicalFilter>(scan, divide);

  LogicalPlanPtr optimized = optimize(filter);
  const auto* result_filter = dynamic_cast<const LogicalFilter*>(optimized.get());
  ASSERT_NE(result_filter, nullptr);
  // Left unfolded (a BinaryExpression, not collapsed to a literal).
  EXPECT_NE(dynamic_cast<const BinaryExpression*>(result_filter->predicate().get()), nullptr);
}

// Uses a LogicalProjection item, not a LogicalFilter predicate: a filter
// whose predicate folds to literal TRUE is removed entirely by rewrite_plan
// (see RemovesFilterThatFoldsToTrue), which would make a folded-true case
// here indistinguishable from "didn't fold at all" -- a projection item
// has no such special-casing, so the folded literal is always directly
// observable regardless of which way it folds.
TEST(Optimizer, ConstantFoldsEveryDoubleComparisonOperator) {
  Schema schema({Field{"a", int64_type(false)}});
  auto scan = std::make_shared<LogicalScan>(std::vector<std::string>{"/x.parquet"}, schema);
  const std::vector<std::tuple<BinaryOperator, double, double, bool>> cases = {
      {BinaryOperator::Equal, 1.5, 1.5, true},   {BinaryOperator::NotEqual, 1.5, 2.5, true},
      {BinaryOperator::Less, 1.5, 2.5, true},    {BinaryOperator::LessEqual, 2.5, 2.5, true},
      {BinaryOperator::Greater, 2.5, 1.5, true}, {BinaryOperator::GreaterEqual, 2.5, 2.5, true},
  };
  for (const auto& [op, left, right, expected] : cases) {
    auto left_literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(left));
    auto right_literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(right));
    auto binary = std::make_shared<BinaryExpression>(op, left_literal, right_literal, boolean_type(false));
    auto projection =
        std::make_shared<LogicalProjection>(scan, std::vector<NamedExpression>{{binary, "result"}});

    LogicalPlanPtr optimized = optimize(projection);
    const auto* result_projection = dynamic_cast<const LogicalProjection*>(optimized.get());
    ASSERT_NE(result_projection, nullptr);
    const auto* literal = dynamic_cast<const LiteralExpression*>(result_projection->items()[0].expr.get());
    ASSERT_NE(literal, nullptr) << "op index " << static_cast<int>(op);
    ASSERT_TRUE(std::holds_alternative<bool>(literal->value()));
    EXPECT_EQ(std::get<bool>(literal->value()), expected) << "op index " << static_cast<int>(op);
  }
}

// fold_integer_comparison()'s Greater case specifically -- the existing
// ConstantFoldsRemainingComparisonOperators test covers Equal/NotEqual/
// Less/LessEqual/GreaterEqual but never bare '>' with int64 literals.
TEST(Optimizer, ConstantFoldsIntegerGreaterComparison) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE a > 0 AND 3 > 2", two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  // "3 > 2" folds to TRUE, short-circuiting the AND down to just "a > 0".
  EXPECT_EQ(filter->predicate()->to_string(), "(a > 0)");
}

// simplify_expression()'s UnaryExpression branch rebuilds a new
// UnaryExpression when its operand changed but didn't collapse away (not a
// literal, not a double-negation) -- FoldsNotOfLiteralAndEliminatesDoubleNegation
// above only ever exercises the two collapsing cases, never this
// rebuild-and-keep path.
TEST(Optimizer, SimplifiesUnaryOperandWithoutCollapsingTheUnaryItself) {
  auto plan = plan_for("SELECT a FROM read_parquet('/x.parquet') WHERE NOT (a > 1 + 2)", two_column_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  EXPECT_EQ(filter->predicate()->to_string(), "NOT ((a > 3))");
}

// push_predicate_through_join()'s remaining_conjuncts-non-empty-AFTER-some-
// conjuncts-were-pushed path (optimizer.cpp's own `return
// std::make_shared<LogicalFilter>(std::move(new_join), ...)` line) needs a
// WHERE clause mixing at least one single-sided (pushable) conjunct with at
// least one cross-side (unpushable) conjunct -- DoesNotPushCrossSidePredicateThroughJoin
// above uses a cross-side-only predicate, which returns nullptr before ever
// reaching this line (see that function's own early-return when neither
// side got anything pushed).
TEST(Optimizer, PushesSingleSidedConjunctAndKeepsCrossSideConjunctAboveJoin) {
  auto plan = plan_for_join(
      "SELECT o_orderkey FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.o_custkey = c.c_custkey "
      "WHERE c_mktsegment = 'BUILDING' AND o_orderkey > c_custkey",
      std::vector<Schema>{orders_schema(), customer_schema()});
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  // The cross-side conjunct keeps a Filter sitting directly on the join.
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  EXPECT_NE(dynamic_cast<const LogicalJoin*>(filter->children()[0].get()), nullptr);
  EXPECT_EQ(filter->predicate()->to_string(), "(o_orderkey > c_custkey)");

  // The single-sided conjunct still got pushed all the way to customer's scan.
  const LogicalScan* customer_scan = find_scan_with_column(plan, "c_mktsegment");
  ASSERT_NE(customer_scan, nullptr);
  ASSERT_EQ(customer_scan->pushable_predicates().size(), 1u);
  EXPECT_EQ(customer_scan->pushable_predicates()[0].column_name, "c_mktsegment");
}

// rewrite_plan()'s final `throw PlanningError(...)` guards against a
// LogicalPlanNode subtype it doesn't recognize -- unreachable through any
// of the seven real subtypes (all handled), only reachable via a custom one
// built just for this test, the same technique
// expression_compiler_cpu_test.cpp's UnrecognizedExpressionTypeThrows uses
// for compile_expression_cpu()'s own equivalent guard.
namespace {
class UnknownLogicalPlanNode final : public LogicalPlanNode {
 public:
  explicit UnknownLogicalPlanNode(Schema schema) : schema_(std::move(schema)) {}
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "UnknownLogicalPlanNode"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {}; }

 private:
  Schema schema_;
};
}  // namespace

TEST(Optimizer, RejectsUnrecognizedLogicalPlanNodeType) {
  auto unknown = std::make_shared<UnknownLogicalPlanNode>(two_column_schema());
  EXPECT_THROW((void)optimize(unknown), PlanningError);
}

TEST(Optimizer, HavingFilterOnAggregateDoesNotPolluteScanColumnsOrPushablePredicates) {
  auto plan = plan_for(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING SUM(amount) > 50",
      sales_schema());
  plan = optimize(std::move(plan));

  const auto* projection = dynamic_cast<const LogicalProjection*>(plan.get());
  ASSERT_NE(projection, nullptr);
  const auto* filter = dynamic_cast<const LogicalFilter*>(projection->children()[0].get());
  ASSERT_NE(filter, nullptr);
  const auto* aggregate = dynamic_cast<const LogicalAggregate*>(filter->children()[0].get());
  ASSERT_NE(aggregate, nullptr);

  const LogicalScan* scan = find_scan(plan);
  ASSERT_NE(scan, nullptr);
  // region (GROUP BY key) + amount (SUM's argument) -- not "total" (the
  // aggregate's own output name) and not misindexed against the HAVING
  // predicate's aggregate-output column position.
  ASSERT_EQ(scan->required_columns().size(), 2u);
  for (const char* expected : {"region", "amount"}) {
    EXPECT_NE(std::find(scan->required_columns().begin(), scan->required_columns().end(), expected),
              scan->required_columns().end());
  }
  EXPECT_TRUE(scan->pushable_predicates().empty());
}

}  // namespace
}  // namespace kernellake
