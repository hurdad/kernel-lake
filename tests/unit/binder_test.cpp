#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/planner/binder.hpp"
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

TEST(Binder, BindsGeneralMvpQuery) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total_amount, COUNT(*) AS order_count "
      "FROM read_parquet('/data/sales/*.parquet') "
      "WHERE event_date >= DATE '2026-01-01' AND amount > 0 "
      "GROUP BY region "
      "LIMIT 100");
  const BoundQuery bound = bind_query(stmt, sales_schema());

  ASSERT_EQ(bound.select_list.size(), 3u);
  EXPECT_EQ(bound.select_list[0].output_name, "region");
  EXPECT_EQ(bound.select_list[1].output_name, "total_amount");
  EXPECT_EQ(bound.select_list[1].expr->result_type().id, TypeId::Float64);
  EXPECT_EQ(bound.select_list[2].output_name, "order_count");
  EXPECT_EQ(bound.select_list[2].expr->result_type().id, TypeId::Int64);
  EXPECT_TRUE(bound.is_aggregate_query);
  ASSERT_NE(bound.where, nullptr);
  EXPECT_EQ(bound.where->result_type().id, TypeId::Boolean);
  ASSERT_EQ(bound.group_by.size(), 1u);
  EXPECT_EQ(bound.limit, 100);
  EXPECT_EQ(bound.output_schema.field_count(), 3u);
}

TEST(Binder, BindsTpchQ6ArithmeticOnGpuShape) {
  const auto stmt = sql::parse_sql(
      "SELECT SUM(l_extendedprice * l_discount) AS revenue "
      "FROM read_parquet('/data/tpch/lineitem/*.parquet') "
      "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
      "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24");
  const BoundQuery bound = bind_query(stmt, lineitem_schema());

  ASSERT_EQ(bound.select_list.size(), 1u);
  EXPECT_EQ(bound.select_list[0].output_name, "revenue");
  EXPECT_EQ(bound.select_list[0].expr->result_type().id, TypeId::Float64);
  EXPECT_TRUE(bound.is_aggregate_query);
  ASSERT_NE(bound.where, nullptr);
  EXPECT_EQ(bound.where->result_type().id, TypeId::Boolean);
}

TEST(Binder, RejectsUnknownColumn) {
  const auto stmt = sql::parse_sql("SELECT nonexistent FROM read_parquet('/x.parquet')");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsUngroupedColumnInAggregateQuery) {
  const auto stmt =
      sql::parse_sql("SELECT region, amount, SUM(amount) FROM read_parquet('/x.parquet') GROUP BY region");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, AllowsGroupedColumnInAggregateQuery) {
  const auto stmt =
      sql::parse_sql("SELECT region, SUM(amount) FROM read_parquet('/x.parquet') GROUP BY region");
  EXPECT_NO_THROW(bind_query(stmt, sales_schema()));
}

TEST(Binder, RejectsIncompatibleComparison) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region > 5");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsNonBooleanWhere) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsDuplicateOutputNames) {
  const auto stmt = sql::parse_sql("SELECT region AS x, amount AS x FROM read_parquet('/x.parquet')");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsAggregateInWhere) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE SUM(amount) > 0");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, ExpandsStar) {
  const auto stmt = sql::parse_sql("SELECT * FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.select_list.size(), 3u);
  EXPECT_EQ(bound.select_list[0].output_name, "region");
  EXPECT_EQ(bound.select_list[2].output_name, "event_date");
}

TEST(Binder, InsertsSafeNumericCastForComparison) {
  Schema schema({Field{"a", int32_type(false)}});
  const auto stmt = sql::parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a > 1.5");
  const BoundQuery bound = bind_query(stmt, schema);
  ASSERT_NE(bound.where, nullptr);
  EXPECT_EQ(bound.where->result_type().id, TypeId::Boolean);
}

TEST(Binder, NullLiteralComparisonBindsWithoutError) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount = NULL");
  EXPECT_NO_THROW(bind_query(stmt, sales_schema()));
}

TEST(Binder, LikeRequiresStringOperandAndLiteralPattern) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region LIKE 'A%'");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_NE(bound.where, nullptr);
  const auto* like = dynamic_cast<const LikeExpression*>(bound.where.get());
  ASSERT_NE(like, nullptr);
  EXPECT_EQ(like->pattern(), "A%");
  EXPECT_FALSE(like->negated());

  const auto not_stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region NOT LIKE 'A%'");
  const BoundQuery not_bound = bind_query(not_stmt, sales_schema());
  const auto* not_like = dynamic_cast<const LikeExpression*>(not_bound.where.get());
  ASSERT_NE(not_like, nullptr);
  EXPECT_TRUE(not_like->negated());

  const auto numeric_stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount LIKE 'A%'");
  EXPECT_THROW(bind_query(numeric_stmt, sales_schema()), BindingError);
}

TEST(Binder, InDesugarsIntoOrChainOfEqualityComparisons) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region IN ('A', 'B', 'C')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_NE(bound.where, nullptr);
  // (region = 'A') OR (region = 'B') OR (region = 'C'), left-associated.
  const auto* outer_or = dynamic_cast<const BinaryExpression*>(bound.where.get());
  ASSERT_NE(outer_or, nullptr);
  EXPECT_EQ(outer_or->op(), BinaryOperator::Or);
  const auto* inner_or = dynamic_cast<const BinaryExpression*>(outer_or->left().get());
  ASSERT_NE(inner_or, nullptr);
  EXPECT_EQ(inner_or->op(), BinaryOperator::Or);
  const auto* first_eq = dynamic_cast<const BinaryExpression*>(inner_or->left().get());
  ASSERT_NE(first_eq, nullptr);
  EXPECT_EQ(first_eq->op(), BinaryOperator::Equal);
}

TEST(Binder, NotInNegatesTheDesugaredOrChain) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region NOT IN ('A', 'B')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  const auto* not_expr = dynamic_cast<const UnaryExpression*>(bound.where.get());
  ASSERT_NE(not_expr, nullptr);
  EXPECT_EQ(not_expr->op(), UnaryOperator::Not);
}

TEST(Binder, CaseRequiresBooleanConditionsAndUnifiesResultTypes) {
  const auto stmt = sql::parse_sql(
      "SELECT CASE WHEN amount > 500 THEN 1 WHEN amount > 100 THEN 2 ELSE 0 END AS bucket "
      "FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.select_list.size(), 1u);
  const auto* case_expr = dynamic_cast<const CaseExpression*>(bound.select_list[0].expr.get());
  ASSERT_NE(case_expr, nullptr);
  EXPECT_EQ(case_expr->when_then().size(), 2u);
  EXPECT_NE(case_expr->else_branch(), nullptr);
  EXPECT_EQ(case_expr->result_type().id, TypeId::Int64);
  EXPECT_FALSE(case_expr->result_type().nullable);  // ELSE present, all branches non-null

  const auto no_else_stmt =
      sql::parse_sql("SELECT CASE WHEN amount > 500 THEN 1 END AS bucket FROM read_parquet('/x.parquet')");
  const BoundQuery no_else_bound = bind_query(no_else_stmt, sales_schema());
  const auto* no_else_case = dynamic_cast<const CaseExpression*>(no_else_bound.select_list[0].expr.get());
  ASSERT_NE(no_else_case, nullptr);
  EXPECT_TRUE(no_else_case->result_type().nullable);  // no ELSE -> NULL is possible

  const auto non_boolean_condition_stmt =
      sql::parse_sql("SELECT CASE WHEN amount THEN 1 ELSE 0 END AS bucket FROM read_parquet('/x.parquet')");
  EXPECT_THROW(bind_query(non_boolean_condition_stmt, sales_schema()), BindingError);

  const auto incompatible_branches_stmt = sql::parse_sql(
      "SELECT CASE WHEN amount > 500 THEN 'high' ELSE 0 END AS bucket FROM read_parquet('/x.parquet')");
  EXPECT_THROW(bind_query(incompatible_branches_stmt, sales_schema()), BindingError);
}

TEST(Binder, CastMapsSqlTypeNamesToDataTypes) {
  const auto stmt = sql::parse_sql("SELECT CAST(amount AS BIGINT) AS x FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  const auto* cast = dynamic_cast<const CastExpression*>(bound.select_list[0].expr.get());
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(cast->result_type().id, TypeId::Int64);

  // hsql's grammar requires an explicit length for VARCHAR (bare VARCHAR
  // with no length is a parse error, unlike BIGINT/INT/etc.); the length
  // itself is unused by KernelLake's binder, which only has one unbounded
  // String type.
  const auto varchar_stmt =
      sql::parse_sql("SELECT CAST(amount AS VARCHAR(255)) AS x FROM read_parquet('/x.parquet')");
  const BoundQuery varchar_bound = bind_query(varchar_stmt, sales_schema());
  const auto* varchar_cast = dynamic_cast<const CastExpression*>(varchar_bound.select_list[0].expr.get());
  ASSERT_NE(varchar_cast, nullptr);
  EXPECT_EQ(varchar_cast->result_type().id, TypeId::String);

  const auto decimal_stmt =
      sql::parse_sql("SELECT CAST(amount AS DECIMAL) AS x FROM read_parquet('/x.parquet')");
  EXPECT_THROW(bind_query(decimal_stmt, sales_schema()), BindingError);
}

TEST(Binder, GroupByResolvesSelectListAliasForComputedExpressions) {
  // "bucket" is not a base-table column -- it can only resolve against the
  // SELECT-list alias, since that's the only way to name a computed GROUP
  // BY key like a CASE expression.
  const auto stmt = sql::parse_sql(
      "SELECT CASE WHEN amount > 500 THEN 'high' ELSE 'low' END AS bucket, COUNT(*) AS n "
      "FROM read_parquet('/x.parquet') GROUP BY bucket");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.group_by.size(), 1u);
  EXPECT_NE(dynamic_cast<const CaseExpression*>(bound.group_by[0].get()), nullptr);

  const auto unknown_alias_stmt = sql::parse_sql(
      "SELECT region, COUNT(*) AS n FROM read_parquet('/x.parquet') GROUP BY nonexistent_alias");
  EXPECT_THROW(bind_query(unknown_alias_stmt, sales_schema()), BindingError);
}

TEST(Binder, GroupByPrefersBaseColumnOverSameNamedAlias) {
  // "region" is both a real base-table column and this query's own alias
  // for it -- resolving against the base column either way gives the same
  // answer, but this pins down that base-table resolution is tried first.
  const auto stmt = sql::parse_sql(
      "SELECT region AS region, COUNT(*) AS n FROM read_parquet('/x.parquet') GROUP BY region");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.group_by.size(), 1u);
  const auto* column = dynamic_cast<const ColumnExpression*>(bound.group_by[0].get());
  ASSERT_NE(column, nullptr);
  EXPECT_EQ(column->name(), "region");
}

}  // namespace
}  // namespace kernellake
