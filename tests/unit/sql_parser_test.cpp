#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/sql/parser.hpp"

namespace kernellake::sql {
namespace {

TEST(SqlParser, ParsesGeneralMvpQuery) {
  const auto stmt = parse_sql(
      "SELECT region, SUM(amount) AS total_amount, COUNT(*) AS order_count "
      "FROM read_parquet('/data/sales/*.parquet') "
      "WHERE event_date >= DATE '2026-01-01' AND amount > 0 "
      "GROUP BY region "
      "LIMIT 100");

  ASSERT_EQ(stmt.from.paths.size(), 1u);
  EXPECT_EQ(stmt.from.paths[0], "/data/sales/*.parquet");
  ASSERT_EQ(stmt.select_list.size(), 3u);
  EXPECT_TRUE(std::holds_alternative<AstColumnRef>(stmt.select_list[0]->node));
  ASSERT_TRUE(std::holds_alternative<AstAggregate>(stmt.select_list[1]->node));
  EXPECT_EQ(std::get<AstAggregate>(stmt.select_list[1]->node).function, AstAggregateFunc::Sum);
  EXPECT_EQ(stmt.select_list[1]->alias, "total_amount");
  ASSERT_TRUE(std::holds_alternative<AstAggregate>(stmt.select_list[2]->node));
  EXPECT_EQ(std::get<AstAggregate>(stmt.select_list[2]->node).function, AstAggregateFunc::CountStar);

  ASSERT_NE(stmt.where, nullptr);
  ASSERT_EQ(stmt.group_by.size(), 1u);
  ASSERT_TRUE(stmt.limit.has_value());
  EXPECT_EQ(*stmt.limit, 100);
}

TEST(SqlParser, ParsesTpchQ6Shape) {
  const auto stmt = parse_sql(
      "SELECT SUM(l_extendedprice * l_discount) AS revenue "
      "FROM read_parquet('/data/tpch/lineitem/*.parquet') "
      "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
      "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24");

  ASSERT_EQ(stmt.select_list.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<AstAggregate>(stmt.select_list[0]->node));
  const auto& sum = std::get<AstAggregate>(stmt.select_list[0]->node);
  EXPECT_EQ(sum.function, AstAggregateFunc::Sum);
  ASSERT_TRUE(std::holds_alternative<AstBinary>(sum.argument->node));
  EXPECT_EQ(std::get<AstBinary>(sum.argument->node).op, AstBinaryOp::Multiply);

  ASSERT_NE(stmt.where, nullptr);
  ASSERT_TRUE(std::holds_alternative<AstBinary>(stmt.where->node));
  EXPECT_EQ(std::get<AstBinary>(stmt.where->node).op, AstBinaryOp::And);
}

TEST(SqlParser, ParsesMultiplePathArguments) {
  const auto stmt = parse_sql("SELECT a FROM read_parquet('/data/a.parquet', '/data/b.parquet')");
  ASSERT_EQ(stmt.from.paths.size(), 2u);
  EXPECT_EQ(stmt.from.paths[0], "/data/a.parquet");
  EXPECT_EQ(stmt.from.paths[1], "/data/b.parquet");
}

TEST(SqlParser, ParsesIsNullAndIsNotNull) {
  const auto is_null = parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a IS NULL");
  ASSERT_TRUE(std::holds_alternative<AstUnary>(is_null.where->node));
  EXPECT_EQ(std::get<AstUnary>(is_null.where->node).op, AstUnaryOp::IsNull);

  const auto is_not_null = parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a IS NOT NULL");
  ASSERT_TRUE(std::holds_alternative<AstUnary>(is_not_null.where->node));
  EXPECT_EQ(std::get<AstUnary>(is_not_null.where->node).op, AstUnaryOp::IsNotNull);
}

TEST(SqlParser, ParsesOrderByAndStar) {
  const auto stmt = parse_sql("SELECT * FROM read_parquet('/x.parquet') ORDER BY a DESC, b ASC");
  ASSERT_EQ(stmt.select_list.size(), 1u);
  EXPECT_TRUE(std::holds_alternative<AstStar>(stmt.select_list[0]->node));
  ASSERT_EQ(stmt.order_by.size(), 2u);
  EXPECT_FALSE(stmt.order_by[0].ascending);
  EXPECT_TRUE(stmt.order_by[1].ascending);
}

TEST(SqlParser, ParsesBooleanLiterals) {
  const auto stmt = parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE b = TRUE");
  ASSERT_TRUE(std::holds_alternative<AstBinary>(stmt.where->node));
  const auto& binary = std::get<AstBinary>(stmt.where->node);
  ASSERT_TRUE(std::holds_alternative<AstLiteral>(binary.right->node));
  const auto& literal = std::get<AstLiteral>(binary.right->node);
  EXPECT_EQ(literal.kind, AstLiteralKind::Boolean);
  EXPECT_TRUE(literal.bool_value);
}

TEST(SqlParser, RejectsMissingReadParquetSource) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM sales")), SqlError);
}

TEST(SqlParser, RejectsJoinsWithoutAliases) {
  // Both sides of a JOIN must be aliased -- unqualified column references
  // like `a.order_id` would otherwise have no way to pick a side.
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') JOIN read_parquet('/y.parquet') ON true")),
               SqlError);
}

TEST(SqlParser, ParsesTwoTableInnerJoin) {
  const auto stmt = parse_sql(
      "SELECT a.x, b.y FROM read_parquet('/x.parquet') AS a "
      "JOIN read_parquet('/y.parquet') AS b ON a.order_id = b.order_id");
  ASSERT_TRUE(stmt.join.has_value());
  EXPECT_EQ(stmt.join->left.paths, std::vector<std::string>{"/x.parquet"});
  EXPECT_EQ(stmt.join->left.alias, "a");
  EXPECT_EQ(stmt.join->right.paths, std::vector<std::string>{"/y.parquet"});
  EXPECT_EQ(stmt.join->right.alias, "b");
  ASSERT_NE(stmt.join->condition, nullptr);
  const auto* condition = std::get_if<AstBinary>(&stmt.join->condition->node);
  ASSERT_NE(condition, nullptr);
  EXPECT_EQ(condition->op, AstBinaryOp::Eq);
  const auto* left_ref = std::get_if<AstColumnRef>(&condition->left->node);
  ASSERT_NE(left_ref, nullptr);
  EXPECT_EQ(left_ref->name, "order_id");
  ASSERT_TRUE(left_ref->table.has_value());
  EXPECT_EQ(*left_ref->table, "a");

  ASSERT_EQ(stmt.select_list.size(), 2u);
  const auto* select_a = std::get_if<AstColumnRef>(&stmt.select_list[0]->node);
  ASSERT_NE(select_a, nullptr);
  EXPECT_EQ(select_a->name, "x");
  ASSERT_TRUE(select_a->table.has_value());
  EXPECT_EQ(*select_a->table, "a");
}

TEST(SqlParser, RejectsNonInnerJoin) {
  EXPECT_THROW((void)(parse_sql("SELECT a.x FROM read_parquet('/x.parquet') AS a "
                         "LEFT JOIN read_parquet('/y.parquet') AS b ON a.order_id = b.order_id")),
               SqlError);
}

TEST(SqlParser, RejectsCommaStyleJoin) {
  EXPECT_THROW((void)(parse_sql("SELECT a.x FROM read_parquet('/x.parquet') AS a, read_parquet('/y.parquet') AS b "
                         "WHERE a.order_id = b.order_id")),
               SqlError);
}

TEST(SqlParser, RejectsThreeReadParquetSources) {
  EXPECT_THROW((void)(parse_sql("SELECT a.x FROM read_parquet('/x.parquet') AS a "
                         "JOIN read_parquet('/y.parquet') AS b ON a.order_id = b.order_id "
                         "JOIN read_parquet('/z.parquet') AS c ON b.order_id = c.order_id")),
               SqlError);
}

TEST(SqlParser, RejectsUnsupportedFunction) {
  EXPECT_THROW((void)(parse_sql("SELECT UPPER(a) FROM read_parquet('/x.parquet')")), SqlError);
}

TEST(SqlParser, RejectsMalformedSql) {
  EXPECT_THROW((void)(parse_sql("SELECT FROM read_parquet('/x.parquet') WHERE")), SqlError);
}

TEST(SqlParser, RejectsInvalidDateLiteral) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE d >= DATE '2026-13-40'")), SqlError);
}

TEST(SqlParser, RejectsOffset) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') LIMIT 10 OFFSET 5")), SqlError);
}

}  // namespace
}  // namespace kernellake::sql
