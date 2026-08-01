#include <gtest/gtest.h>

#include "kernellake/expression/expression.hpp"

namespace kernellake {
namespace {

TEST(Expression, ColumnToString) {
  ColumnExpression column("region", 0, string_type());
  EXPECT_EQ(column.to_string(), "region");
  EXPECT_EQ(column.result_type().id, TypeId::String);
}

TEST(Expression, ArithmeticExpressionMatchesTpchShape) {
  // l_extendedprice * l_discount
  auto price = std::make_shared<ColumnExpression>("l_extendedprice", 0, float64_type());
  auto discount = std::make_shared<ColumnExpression>("l_discount", 1, float64_type());
  BinaryExpression product(BinaryOperator::Multiply, price, discount, float64_type());

  EXPECT_EQ(product.to_string(), "(l_extendedprice * l_discount)");
  EXPECT_EQ(product.result_type().id, TypeId::Float64);
  EXPECT_TRUE(is_arithmetic(product.op()));
  EXPECT_FALSE(is_comparison(product.op()));
}

TEST(Expression, BetweenExpressionToString) {
  auto discount = std::make_shared<ColumnExpression>("l_discount", 0, float64_type());
  auto lower = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(0.05));
  auto upper = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(0.07));
  BetweenExpression between(discount, lower, upper);

  EXPECT_EQ(between.to_string(), "l_discount BETWEEN 0.050000 AND 0.070000");
  EXPECT_EQ(between.result_type().id, TypeId::Boolean);
}

TEST(Expression, AggregateExpressionToString) {
  auto amount = std::make_shared<ColumnExpression>("amount", 0, float64_type());
  AggregateExpression sum(AggregateFunction::Sum, amount, float64_type());
  EXPECT_EQ(sum.to_string(), "SUM(amount)");

  AggregateExpression count_star(AggregateFunction::CountStar, nullptr, int64_type(false));
  EXPECT_EQ(count_star.to_string(), "COUNT(*)");
}

TEST(Expression, LiteralNullRoundTrip) {
  LiteralExpression null_literal = LiteralExpression::make_null(int64_type());
  EXPECT_TRUE(null_literal.is_null());
  EXPECT_EQ(null_literal.to_string(), "NULL");
}

TEST(Expression, UnaryIsNullToString) {
  auto column = std::make_shared<ColumnExpression>("discount", 0, float64_type());
  UnaryExpression is_null(UnaryOperator::IsNull, column, boolean_type(false));
  EXPECT_EQ(is_null.to_string(), "discount IS NULL");
}

TEST(Expression, CastExpressionToString) {
  auto column = std::make_shared<ColumnExpression>("id", 0, int32_type());
  CastExpression cast(column, int64_type());
  EXPECT_EQ(cast.to_string(), "CAST(id AS INT64)");
}

}  // namespace
}  // namespace kernellake
