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

// UnaryExpression::to_string()'s own switch has a case per UnaryOperator --
// UnaryIsNullToString above only ever exercised IsNull.
TEST(Expression, UnaryExpressionToStringCoversEveryOperator) {
  auto column = std::make_shared<ColumnExpression>("flag", 0, boolean_type());
  EXPECT_EQ(UnaryExpression(UnaryOperator::Not, column, boolean_type(false)).to_string(), "NOT (flag)");
  EXPECT_EQ(UnaryExpression(UnaryOperator::Negate, column, boolean_type(false)).to_string(), "-flag");
  EXPECT_EQ(UnaryExpression(UnaryOperator::IsNotNull, column, boolean_type(false)).to_string(),
            "flag IS NOT NULL");
}

TEST(Expression, UnaryExpressionStructuralKeyCoversEveryOperator) {
  // ColumnExpression::structural_key() encodes "#<column_index>", not the
  // bare name -- see structural_key()'s own class-level doc comment.
  auto column = std::make_shared<ColumnExpression>("flag", 0, boolean_type());
  EXPECT_EQ(UnaryExpression(UnaryOperator::Not, column, boolean_type(false)).structural_key(), "NOT (#0)");
  EXPECT_EQ(UnaryExpression(UnaryOperator::Negate, column, boolean_type(false)).structural_key(), "-#0");
  EXPECT_EQ(UnaryExpression(UnaryOperator::IsNotNull, column, boolean_type(false)).structural_key(),
            "#0 IS NOT NULL");
}

// The standalone free kernellake::to_string(UnaryOperator) (expression.cpp) is
// distinct from UnaryExpression::to_string()'s own switch above -- neither
// exercises the other -- and was never called directly by any prior test.
TEST(Expression, FreeToStringCoversEveryUnaryOperator) {
  EXPECT_EQ(to_string(UnaryOperator::Not), "NOT");
  EXPECT_EQ(to_string(UnaryOperator::Negate), "-");
  EXPECT_EQ(to_string(UnaryOperator::IsNull), "IS NULL");
  EXPECT_EQ(to_string(UnaryOperator::IsNotNull), "IS NOT NULL");
}

TEST(Expression, FreeToStringCoversEveryBinaryOperator) {
  EXPECT_EQ(to_string(BinaryOperator::Add), "+");
  EXPECT_EQ(to_string(BinaryOperator::Subtract), "-");
  EXPECT_EQ(to_string(BinaryOperator::Multiply), "*");
  EXPECT_EQ(to_string(BinaryOperator::Divide), "/");
  EXPECT_EQ(to_string(BinaryOperator::Equal), "=");
  EXPECT_EQ(to_string(BinaryOperator::NotEqual), "!=");
  EXPECT_EQ(to_string(BinaryOperator::Less), "<");
  EXPECT_EQ(to_string(BinaryOperator::LessEqual), "<=");
  EXPECT_EQ(to_string(BinaryOperator::Greater), ">");
  EXPECT_EQ(to_string(BinaryOperator::GreaterEqual), ">=");
  EXPECT_EQ(to_string(BinaryOperator::And), "AND");
  EXPECT_EQ(to_string(BinaryOperator::Or), "OR");
}

TEST(Expression, FreeToStringCoversEveryAggregateFunction) {
  EXPECT_EQ(to_string(AggregateFunction::Sum), "SUM");
  EXPECT_EQ(to_string(AggregateFunction::Count), "COUNT");
  EXPECT_EQ(to_string(AggregateFunction::CountStar), "COUNT");
  EXPECT_EQ(to_string(AggregateFunction::Min), "MIN");
  EXPECT_EQ(to_string(AggregateFunction::Max), "MAX");
  EXPECT_EQ(to_string(AggregateFunction::Avg), "AVG");
}

// Every to_string(<enum>) overload in expression.cpp falls through an
// otherwise-exhaustive switch to a defensive `return "?";` for a value with
// no matching case -- only reachable via an out-of-range static_cast, never
// through the real enum, but exercised here to confirm the fallback itself
// actually returns "?" rather than being genuinely dead code.
TEST(Expression, UnknownEnumeratorsFallBackToQuestionMark) {
  EXPECT_EQ(to_string(static_cast<BinaryOperator>(255)), "?");
  EXPECT_EQ(to_string(static_cast<UnaryOperator>(255)), "?");
  EXPECT_EQ(to_string(static_cast<DatePart>(255)), "?");
  EXPECT_EQ(to_string(static_cast<AggregateFunction>(255)), "?");

  auto column = std::make_shared<ColumnExpression>("flag", 0, boolean_type());
  const UnaryExpression unknown_unary(static_cast<UnaryOperator>(255), column, boolean_type(false));
  EXPECT_EQ(unknown_unary.to_string(), "?");
  EXPECT_EQ(unknown_unary.structural_key(), "?");
}

TEST(Expression, CastExpressionToString) {
  auto column = std::make_shared<ColumnExpression>("id", 0, int32_type());
  CastExpression cast(column, int64_type());
  EXPECT_EQ(cast.to_string(), "CAST(id AS INT64)");
}

TEST(Expression, BinaryExpressionStructuralKeyMatchesForStructurallyIdenticalExpressions) {
  auto left_a = std::make_shared<ColumnExpression>("amount", 0, float64_type());
  auto right_a = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(1.0));
  BinaryExpression first(BinaryOperator::Greater, left_a, right_a, boolean_type(false));

  auto left_b = std::make_shared<ColumnExpression>("amount", 0, float64_type());
  auto right_b = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(1.0));
  BinaryExpression second(BinaryOperator::Greater, left_b, right_b, boolean_type(false));

  EXPECT_EQ(first.structural_key(), second.structural_key());
}

// structural_key() is the mechanism that lets the binder/logical planner
// distinguish `a.x` from `b.x` after a JOIN (e.g. GROUP BY key matching,
// aggregate-slot dedup): to_string() alone collapses both to the identical
// bare name "x", but structural_key() additionally encodes the resolved
// column_index(), which differs between the two sides. Direct coverage --
// previously only exercised indirectly through the binder/optimizer.
TEST(Expression, BinaryExpressionStructuralKeyDiffersForColumnsWithSameNameDifferentIndex) {
  auto left_side = std::make_shared<ColumnExpression>("x", 0, float64_type());
  auto right_side = std::make_shared<ColumnExpression>("x", 5, float64_type());
  auto literal = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(1.0));

  BinaryExpression left_comparison(BinaryOperator::Greater, left_side, literal, boolean_type(false));
  BinaryExpression right_comparison(BinaryOperator::Greater, right_side, literal, boolean_type(false));

  EXPECT_NE(left_comparison.structural_key(), right_comparison.structural_key());
  EXPECT_EQ(left_comparison.to_string(), right_comparison.to_string());
}

TEST(Expression, UnaryExpressionStructuralKeyDiffersForColumnsWithSameNameDifferentIndex) {
  auto left_side = std::make_shared<ColumnExpression>("flag", 0, boolean_type());
  auto right_side = std::make_shared<ColumnExpression>("flag", 3, boolean_type());
  UnaryExpression left_is_null(UnaryOperator::IsNull, left_side, boolean_type(false));
  UnaryExpression right_is_null(UnaryOperator::IsNull, right_side, boolean_type(false));

  EXPECT_NE(left_is_null.structural_key(), right_is_null.structural_key());
  EXPECT_EQ(left_is_null.to_string(), right_is_null.to_string());
}

TEST(Expression, AggregateExpressionStructuralKeyDiffersForColumnsWithSameNameDifferentIndex) {
  auto left_amount = std::make_shared<ColumnExpression>("amount", 0, float64_type());
  auto right_amount = std::make_shared<ColumnExpression>("amount", 4, float64_type());
  AggregateExpression left_sum(AggregateFunction::Sum, left_amount, float64_type());
  AggregateExpression right_sum(AggregateFunction::Sum, right_amount, float64_type());

  EXPECT_NE(left_sum.structural_key(), right_sum.structural_key());
  EXPECT_EQ(left_sum.to_string(), right_sum.to_string());
}

TEST(Expression, IsLogicalIdentifiesAndOr) {
  EXPECT_TRUE(is_logical(BinaryOperator::And));
  EXPECT_TRUE(is_logical(BinaryOperator::Or));
  EXPECT_FALSE(is_logical(BinaryOperator::Add));
  EXPECT_FALSE(is_logical(BinaryOperator::Greater));
}

TEST(Expression, DatePartToString) {
  EXPECT_EQ(to_string(DatePart::Year), "YEAR");
  EXPECT_EQ(to_string(DatePart::Month), "MONTH");
  EXPECT_EQ(to_string(DatePart::Day), "DAY");
}

}  // namespace
}  // namespace kernellake
