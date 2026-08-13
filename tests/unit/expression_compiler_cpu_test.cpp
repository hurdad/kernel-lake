// literal_to_arrow_datum()/compile_expression_cpu() (expression_compiler_cpu.cpp)
// had no dedicated unit test file at all -- they were only ever exercised
// indirectly through real CPU-backend query execution, which never happens
// to touch every TypeId/BinaryOperator/UnaryOperator/DatePart case, the
// value/type-mismatch coercion branches in as_double()/as_int64() (only
// reachable via the raw LiteralExpression(value, type) constructor, not
// the make_*() factories real binder output always uses), or the
// "unreachable" defensive throws guarding against a future enumerator
// added without updating the corresponding switch. This file drives both
// functions directly with hand-built Expression trees.
#include "kernellake/execution_cpu/expression_compiler_cpu.hpp"

#include <gtest/gtest.h>

#include <arrow/compute/exec.h>

#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

// ---- literal_to_arrow_datum() ----

TEST(LiteralToArrowDatum, NullLiteralProducesNullScalarOfTheRequestedType) {
  const LiteralExpression literal = LiteralExpression::make_null(int64_type(true));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  ASSERT_TRUE(datum.is_scalar());
  EXPECT_FALSE(datum.scalar()->is_valid);
}

TEST(LiteralToArrowDatum, BooleanLiteral) {
  const arrow::Datum datum = literal_to_arrow_datum(LiteralExpression::make_bool(true));
  EXPECT_EQ(std::static_pointer_cast<arrow::BooleanScalar>(datum.scalar())->value, true);
}

TEST(LiteralToArrowDatum, Int64Literal) {
  const arrow::Datum datum = literal_to_arrow_datum(LiteralExpression::make_int64(42));
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(datum.scalar())->value, 42);
}

TEST(LiteralToArrowDatum, Int32Literal) {
  const LiteralExpression literal(std::int64_t{7}, int32_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_EQ(std::static_pointer_cast<arrow::Int32Scalar>(datum.scalar())->value, 7);
}

TEST(LiteralToArrowDatum, UInt32Literal) {
  const LiteralExpression literal(std::int64_t{7}, uint32_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_EQ(std::static_pointer_cast<arrow::UInt32Scalar>(datum.scalar())->value, 7u);
}

TEST(LiteralToArrowDatum, UInt64Literal) {
  const LiteralExpression literal(std::int64_t{7}, uint64_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_EQ(std::static_pointer_cast<arrow::UInt64Scalar>(datum.scalar())->value, 7u);
}

TEST(LiteralToArrowDatum, Float32Literal) {
  const LiteralExpression literal(1.5, float32_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_FLOAT_EQ(std::static_pointer_cast<arrow::FloatScalar>(datum.scalar())->value, 1.5F);
}

TEST(LiteralToArrowDatum, Float64Literal) {
  const arrow::Datum datum = literal_to_arrow_datum(LiteralExpression::make_float64(1.5));
  EXPECT_DOUBLE_EQ(std::static_pointer_cast<arrow::DoubleScalar>(datum.scalar())->value, 1.5);
}

TEST(LiteralToArrowDatum, StringLiteral) {
  const arrow::Datum datum = literal_to_arrow_datum(LiteralExpression::make_string("hello"));
  EXPECT_EQ(std::static_pointer_cast<arrow::StringScalar>(datum.scalar())->value->ToString(), "hello");
}

TEST(LiteralToArrowDatum, Date32Literal) {
  const arrow::Datum datum = literal_to_arrow_datum(LiteralExpression::make_date32(19723));
  EXPECT_EQ(std::static_pointer_cast<arrow::Date32Scalar>(datum.scalar())->value, 19723);
}

TEST(LiteralToArrowDatum, TimestampLiteral) {
  const arrow::Datum datum = literal_to_arrow_datum(LiteralExpression::make_timestamp(1704067200000000LL));
  EXPECT_EQ(std::static_pointer_cast<arrow::TimestampScalar>(datum.scalar())->value, 1704067200000000LL);
}

TEST(LiteralToArrowDatum, DecimalLiteralThrows) {
  const LiteralExpression literal(std::int64_t{0}, decimal_type(10, 2, false));
  EXPECT_THROW((void)literal_to_arrow_datum(literal), ExecutionError);
}

// as_double()/as_int64() coerce whichever LiteralStorage alternative is
// actually present, not just the one that "naturally" matches the
// requested TypeId -- only reachable by mismatching value and type via the
// raw constructor, since every make_*() factory always pairs them
// correctly. This is defensive coercion, not a real binder output shape,
// but the code exists and needs coverage.
TEST(LiteralToArrowDatum, Int64TypeWithBoolStorageCoercesToZeroOrOne) {
  const LiteralExpression literal(true, int64_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(datum.scalar())->value, 1);
}

TEST(LiteralToArrowDatum, Int64TypeWithDoubleStorageTruncates) {
  const LiteralExpression literal(3.9, int64_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(datum.scalar())->value, 3);
}

TEST(LiteralToArrowDatum, Int64TypeWithStringStorageFallsBackToZero) {
  const LiteralExpression literal(std::string("x"), int64_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(datum.scalar())->value, 0);
}

TEST(LiteralToArrowDatum, Float64TypeWithInt64StorageConverts) {
  const LiteralExpression literal(std::int64_t{4}, float64_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_DOUBLE_EQ(std::static_pointer_cast<arrow::DoubleScalar>(datum.scalar())->value, 4.0);
}

TEST(LiteralToArrowDatum, Float64TypeWithBoolStorageCoercesToZeroOrOne) {
  const LiteralExpression literal(false, float64_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_DOUBLE_EQ(std::static_pointer_cast<arrow::DoubleScalar>(datum.scalar())->value, 0.0);
}

TEST(LiteralToArrowDatum, Float64TypeWithStringStorageFallsBackToZero) {
  const LiteralExpression literal(std::string("x"), float64_type(false));
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  EXPECT_DOUBLE_EQ(std::static_pointer_cast<arrow::DoubleScalar>(datum.scalar())->value, 0.0);
}

// An out-of-range TypeId (never possible through the real enum, only via a
// raw static_cast) is rejected by to_arrow_type() (arrow_adapter.cpp,
// called first at the top of literal_to_arrow_datum()) before this
// function's own switch is ever reached -- its own "unreachable: unknown
// KernelLake TypeId in CPU expression compiler" throw is therefore truly
// dead code along this path, not something this test can cover. Still
// worth asserting that an invalid TypeId fails loudly end-to-end rather
// than silently miscompiling.
TEST(LiteralToArrowDatum, UnknownTypeIdFailsInToArrowTypeBeforeReachingThisFunctionsOwnSwitch) {
  DataType bogus_type;
  bogus_type.id = static_cast<TypeId>(255);
  bogus_type.nullable = false;
  const LiteralExpression literal(std::int64_t{0}, bogus_type);
  EXPECT_THROW((void)literal_to_arrow_datum(literal), PlanningError);
}

// ---- compile_expression_cpu(): BinaryExpression ----

TEST(CompileExpressionCpu, EveryBinaryOperatorMapsToItsArrowComputeFunctionName) {
  const ExpressionPtr left = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  const ExpressionPtr right = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(2));
  const std::vector<std::pair<BinaryOperator, std::string>> cases = {
      {BinaryOperator::Add, "add"},           {BinaryOperator::Subtract, "subtract"},
      {BinaryOperator::Multiply, "multiply"}, {BinaryOperator::Divide, "divide"},
      {BinaryOperator::Equal, "equal"},       {BinaryOperator::NotEqual, "not_equal"},
      {BinaryOperator::Less, "less"},         {BinaryOperator::LessEqual, "less_equal"},
      {BinaryOperator::Greater, "greater"},   {BinaryOperator::GreaterEqual, "greater_equal"},
      {BinaryOperator::And, "and_kleene"},    {BinaryOperator::Or, "or_kleene"},
  };
  for (const auto& [op, function_name] : cases) {
    const BinaryExpression binary(op, left, right, int64_type(false));
    const arrow::compute::Expression compiled = compile_expression_cpu(binary);
    ASSERT_TRUE(compiled.call() != nullptr) << function_name;
    EXPECT_EQ(compiled.call()->function_name, function_name);
  }
}

TEST(CompileExpressionCpu, UnknownBinaryOperatorHitsUnreachableThrow) {
  const ExpressionPtr left = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  const ExpressionPtr right = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(2));
  const BinaryExpression binary(static_cast<BinaryOperator>(255), left, right, int64_type(false));
  EXPECT_THROW((void)compile_expression_cpu(binary), ExecutionError);
}

// ---- compile_expression_cpu(): UnaryExpression ----

TEST(CompileExpressionCpu, EveryUnaryOperatorMapsToItsArrowComputeFunctionName) {
  const ExpressionPtr operand = std::make_shared<LiteralExpression>(LiteralExpression::make_bool(true));
  const std::vector<std::pair<UnaryOperator, std::string>> cases = {
      {UnaryOperator::Not, "invert"},
      {UnaryOperator::Negate, "negate"},
      {UnaryOperator::IsNull, "is_null"},
      {UnaryOperator::IsNotNull, "is_valid"},
  };
  for (const auto& [op, function_name] : cases) {
    const UnaryExpression unary(op, operand, boolean_type(false));
    const arrow::compute::Expression compiled = compile_expression_cpu(unary);
    ASSERT_TRUE(compiled.call() != nullptr) << function_name;
    EXPECT_EQ(compiled.call()->function_name, function_name);
  }
}

TEST(CompileExpressionCpu, UnknownUnaryOperatorHitsUnreachableThrow) {
  const ExpressionPtr operand = std::make_shared<LiteralExpression>(LiteralExpression::make_bool(true));
  const UnaryExpression unary(static_cast<UnaryOperator>(255), operand, boolean_type(false));
  EXPECT_THROW((void)compile_expression_cpu(unary), ExecutionError);
}

// ---- compile_expression_cpu(): BETWEEN ----

TEST(CompileExpressionCpu, BetweenCompilesToAndedGreaterEqualLessEqual) {
  const ExpressionPtr value = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(5));
  const ExpressionPtr lower = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(0));
  const ExpressionPtr upper = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(10));
  const BetweenExpression between(value, lower, upper);
  const arrow::compute::Expression compiled = compile_expression_cpu(between);
  ASSERT_TRUE(compiled.call() != nullptr);
  EXPECT_EQ(compiled.call()->function_name, "and_kleene");
}

// ---- compile_expression_cpu(): CAST ----

TEST(CompileExpressionCpu, NumericCastCompilesToCastCall) {
  const ExpressionPtr operand = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  const CastExpression cast(operand, float64_type(false));
  const arrow::compute::Expression compiled = compile_expression_cpu(cast);
  ASSERT_TRUE(compiled.call() != nullptr);
  EXPECT_EQ(compiled.call()->function_name, "cast");
}

TEST(CompileExpressionCpu, CastToDecimalThrows) {
  const ExpressionPtr operand = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  const CastExpression cast(operand, decimal_type(10, 2, false));
  EXPECT_THROW((void)compile_expression_cpu(cast), ExecutionError);
}

TEST(CompileExpressionCpu, CastToStringThrows) {
  const ExpressionPtr operand = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  const CastExpression cast(operand, string_type(false));
  EXPECT_THROW((void)compile_expression_cpu(cast), ExecutionError);
}

// ---- compile_expression_cpu(): CASE ----

TEST(CompileExpressionCpu, CaseWithElseBranchCompilesToCaseWhenCall) {
  const ExpressionPtr condition = std::make_shared<LiteralExpression>(LiteralExpression::make_bool(true));
  const ExpressionPtr result = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  const ExpressionPtr else_branch = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(0));
  const CaseExpression case_expr({CaseExpression::WhenThen{condition, result}}, else_branch,
                                 int64_type(false));
  const arrow::compute::Expression compiled = compile_expression_cpu(case_expr);
  ASSERT_TRUE(compiled.call() != nullptr);
  EXPECT_EQ(compiled.call()->function_name, "case_when");
  // condition struct + 1 result + 1 else = 3 call arguments.
  EXPECT_EQ(compiled.call()->arguments.size(), 3u);
}

TEST(CompileExpressionCpu, CaseWithoutElseBranchOmitsTrailingArgument) {
  const ExpressionPtr condition = std::make_shared<LiteralExpression>(LiteralExpression::make_bool(true));
  const ExpressionPtr result = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  const CaseExpression case_expr({CaseExpression::WhenThen{condition, result}}, nullptr, int64_type(false));
  const arrow::compute::Expression compiled = compile_expression_cpu(case_expr);
  ASSERT_TRUE(compiled.call() != nullptr);
  // condition struct + 1 result, no else = 2 call arguments.
  EXPECT_EQ(compiled.call()->arguments.size(), 2u);
}

// ---- compile_expression_cpu(): LIKE ----

TEST(CompileExpressionCpu, LikeCompilesToMatchLike) {
  const ExpressionPtr value = std::make_shared<LiteralExpression>(LiteralExpression::make_string("abc"));
  const LikeExpression like(value, "a%", false);
  const arrow::compute::Expression compiled = compile_expression_cpu(like);
  ASSERT_TRUE(compiled.call() != nullptr);
  EXPECT_EQ(compiled.call()->function_name, "match_like");
}

TEST(CompileExpressionCpu, NotLikeWrapsMatchLikeInInvert) {
  const ExpressionPtr value = std::make_shared<LiteralExpression>(LiteralExpression::make_string("abc"));
  const LikeExpression like(value, "a%", true);
  const arrow::compute::Expression compiled = compile_expression_cpu(like);
  ASSERT_TRUE(compiled.call() != nullptr);
  EXPECT_EQ(compiled.call()->function_name, "invert");
}

// ---- compile_expression_cpu(): EXTRACT ----

TEST(CompileExpressionCpu, EveryDatePartMapsToItsArrowComputeFunctionName) {
  const ExpressionPtr operand = std::make_shared<LiteralExpression>(LiteralExpression::make_date32(19723));
  const std::vector<std::pair<DatePart, std::string>> cases = {
      {DatePart::Year, "year"},
      {DatePart::Month, "month"},
      {DatePart::Day, "day"},
  };
  for (const auto& [part, function_name] : cases) {
    const ExtractExpression extract(part, operand, int64_type(false));
    const arrow::compute::Expression compiled = compile_expression_cpu(extract);
    ASSERT_TRUE(compiled.call() != nullptr) << function_name;
    EXPECT_EQ(compiled.call()->function_name, function_name);
  }
}

// ---- compile_expression_cpu(): unrecognized node type ----

namespace {
// A minimal Expression subtype that isn't any of the dynamic_cast targets
// compile_expression_cpu() knows about -- the only way to reach its final
// catch-all throw, since every real Expression subclass in
// kernellake/expression/expression.hpp is already handled.
class UnknownExpression final : public Expression {
 public:
  [[nodiscard]] const DataType& result_type() const override {
    static const DataType type = boolean_type(false);
    return type;
  }
  [[nodiscard]] std::string to_string() const override { return "UNKNOWN"; }
  [[nodiscard]] std::string structural_key() const override { return "UNKNOWN"; }
};
}  // namespace

TEST(CompileExpressionCpu, UnrecognizedExpressionTypeThrows) {
  const UnknownExpression expr;
  EXPECT_THROW((void)compile_expression_cpu(expr), ExecutionError);
}

}  // namespace
}  // namespace kernellake
