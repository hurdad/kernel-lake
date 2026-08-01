#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>

#include "kernellake/execution/expression_compiler.hpp"

namespace kernellake {
namespace {

template <typename T>
std::vector<T> copy_to_host(const cudf::column_view& view) {
  std::vector<T> host(static_cast<std::size_t>(view.size()));
  cudaMemcpy(host.data(), view.data<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return host;
}

std::unique_ptr<cudf::column> filled_int32_column(int32_t value, cudf::size_type num_rows) {
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto scalar = cudf::make_fixed_width_scalar<int32_t>(value);
  cudf::mutable_column_view view = column->mutable_view();
  cudf::fill_in_place(view, 0, num_rows, *scalar);
  return column;
}

std::unique_ptr<cudf::column> filled_float64_column(double value, cudf::size_type num_rows) {
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, num_rows);
  auto scalar = cudf::make_fixed_width_scalar<double>(value);
  cudf::mutable_column_view view = column->mutable_view();
  cudf::fill_in_place(view, 0, num_rows, *scalar);
  return column;
}

TEST(ExpressionCompiler, ComparisonProducesCorrectBooleanColumn) {
  // column a = 5 for all 20 rows; compile and evaluate `a > 3`.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(5, 20));
  cudf::table table(std::move(columns));

  // cudf::ast requires exact-matching operand types (no implicit numeric
  // coercion at the kernel level) -- mirror what the real binder would
  // produce by casting the narrower int32 column up to int64 explicitly.
  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto a_as_int64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto three = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(3));
  BinaryExpression greater(BinaryOperator::Greater, a_as_int64, three, boolean_type(false));

  ExpressionCompiler compiler;
  const cudf::ast::expression& compiled = compiler.compile(greater);
  std::unique_ptr<cudf::column> result = cudf::compute_column(table.view(), compiled);

  ASSERT_EQ(result->type().id(), cudf::type_id::BOOL8);
  ASSERT_EQ(result->size(), 20);

  cudaDeviceSynchronize();
  // BOOL8 is one byte per value; std::vector<bool> is bit-packed and has no
  // .data(), so copy as raw bytes instead.
  for (unsigned char value : copy_to_host<unsigned char>(result->view())) EXPECT_TRUE(value != 0);
}

TEST(ExpressionCompiler, CompilesTpchQ6FilterShapeAndArithmetic) {
  // WHERE l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24, plus the
  // projection expression l_extendedprice * l_discount -- the exact shape
  // TPC-H Q6 needs, evaluated against real GPU columns.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_float64_column(100.0, 10));  // l_extendedprice (col 0)
  columns.push_back(filled_float64_column(0.06, 10));   // l_discount (col 1)
  columns.push_back(filled_int32_column(10, 10));       // l_quantity (col 2)
  cudf::table table(std::move(columns));

  auto discount = std::make_shared<ColumnExpression>("l_discount", 1, float64_type(false));
  auto lower = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(0.05));
  auto upper = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(0.07));
  auto between = std::make_shared<BetweenExpression>(discount, lower, upper);

  // l_quantity is INT32; cast to INT64 to match the literal's type, exactly
  // as the real binder would (cudf::ast requires exact-matching operands).
  auto quantity = std::make_shared<ColumnExpression>("l_quantity", 2, int32_type(false));
  auto quantity_as_int64 = std::make_shared<CastExpression>(quantity, int64_type(false));
  auto twenty_four = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(24));
  auto quantity_cmp = std::make_shared<BinaryExpression>(BinaryOperator::Less, quantity_as_int64,
                                                          twenty_four, boolean_type(false));

  BinaryExpression predicate(BinaryOperator::And, between, quantity_cmp, boolean_type(false));

  ExpressionCompiler filter_compiler;
  std::unique_ptr<cudf::column> mask =
      cudf::compute_column(table.view(), filter_compiler.compile(predicate));
  EXPECT_EQ(mask->type().id(), cudf::type_id::BOOL8);
  EXPECT_EQ(mask->size(), 10);

  auto extendedprice = std::make_shared<ColumnExpression>("l_extendedprice", 0, float64_type(false));
  BinaryExpression revenue_expr(BinaryOperator::Multiply, extendedprice, discount, float64_type(false));

  ExpressionCompiler projection_compiler;
  std::unique_ptr<cudf::column> revenue =
      cudf::compute_column(table.view(), projection_compiler.compile(revenue_expr));
  ASSERT_EQ(revenue->type().id(), cudf::type_id::FLOAT64);

  cudaDeviceSynchronize();
  for (double value : copy_to_host<double>(revenue->view())) EXPECT_DOUBLE_EQ(value, 6.0);  // 100.0*0.06
}

}  // namespace
}  // namespace kernellake
