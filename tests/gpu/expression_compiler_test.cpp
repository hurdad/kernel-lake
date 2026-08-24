#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <rmm/device_buffer.hpp>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/execution_gpu/expression_compiler.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

// Minimal real ExecutionContext for ExpressionCompiler::compile()'s
// stream/mr threading -- these tests evaluate against real GPU columns but
// don't need a full operator tree, just a valid stream/memory_resource
// pair. See cudf_adapter_test.cpp's identical helper for the same
// rationale.
struct TestContext {
  EngineConfig config = default_config();
  RmmEnvironment env{config};
  CudaStream stream;
  QueryMemoryTracker tracker = env.make_query_tracker();
  ExecutionContext context{"test-query", 0, stream.get(), tracker.resource_ref(), nullptr, nullptr, &tracker};
};

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

  TestContext ctx;
  ExpressionCompiler compiler;
  const cudf::ast::expression& compiled = compiler.compile(greater, ctx.context);
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
  auto quantity_cmp = std::make_shared<BinaryExpression>(BinaryOperator::Less, quantity_as_int64, twenty_four,
                                                         boolean_type(false));

  BinaryExpression predicate(BinaryOperator::And, between, quantity_cmp, boolean_type(false));

  TestContext ctx;
  ExpressionCompiler filter_compiler;
  std::unique_ptr<cudf::column> mask =
      cudf::compute_column(table.view(), filter_compiler.compile(predicate, ctx.context));
  EXPECT_EQ(mask->type().id(), cudf::type_id::BOOL8);
  EXPECT_EQ(mask->size(), 10);

  auto extendedprice = std::make_shared<ColumnExpression>("l_extendedprice", 0, float64_type(false));
  BinaryExpression revenue_expr(BinaryOperator::Multiply, extendedprice, discount, float64_type(false));

  ExpressionCompiler projection_compiler;
  std::unique_ptr<cudf::column> revenue =
      cudf::compute_column(table.view(), projection_compiler.compile(revenue_expr, ctx.context));
  ASSERT_EQ(revenue->type().id(), cudf::type_id::FLOAT64);

  cudaDeviceSynchronize();
  for (double value : copy_to_host<double>(revenue->view())) EXPECT_DOUBLE_EQ(value, 6.0);  // 100.0*0.06
}

TEST(ExpressionCompiler, CastingNegativeInt64ToUInt64SilentlyWrapsAround) {
  // Characterization test: pins down the exact cudf::ast behavior that
  // motivates binder.cpp's promote_numeric() rejecting a signed/unsigned
  // integer mix instead of promoting to UInt64 (see
  // Binder.MixingSignedAndUnsignedIntegerTypesInComparisonIsRejected).
  // Without that bind-time rejection, `WHERE signed_col < unsigned_col`
  // would silently misevaluate for any negative signed_col: CAST_TO_UINT64
  // two's-complement-wraps rather than erroring or saturating. If a future
  // cudf upgrade ever changes this, this test fails first and the
  // corresponding binder rejection can be revisited.
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64}, 1);
  auto scalar = cudf::make_fixed_width_scalar<int64_t>(-5);
  cudf::mutable_column_view mutable_view = column->mutable_view();
  cudf::fill_in_place(mutable_view, 0, 1, *scalar);
  columns.push_back(std::move(column));
  cudf::table table(std::move(columns));

  auto col = std::make_shared<ColumnExpression>("signed_col", 0, int64_type(false));
  auto cast_expr = std::make_shared<CastExpression>(col, uint64_type(false));

  TestContext ctx;
  ExpressionCompiler compiler;
  std::unique_ptr<cudf::column> result =
      cudf::compute_column(table.view(), compiler.compile(*cast_expr, ctx.context));
  ASSERT_EQ(result->type().id(), cudf::type_id::UINT64);

  cudaDeviceSynchronize();
  const std::vector<std::uint64_t> host = copy_to_host<std::uint64_t>(result->view());
  EXPECT_EQ(host[0], 18446744073709551611ULL);  // 2^64 - 5, not an error and not saturated to 0.
}

// make_literal()'s switch has one case per KernelLake TypeId -- every
// existing test above only ever constructs Int64/Float64 literals (via
// LiteralExpression::make_int64/make_float64), leaving every other
// concrete numeric/temporal/decimal case unexercised. LiteralExpression's
// own two-argument constructor (value, type) is the general-purpose API
// the make_* factories are just convenience wrappers over, so building one
// directly with an explicit non-default type is the natural way to reach
// these, not a contrived workaround.
TEST(ExpressionCompiler, BooleanLiteralCompilesToCorrectValue) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 5));
  cudf::table table(std::move(columns));

  LiteralExpression true_literal(true, boolean_type(false));
  TestContext ctx;
  ExpressionCompiler compiler;
  std::unique_ptr<cudf::column> result =
      cudf::compute_column(table.view(), compiler.compile(true_literal, ctx.context));
  ASSERT_EQ(result->type().id(), cudf::type_id::BOOL8);

  cudaDeviceSynchronize();
  for (unsigned char value : copy_to_host<unsigned char>(result->view())) EXPECT_TRUE(value != 0);
}

TEST(ExpressionCompiler, Int32TypedLiteralCompilesToCorrectValue) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 5));
  cudf::table table(std::move(columns));

  LiteralExpression literal(std::int64_t{7}, int32_type(false));
  TestContext ctx;
  ExpressionCompiler compiler;
  std::unique_ptr<cudf::column> result =
      cudf::compute_column(table.view(), compiler.compile(literal, ctx.context));
  ASSERT_EQ(result->type().id(), cudf::type_id::INT32);

  cudaDeviceSynchronize();
  for (std::int32_t value : copy_to_host<std::int32_t>(result->view())) EXPECT_EQ(value, 7);
}

TEST(ExpressionCompiler, UInt64TypedLiteralCompilesToCorrectValue) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 5));
  cudf::table table(std::move(columns));

  LiteralExpression literal(std::int64_t{9}, uint64_type(false));
  TestContext ctx;
  ExpressionCompiler compiler;
  std::unique_ptr<cudf::column> result =
      cudf::compute_column(table.view(), compiler.compile(literal, ctx.context));
  ASSERT_EQ(result->type().id(), cudf::type_id::UINT64);

  cudaDeviceSynchronize();
  for (std::uint64_t value : copy_to_host<std::uint64_t>(result->view())) EXPECT_EQ(value, 9u);
}

TEST(ExpressionCompiler, Float32TypedLiteralCompilesToCorrectValue) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 5));
  cudf::table table(std::move(columns));

  LiteralExpression literal(2.5, float32_type(false));
  TestContext ctx;
  ExpressionCompiler compiler;
  std::unique_ptr<cudf::column> result =
      cudf::compute_column(table.view(), compiler.compile(literal, ctx.context));
  ASSERT_EQ(result->type().id(), cudf::type_id::FLOAT32);

  cudaDeviceSynchronize();
  for (float value : copy_to_host<float>(result->view())) EXPECT_FLOAT_EQ(value, 2.5F);
}

TEST(ExpressionCompiler, TimestampLiteralCompilesToCorrectValue) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 5));
  cudf::table table(std::move(columns));

  const LiteralExpression literal = LiteralExpression::make_timestamp(1'700'000'000'000'000LL);
  TestContext ctx;
  ExpressionCompiler compiler;
  std::unique_ptr<cudf::column> result =
      cudf::compute_column(table.view(), compiler.compile(literal, ctx.context));
  ASSERT_EQ(result->type().id(), cudf::type_id::TIMESTAMP_MICROSECONDS);

  cudaDeviceSynchronize();
  for (std::int64_t value : copy_to_host<std::int64_t>(result->view()))
    EXPECT_EQ(value, 1'700'000'000'000'000LL);
}

// decimal_raw_value() (cudf_adapter.cpp) picks cudf's DECIMAL32/64/128
// storage width from the DataType's own precision (<=9, <=18, >18) --
// exercising all three needs a decimal literal at each precision tier, not
// just one.
TEST(ExpressionCompiler, DecimalLiteralsCompileAtEachCudfStorageWidth) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 1));
  cudf::table table(std::move(columns));
  TestContext ctx;

  {
    LiteralExpression literal(12.34, decimal_type(/*precision=*/5, /*scale=*/2, false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(literal, ctx.context));
    EXPECT_EQ(result->type().id(), cudf::type_id::DECIMAL32);
  }
  {
    LiteralExpression literal(12.34, decimal_type(/*precision=*/15, /*scale=*/2, false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(literal, ctx.context));
    EXPECT_EQ(result->type().id(), cudf::type_id::DECIMAL64);
  }
  {
    LiteralExpression literal(12.34, decimal_type(/*precision=*/30, /*scale=*/2, false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(literal, ctx.context));
    EXPECT_EQ(result->type().id(), cudf::type_id::DECIMAL128);
  }
}

// compile()'s dedicated AggregateExpression check (as opposed to falling
// through to the generic "unrecognized expression type" case) exists so a
// misplaced aggregate produces a specific, actionable error message --
// only reachable by calling the row-wise compiler directly on an
// AggregateExpression, since the real planner never routes one here.
TEST(ExpressionCompiler, CompilingAggregateExpressionThrowsSpecificError) {
  auto count_star =
      std::make_shared<AggregateExpression>(AggregateFunction::CountStar, nullptr, int64_type(false));
  TestContext ctx;
  ExpressionCompiler compiler;
  EXPECT_THROW({ (void)compiler.compile(*count_star, ctx.context); }, ExecutionError);
}

// to_ast_operator()/compile_binary(): every existing test above only
// exercises Add/Multiply/Greater/Less/And -- the other half of
// BinaryOperator's enumerators (used by real queries just as often, e.g.
// `<>`/`<=`/`OR`) had no coverage of their own mapping to cudf::ast.
TEST(ExpressionCompiler, RemainingBinaryOperatorsMapToCorrectCudfOperator) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(6, 4));
  cudf::table table(std::move(columns));
  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto a_as_int64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto three = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(3));
  TestContext ctx;

  for (const BinaryOperator op :
       {BinaryOperator::Divide, BinaryOperator::NotEqual, BinaryOperator::LessEqual, BinaryOperator::Or}) {
    BinaryExpression expr(op, a_as_int64, three, boolean_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(expr, ctx.context));
    EXPECT_EQ(result->size(), 4) << "operator " << static_cast<int>(op);
  }
}

// compile_unary(): Negate is the only UnaryOperator any existing test
// exercises. Not/IsNull/IsNotNull are the other three real cases the real
// binder produces (`NOT x`, `x IS NULL`, `x IS NOT NULL`).
TEST(ExpressionCompiler, NotIsNullIsNotNullUnaryOperatorsProduceCorrectResults) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 3));
  cudf::table table(std::move(columns));
  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto a_as_int64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto three = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(3));
  auto is_three =
      std::make_shared<BinaryExpression>(BinaryOperator::Equal, a_as_int64, three, boolean_type(false));
  TestContext ctx;

  {
    UnaryExpression not_expr(UnaryOperator::Not, is_three, boolean_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(not_expr, ctx.context));
    ASSERT_EQ(result->type().id(), cudf::type_id::BOOL8);
    cudaDeviceSynchronize();
    // a == 0 for every row, three-literal is 3 -> `a == 3` is false ->
    // `NOT (a == 3)` is true for every row.
    for (unsigned char value : copy_to_host<unsigned char>(result->view())) EXPECT_TRUE(value != 0);
  }
  {
    UnaryExpression is_null_expr(UnaryOperator::IsNull, a_as_int64, boolean_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(is_null_expr, ctx.context));
    ASSERT_EQ(result->type().id(), cudf::type_id::BOOL8);
    cudaDeviceSynchronize();
    // Column has no nulls (filled_int32_column has no null mask).
    for (unsigned char value : copy_to_host<unsigned char>(result->view())) EXPECT_TRUE(value == 0);
  }
  {
    UnaryExpression is_not_null_expr(UnaryOperator::IsNotNull, a_as_int64, boolean_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(is_not_null_expr, ctx.context));
    ASSERT_EQ(result->type().id(), cudf::type_id::BOOL8);
    cudaDeviceSynchronize();
    for (unsigned char value : copy_to_host<unsigned char>(result->view())) EXPECT_TRUE(value != 0);
  }
}

// to_ast_operator() must map SQL AND/OR to cudf::ast's NULL_LOGICAL_AND/
// NULL_LOGICAL_OR, not plain LOGICAL_AND/LOGICAL_OR -- the plain variants
// propagate NULL whenever *either* operand is null, with no special-casing
// of a definitively-FALSE/TRUE operand, whereas SQL's three-valued (Kleene)
// logic requires `TRUE OR NULL = TRUE` and `FALSE AND NULL = FALSE`. This
// regressed for real: `WHERE x IS NULL OR x = 3` silently dropped every
// NULL row (IS NULL evaluates to TRUE, but LOGICAL_OR(TRUE, NULL) came out
// NULL instead of TRUE, since `x = 3` is itself NULL when x is NULL) --
// caught by a LEFT OUTER JOIN test since that was the first path to
// produce a genuinely nullable column feeding a WHERE clause, but the bug
// itself has nothing to do with joins, hence this direct, join-free
// regression test.
TEST(ExpressionCompiler, LogicalOrAndAndUseKleeneNullSemantics) {
  // 4 rows: a = [0, NULL, 2, NULL] (rows 1 and 3 null).
  auto column =
      cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64}, 4, cudf::mask_state::ALL_VALID);
  {
    cudf::mutable_column_view view = column->mutable_view();
    const std::vector<std::int64_t> host_values = {0, 0, 2, 0};
    cudaMemcpy(view.data<std::int64_t>(), host_values.data(), host_values.size() * sizeof(std::int64_t),
               cudaMemcpyHostToDevice);
  }
  const cudf::bitmask_type valid_mask_word = 0b0101;  // rows 0,2 valid; rows 1,3 null.
  rmm::device_buffer null_mask(cudf::bitmask_allocation_size_bytes(4), rmm::cuda_stream_default);
  cudaMemsetAsync(null_mask.data(), 0, null_mask.size(), rmm::cuda_stream_default.value());
  cudaMemcpyAsync(null_mask.data(), &valid_mask_word, sizeof(valid_mask_word), cudaMemcpyHostToDevice,
                  rmm::cuda_stream_default.value());
  rmm::cuda_stream_default.synchronize();
  column->set_null_mask(std::move(null_mask), /*new_null_count=*/2);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  cudf::table table(std::move(columns));

  auto a = std::make_shared<ColumnExpression>("a", 0, int64_type(true));
  auto is_null = std::make_shared<UnaryExpression>(UnaryOperator::IsNull, a, boolean_type(false));
  auto two = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(2));
  auto equals_two = std::make_shared<BinaryExpression>(BinaryOperator::Equal, a, two, boolean_type(false));
  TestContext ctx;

  {
    // `a IS NULL OR a = 2`: row 0 (0==2 false, not null) -> FALSE; row 1
    // (null, IS NULL true) -> TRUE via NULL_LOGICAL_OR's null-with-true
    // case; row 2 (2==2 true) -> TRUE; row 3 (null) -> TRUE. All 4 rows
    // must be non-null booleans -- none should come out NULL.
    BinaryExpression or_expr(BinaryOperator::Or, is_null, equals_two, boolean_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(or_expr, ctx.context));
    ASSERT_EQ(result->null_count(), 0);
    cudaDeviceSynchronize();
    const std::vector<unsigned char> values = copy_to_host<unsigned char>(result->view());
    EXPECT_EQ(values[0], 0);
    EXPECT_NE(values[1], 0);
    EXPECT_NE(values[2], 0);
    EXPECT_NE(values[3], 0);
  }
  {
    // `a IS NOT NULL AND a = 2`: row 0 (not null, 0==2 false) -> FALSE;
    // row 1 (null -> IS NOT NULL false) -> FALSE via NULL_LOGICAL_AND's
    // null-with-false case; row 2 (not null, 2==2 true) -> TRUE; row 3
    // (null) -> FALSE. Again none should come out NULL.
    auto is_not_null = std::make_shared<UnaryExpression>(UnaryOperator::IsNotNull, a, boolean_type(false));
    BinaryExpression and_expr(BinaryOperator::And, is_not_null, equals_two, boolean_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(and_expr, ctx.context));
    ASSERT_EQ(result->null_count(), 0);
    cudaDeviceSynchronize();
    const std::vector<unsigned char> values = copy_to_host<unsigned char>(result->view());
    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[1], 0);
    EXPECT_NE(values[2], 0);
    EXPECT_EQ(values[3], 0);
  }
}

// compile_cast()'s default case: cudf::ast only supports widening casts to
// INT64/UINT64/FLOAT64 (its own comment) -- casting to anything else (e.g.
// back down to a narrower/non-numeric type) must fail with a clear error
// rather than silently doing the wrong thing.
TEST(ExpressionCompiler, CastToUnsupportedTargetTypeThrows) {
  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto cast_to_bool = std::make_shared<CastExpression>(a, boolean_type(false));
  TestContext ctx;
  ExpressionCompiler compiler;
  EXPECT_THROW({ (void)compiler.compile(*cast_to_bool, ctx.context); }, ExecutionError);
}

TEST(ExpressionCompiler, UnaryNegateOnUnsignedColumnSilentlyWrapsAround) {
  // Companion characterization test for
  // Binder.UnaryNegateOnUnsignedColumnIsRejected: expression_compiler.cpp's
  // Negate case synthesizes `0 - x` in x's own type (cudf::ast has no
  // dedicated negation operator); for unsigned x that wraps instead of
  // producing a negative value or erroring. If a future cudf upgrade ever
  // changes this, this test fails first and the corresponding binder
  // rejection can be revisited.
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32}, 1);
  auto scalar = cudf::make_fixed_width_scalar<std::uint32_t>(5);
  cudf::mutable_column_view mutable_view = column->mutable_view();
  cudf::fill_in_place(mutable_view, 0, 1, *scalar);
  columns.push_back(std::move(column));
  cudf::table table(std::move(columns));

  auto col = std::make_shared<ColumnExpression>("u", 0, uint32_type(false));
  auto negate_expr = std::make_shared<UnaryExpression>(UnaryOperator::Negate, col, uint32_type(false));

  TestContext ctx;
  ExpressionCompiler compiler;
  std::unique_ptr<cudf::column> result =
      cudf::compute_column(table.view(), compiler.compile(*negate_expr, ctx.context));
  ASSERT_EQ(result->type().id(), cudf::type_id::UINT32);

  cudaDeviceSynchronize();
  const std::vector<std::uint32_t> host = copy_to_host<std::uint32_t>(result->view());
  EXPECT_EQ(host[0], 4294967291U);  // 2^32 - 5, not an error and not saturated to 0.
}

// Characterization tests for as_double()/as_int64()'s bool/fallback
// branches -- LiteralExpression's own two-argument constructor lets a
// test build a value/type combination the real binder would never
// produce (e.g. a bool-holding LiteralStorage declared Float64), the
// same pattern CastingNegativeInt64ToUInt64SilentlyWrapsAround above
// already uses to pin down defined behavior for a case the binder
// itself prevents. Locks in that a bool coerces to 1.0/0.0 (or 1/0) and
// anything else (a string here) falls back to 0.0/0, rather than being
// silently wrong or crashing, should some future caller ever bypass the
// binder's own type-matching.
TEST(ExpressionCompiler, AsDoubleCoercesBoolAndFallsBackToZeroForOtherVariants) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 1));
  cudf::table table(std::move(columns));
  TestContext ctx;

  {
    const LiteralExpression bool_as_float64(true, float64_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(bool_as_float64, ctx.context));
    cudaDeviceSynchronize();
    EXPECT_DOUBLE_EQ(copy_to_host<double>(result->view())[0], 1.0);
  }
  {
    const LiteralExpression string_as_float64(std::string("x"), float64_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(string_as_float64, ctx.context));
    cudaDeviceSynchronize();
    EXPECT_DOUBLE_EQ(copy_to_host<double>(result->view())[0], 0.0);
  }
}

TEST(ExpressionCompiler, AsInt64CoercesDoubleAndBoolAndFallsBackToZeroForOtherVariants) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 1));
  cudf::table table(std::move(columns));
  TestContext ctx;

  {
    const LiteralExpression double_as_int32(3.7, int32_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(double_as_int32, ctx.context));
    cudaDeviceSynchronize();
    EXPECT_EQ(copy_to_host<std::int32_t>(result->view())[0], 3);
  }
  {
    const LiteralExpression bool_as_int32(true, int32_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(bool_as_int32, ctx.context));
    cudaDeviceSynchronize();
    EXPECT_EQ(copy_to_host<std::int32_t>(result->view())[0], 1);
  }
  {
    const LiteralExpression string_as_int32(std::string("x"), int32_type(false));
    ExpressionCompiler compiler;
    std::unique_ptr<cudf::column> result =
        cudf::compute_column(table.view(), compiler.compile(string_as_int32, ctx.context));
    cudaDeviceSynchronize();
    EXPECT_EQ(copy_to_host<std::int32_t>(result->view())[0], 0);
  }
}

// Test-only Expression subclass -- no real Expression kind reaches
// compile()'s final fallback throw, since every kind the real parser/
// binder can produce has its own dynamic_cast check above it. A type
// outside that closed set is the only way to reach it at all.
class UnknownExpression final : public Expression {
 public:
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override { return "UnknownExpression"; }
  [[nodiscard]] std::string structural_key() const override { return "UnknownExpression"; }

 private:
  DataType type_ = int64_type(false);
};

TEST(ExpressionCompiler, CompilingUnrecognizedExpressionTypeThrows) {
  const UnknownExpression unknown;
  TestContext ctx;
  ExpressionCompiler compiler;
  EXPECT_THROW({ (void)compiler.compile(unknown, ctx.context); }, ExecutionError);
}

// BinaryOperator/UnaryOperator both have a fixed uint8_t underlying type
// (expression.hpp), so casting an out-of-range value to either is well-
// defined (not UB) -- just a value with no matching enumerator, exactly
// what to_ast_operator()'s/compile_unary()'s own final fallback throws
// exist to guard against for a corrupted or future-added-but-unhandled
// operator value.
TEST(ExpressionCompiler, CompilingBinaryExpressionWithUnknownOperatorThrows) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 1));
  cudf::table table(std::move(columns));

  auto a = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  auto b = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(2));
  BinaryExpression bogus(static_cast<BinaryOperator>(255), a, b, int64_type(false));

  TestContext ctx;
  ExpressionCompiler compiler;
  EXPECT_THROW({ (void)compiler.compile(bogus, ctx.context); }, ExecutionError);
}

TEST(ExpressionCompiler, CompilingUnaryExpressionWithUnknownOperatorThrows) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(filled_int32_column(0, 1));
  cudf::table table(std::move(columns));

  auto a = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(1));
  UnaryExpression bogus(static_cast<UnaryOperator>(255), a, int64_type(false));

  TestContext ctx;
  ExpressionCompiler compiler;
  EXPECT_THROW({ (void)compiler.compile(bogus, ctx.context); }, ExecutionError);
}

}  // namespace
}  // namespace kernellake
