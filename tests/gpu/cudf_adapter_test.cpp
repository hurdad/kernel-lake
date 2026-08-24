// Direct unit coverage for cudf_adapter.cpp's free functions -- these had
// no dedicated test file before (only incidental coverage from whatever
// literal/CAST/EXTRACT shapes happened to appear in operator-level tests),
// leaving large swaths of literal_to_scalar()'s type-dispatch switch,
// to_cudf_type_id()'s type-dispatch switch, and make_decimal_scalar()'s
// DECIMAL32/64/128 tiering entirely unexercised (measured via an ad-hoc
// gpu-coverage build: 37% line coverage on this file, the worst in
// src/execution_gpu/).
#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/execution_gpu/cudf_adapter.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

// Minimal real ExecutionContext for literal_to_scalar()'s stream/mr
// parameters -- these tests only exercise the type-dispatch switch, not
// query execution, so a single tracker/stream reused across every literal
// case in a TEST() is enough; no need for a full operator tree.
struct TestContext {
  EngineConfig config = default_config();
  RmmEnvironment env{config};
  CudaStream stream;
  QueryMemoryTracker tracker = env.make_query_tracker();
  ExecutionContext context{"test-query", 0, stream.get(), tracker.resource_ref(), nullptr, nullptr, &tracker};
};

TEST(CudfAdapter, ToCudfTypeIdMapsEveryNonDecimalType) {
  EXPECT_EQ(to_cudf_type_id(TypeId::Boolean), cudf::type_id::BOOL8);
  EXPECT_EQ(to_cudf_type_id(TypeId::Int32), cudf::type_id::INT32);
  EXPECT_EQ(to_cudf_type_id(TypeId::Int64), cudf::type_id::INT64);
  EXPECT_EQ(to_cudf_type_id(TypeId::UInt32), cudf::type_id::UINT32);
  EXPECT_EQ(to_cudf_type_id(TypeId::UInt64), cudf::type_id::UINT64);
  EXPECT_EQ(to_cudf_type_id(TypeId::Float32), cudf::type_id::FLOAT32);
  EXPECT_EQ(to_cudf_type_id(TypeId::Float64), cudf::type_id::FLOAT64);
  EXPECT_EQ(to_cudf_type_id(TypeId::String), cudf::type_id::STRING);
  EXPECT_EQ(to_cudf_type_id(TypeId::Date32), cudf::type_id::TIMESTAMP_DAYS);
  EXPECT_EQ(to_cudf_type_id(TypeId::Timestamp), cudf::type_id::TIMESTAMP_MICROSECONDS);
}

TEST(CudfAdapter, ToCudfTypeIdThrowsForDecimal) {
  // Picking DECIMAL32/64/128 needs precision, which a bare TypeId doesn't
  // carry -- see to_cudf_type_id()'s own doc comment; to_cudf_type(DataType)
  // is the Decimal-capable entry point instead (covered below).
  EXPECT_THROW((void)(to_cudf_type_id(TypeId::Decimal)), PlanningError);
}

TEST(CudfAdapter, ToCudfTypeIdThrowsForUnknownTypeId) {
  // TypeId has a fixed uint8_t underlying type (schema.hpp), so this cast
  // is well-defined (not UB) -- a value with no matching enumerator,
  // exactly what this function's own final fallback throw exists to guard
  // against, unreachable through any real TypeId value.
  EXPECT_THROW((void)(to_cudf_type_id(static_cast<TypeId>(255))), PlanningError);
}

TEST(CudfAdapter, ToCudfTypePicksNarrowestDecimalTierByPrecision) {
  const cudf::data_type narrow = to_cudf_type(decimal_type(5, 2));
  EXPECT_EQ(narrow.id(), cudf::type_id::DECIMAL32);
  EXPECT_EQ(narrow.scale(), -2);

  const cudf::data_type mid = to_cudf_type(decimal_type(15, 3));
  EXPECT_EQ(mid.id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(mid.scale(), -3);

  const cudf::data_type wide = to_cudf_type(decimal_type(30, 4));
  EXPECT_EQ(wide.id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(wide.scale(), -4);
}

TEST(CudfAdapter, ToCudfTypeDelegatesToTypeIdForNonDecimal) {
  EXPECT_EQ(to_cudf_type(int64_type(false)).id(), cudf::type_id::INT64);
  EXPECT_EQ(to_cudf_type(string_type(false)).id(), cudf::type_id::STRING);
}

TEST(CudfAdapter, LiteralToScalarHandlesEveryNonDecimalType) {
  TestContext ctx;
  {
    const LiteralExpression expr(true, boolean_type(false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_TRUE(scalar->is_valid());
    EXPECT_TRUE(static_cast<const cudf::numeric_scalar<bool>&>(*scalar).value());
  }
  {
    const LiteralExpression expr(static_cast<std::int64_t>(-7), int32_type(false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int32_t>&>(*scalar).value(), -7);
  }
  {
    const LiteralExpression expr = LiteralExpression::make_int64(1234567890123);
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int64_t>&>(*scalar).value(), 1234567890123);
  }
  {
    const LiteralExpression expr(static_cast<std::int64_t>(42), uint32_type(false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::uint32_t>&>(*scalar).value(), 42u);
  }
  {
    const LiteralExpression expr(static_cast<std::int64_t>(99), uint64_type(false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::uint64_t>&>(*scalar).value(), 99u);
  }
  {
    const LiteralExpression expr(1.5, float32_type(false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_FLOAT_EQ(static_cast<const cudf::numeric_scalar<float>&>(*scalar).value(), 1.5f);
  }
  {
    const LiteralExpression expr = LiteralExpression::make_float64(2.5);
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_DOUBLE_EQ(static_cast<const cudf::numeric_scalar<double>&>(*scalar).value(), 2.5);
  }
  {
    const LiteralExpression expr = LiteralExpression::make_string("hello");
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(static_cast<const cudf::string_scalar&>(*scalar).to_string(), "hello");
  }
  {
    // 19723 days since the Unix epoch = 2024-01-01.
    const LiteralExpression expr = LiteralExpression::make_date32(19723);
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    const auto& timestamp = static_cast<const cudf::timestamp_scalar<cudf::timestamp_D>&>(*scalar);
    EXPECT_EQ(timestamp.value().time_since_epoch().count(), 19723);
  }
  {
    const LiteralExpression expr = LiteralExpression::make_timestamp(1700000000000000);
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    const auto& timestamp = static_cast<const cudf::timestamp_scalar<cudf::timestamp_us>&>(*scalar);
    EXPECT_EQ(timestamp.value().time_since_epoch().count(), 1700000000000000);
  }
}

TEST(CudfAdapter, LiteralToScalarProducesInvalidScalarForNullLiteral) {
  TestContext ctx;
  const LiteralExpression expr = LiteralExpression::make_null(int64_type(true));
  const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
  EXPECT_FALSE(scalar->is_valid());
}

TEST(CudfAdapter, LiteralToScalarBuildsEachDecimalTier) {
  TestContext ctx;
  {
    // precision=5 -> DECIMAL32. 123.45 shifted by scale=2 -> raw 12345.
    const LiteralExpression expr(123.45, decimal_type(5, 2, false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(scalar->type().id(), cudf::type_id::DECIMAL32);
    const auto& decimal = static_cast<const cudf::fixed_point_scalar<numeric::decimal32>&>(*scalar);
    EXPECT_EQ(decimal.value(), 12345);
  }
  {
    // precision=15 -> DECIMAL64. 12345678901.23 shifted by scale=2.
    const LiteralExpression expr(12345678901.23, decimal_type(15, 2, false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(scalar->type().id(), cudf::type_id::DECIMAL64);
    const auto& decimal = static_cast<const cudf::fixed_point_scalar<numeric::decimal64>&>(*scalar);
    EXPECT_EQ(decimal.value(), 1234567890123);
  }
  {
    // precision=30 -> DECIMAL128 (the `default:` tier). 42.99 shifted by
    // scale=2 -- the tier is chosen purely by declared precision, not by
    // how large the actual literal value is, so a small value here still
    // exercises the DECIMAL128 branch specifically.
    const LiteralExpression expr(42.99, decimal_type(30, 2, false));
    const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
    EXPECT_EQ(scalar->type().id(), cudf::type_id::DECIMAL128);
    const auto& decimal = static_cast<const cudf::fixed_point_scalar<numeric::decimal128>&>(*scalar);
    EXPECT_EQ(decimal.value(), 4299);
  }
}

// literal_as_double() (this file's own internal helper feeding
// decimal_raw_value()) had no coverage for a decimal literal whose
// LiteralStorage holds an int64_t rather than a double -- every existing
// decimal test above constructs one directly from a double literal.
TEST(CudfAdapter, LiteralToScalarBuildsDecimalFromInt64ValuedLiteral) {
  TestContext ctx;
  const LiteralExpression expr(static_cast<std::int64_t>(1234), decimal_type(5, 2, false));
  const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
  EXPECT_EQ(scalar->type().id(), cudf::type_id::DECIMAL32);
  const auto& decimal = static_cast<const cudf::fixed_point_scalar<numeric::decimal32>&>(*scalar);
  EXPECT_EQ(decimal.value(), 123400);
}

// make_decimal_scalar()'s DECIMAL32 out-of-range guard is documented
// defense-in-depth for a literal construction path that bypasses
// binder.cpp's own cast_if_needed() range check (see that guard's own
// comment) -- exercising it directly here, the only way to reach it at
// all, since going through the real SQL binder can never produce this
// combination by construction.
TEST(CudfAdapter, LiteralToScalarThrowsWhenDecimal32ValueOutOfRange) {
  TestContext ctx;
  // precision=9 picks the DECIMAL32 tier, but 99,999,999,999 * 10^0 far
  // exceeds INT32_MAX (~2.1 billion) once raw-shifted by scale=0.
  const LiteralExpression expr(99'999'999'999.0, decimal_type(9, 0, false));
  EXPECT_THROW((void)(literal_to_scalar(expr, ctx.context)), PlanningError);
}

// literal_as_double()'s fallback (unlike expression_compiler.cpp's own
// as_double(), this file's version doesn't even check bool -- only
// double/int64) had no coverage -- a bool-holding LiteralStorage
// declared Decimal is a value/type combination the real binder never
// produces, constructible directly the same way
// LiteralToScalarThrowsWhenDecimal32ValueOutOfRange above already does.
TEST(CudfAdapter, LiteralToScalarDecimalFromBoolValuedLiteralFallsBackToZero) {
  TestContext ctx;
  const LiteralExpression expr(true, decimal_type(5, 2, false));
  const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(expr, ctx.context);
  EXPECT_EQ(scalar->type().id(), cudf::type_id::DECIMAL32);
  const auto& decimal = static_cast<const cudf::fixed_point_scalar<numeric::decimal32>&>(*scalar);
  EXPECT_EQ(decimal.value(), 0);
}

// TypeId/DatePart both have a fixed uint8_t underlying type (schema.hpp/
// expression.hpp), so casting an out-of-range value to either is well-
// defined (not UB) -- just a value with no matching enumerator, exactly
// what these two functions' own final fallback throws exist to guard
// against. Neither fallback is reachable through any real TypeId/DatePart
// value, since every one of those already has its own case above it.
TEST(CudfAdapter, ToCudfDatetimeComponentThrowsForUnknownDatePart) {
  EXPECT_THROW((void)(to_cudf_datetime_component(static_cast<DatePart>(255))), ExecutionError);
}

TEST(CudfAdapter, LiteralToScalarThrowsForUnknownTypeId) {
  TestContext ctx;
  const DataType bogus{static_cast<TypeId>(255), false, std::nullopt, std::nullopt};
  const LiteralExpression expr(static_cast<std::int64_t>(1), bogus);
  EXPECT_THROW((void)(literal_to_scalar(expr, ctx.context)), PlanningError);
}

TEST(CudfAdapter, LiteralToScalarThrowsForDecimalMissingPrecisionScale) {
  TestContext ctx;
  const DataType incomplete{TypeId::Decimal, false, std::nullopt, std::nullopt};
  const LiteralExpression expr(static_cast<std::int64_t>(1), incomplete);
  EXPECT_THROW((void)(literal_to_scalar(expr, ctx.context)), PlanningError);
}

TEST(CudfAdapter, ToCudfDatetimeComponentMapsAllParts) {
  EXPECT_EQ(to_cudf_datetime_component(DatePart::Year), cudf::datetime::datetime_component::YEAR);
  EXPECT_EQ(to_cudf_datetime_component(DatePart::Month), cudf::datetime::datetime_component::MONTH);
  EXPECT_EQ(to_cudf_datetime_component(DatePart::Day), cudf::datetime::datetime_component::DAY);
}

}  // namespace
}  // namespace kernellake
