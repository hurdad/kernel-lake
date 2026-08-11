#include "kernellake/execution_gpu/cudf_adapter.hpp"

#include <cmath>
#include <limits>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

double literal_as_double(const LiteralStorage& value) {
  if (std::holds_alternative<double>(value)) return std::get<double>(value);
  if (std::holds_alternative<std::int64_t>(value)) return static_cast<double>(std::get<std::int64_t>(value));
  return 0.0;
}

std::int64_t literal_as_int64(const LiteralStorage& value) {
  if (std::holds_alternative<std::int64_t>(value)) return std::get<std::int64_t>(value);
  if (std::holds_alternative<double>(value)) return static_cast<std::int64_t>(std::get<double>(value));
  return 0;
}

// Picks the narrowest cudf fixed_point representation that can hold `precision`
// decimal digits (the same width tiers Spark/Postgres-family engines use).
cudf::type_id decimal_cudf_type_id(std::int32_t precision) {
  if (precision <= 9) return cudf::type_id::DECIMAL32;
  if (precision <= 18) return cudf::type_id::DECIMAL64;
  return cudf::type_id::DECIMAL128;
}

const DataType& require_decimal_precision_scale(const DataType& type) {
  if (!type.precision.has_value() || !type.scale.has_value()) {
    throw PlanningError("DECIMAL type is missing precision/scale");
  }
  return type;
}

}  // namespace

cudf::type_id to_cudf_type_id(TypeId id) {
  switch (id) {
    case TypeId::Boolean:
      return cudf::type_id::BOOL8;
    case TypeId::Int32:
      return cudf::type_id::INT32;
    case TypeId::Int64:
      return cudf::type_id::INT64;
    case TypeId::UInt32:
      return cudf::type_id::UINT32;
    case TypeId::UInt64:
      return cudf::type_id::UINT64;
    case TypeId::Float32:
      return cudf::type_id::FLOAT32;
    case TypeId::Float64:
      return cudf::type_id::FLOAT64;
    case TypeId::String:
      return cudf::type_id::STRING;
    case TypeId::Date32:
      return cudf::type_id::TIMESTAMP_DAYS;
    case TypeId::Timestamp:
      return cudf::type_id::TIMESTAMP_MICROSECONDS;
    case TypeId::Decimal:
      throw PlanningError("DECIMAL is not yet supported for GPU execution");
  }
  throw PlanningError("unreachable: unknown KernelLake TypeId");
}

cudf::data_type to_cudf_type(const DataType& type) {
  if (type.id == TypeId::Decimal) {
    require_decimal_precision_scale(type);
    return cudf::data_type{decimal_cudf_type_id(*type.precision), -*type.scale};
  }
  return cudf::data_type{to_cudf_type_id(type.id)};
}

DecimalRawValue decimal_raw_value(const DataType& type, const LiteralStorage& value) {
  require_decimal_precision_scale(type);
  const std::int32_t cudf_scale = -*type.scale;
  const __int128_t raw =
      static_cast<__int128_t>(std::llround(literal_as_double(value) * std::pow(10.0, -cudf_scale)));
  return DecimalRawValue{raw, cudf_scale, decimal_cudf_type_id(*type.precision)};
}

std::unique_ptr<cudf::scalar> make_decimal_scalar(const DataType& type, const LiteralStorage& value,
                                                  bool is_valid) {
  const DecimalRawValue raw_value = decimal_raw_value(type, value);
  const numeric::scale_type scale{raw_value.cudf_scale};
  // binder.cpp's cast_if_needed() already rejects an out-of-range literal
  // at bind time (the normal path here), but this is the last point before
  // an unchecked narrowing static_cast would otherwise silently wrap a
  // too-large raw value instead of failing -- kept as defense-in-depth for
  // any DECIMAL literal construction path that bypasses that check.
  switch (raw_value.type_id) {
    case cudf::type_id::DECIMAL32:
      if (raw_value.raw < std::numeric_limits<std::int32_t>::min() ||
          raw_value.raw > std::numeric_limits<std::int32_t>::max()) {
        throw PlanningError("DECIMAL literal value out of range for its declared precision (internal error)");
      }
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
          static_cast<std::int32_t>(raw_value.raw), scale, is_valid);
    case cudf::type_id::DECIMAL64:
      if (raw_value.raw < std::numeric_limits<std::int64_t>::min() ||
          raw_value.raw > std::numeric_limits<std::int64_t>::max()) {
        throw PlanningError("DECIMAL literal value out of range for its declared precision (internal error)");
      }
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
          static_cast<std::int64_t>(raw_value.raw), scale, is_valid);
    default:
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(raw_value.raw, scale, is_valid);
  }
}

std::unique_ptr<cudf::scalar> literal_to_scalar(const LiteralExpression& expr) {
  const bool is_valid = !expr.is_null();
  const LiteralStorage& value = expr.value();
  switch (expr.result_type().id) {
    case TypeId::Boolean:
      return std::make_unique<cudf::numeric_scalar<bool>>(
          std::holds_alternative<bool>(value) && std::get<bool>(value), is_valid);
    case TypeId::Int32:
      return std::make_unique<cudf::numeric_scalar<std::int32_t>>(
          static_cast<std::int32_t>(literal_as_int64(value)), is_valid);
    case TypeId::Int64:
      return std::make_unique<cudf::numeric_scalar<std::int64_t>>(literal_as_int64(value), is_valid);
    case TypeId::UInt32:
      return std::make_unique<cudf::numeric_scalar<std::uint32_t>>(
          static_cast<std::uint32_t>(literal_as_int64(value)), is_valid);
    case TypeId::UInt64:
      return std::make_unique<cudf::numeric_scalar<std::uint64_t>>(
          static_cast<std::uint64_t>(literal_as_int64(value)), is_valid);
    case TypeId::Float32:
      return std::make_unique<cudf::numeric_scalar<float>>(static_cast<float>(literal_as_double(value)),
                                                           is_valid);
    case TypeId::Float64:
      return std::make_unique<cudf::numeric_scalar<double>>(literal_as_double(value), is_valid);
    case TypeId::String:
      return std::make_unique<cudf::string_scalar>(
          std::holds_alternative<std::string>(value) ? std::get<std::string>(value) : "", is_valid);
    case TypeId::Date32:
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
          cudf::duration_D{static_cast<std::int32_t>(literal_as_int64(value))}, is_valid);
    case TypeId::Timestamp:
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
          cudf::duration_us{literal_as_int64(value)}, is_valid);
    case TypeId::Decimal:
      return make_decimal_scalar(expr.result_type(), value, is_valid);
  }
  throw PlanningError("unreachable: unknown KernelLake TypeId");
}

}  // namespace kernellake
