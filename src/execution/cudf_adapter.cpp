#include "kernellake/execution/cudf_adapter.hpp"

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
  return cudf::data_type{to_cudf_type_id(type.id)};
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
      throw PlanningError("DECIMAL literals are not yet supported for GPU execution");
  }
  throw PlanningError("unreachable: unknown KernelLake TypeId");
}

}  // namespace kernellake
