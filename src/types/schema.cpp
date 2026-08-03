#include "kernellake/types/schema.hpp"

#include <algorithm>

namespace kernellake {

std::string_view to_string(TypeId id) noexcept {
  switch (id) {
    case TypeId::Boolean:
      return "BOOLEAN";
    case TypeId::Int32:
      return "INT32";
    case TypeId::Int64:
      return "INT64";
    case TypeId::UInt32:
      return "UINT32";
    case TypeId::UInt64:
      return "UINT64";
    case TypeId::Float32:
      return "FLOAT32";
    case TypeId::Float64:
      return "FLOAT64";
    case TypeId::String:
      return "STRING";
    case TypeId::Date32:
      return "DATE32";
    case TypeId::Timestamp:
      return "TIMESTAMP";
    case TypeId::Decimal:
      return "DECIMAL";
  }
  return "UNKNOWN";
}

std::string DataType::to_string() const {
  std::string result(kernellake::to_string(id));
  if (id == TypeId::Decimal) {
    result += "(" + std::to_string(precision.value_or(0)) + "," + std::to_string(scale.value_or(0)) + ")";
  }
  if (!nullable) {
    result += " NOT NULL";
  }
  return result;
}

DataType boolean_type(bool nullable) {
  return DataType{TypeId::Boolean, nullable, {}, {}};
}
DataType int32_type(bool nullable) {
  return DataType{TypeId::Int32, nullable, {}, {}};
}
DataType int64_type(bool nullable) {
  return DataType{TypeId::Int64, nullable, {}, {}};
}
DataType uint32_type(bool nullable) {
  return DataType{TypeId::UInt32, nullable, {}, {}};
}
DataType uint64_type(bool nullable) {
  return DataType{TypeId::UInt64, nullable, {}, {}};
}
DataType float32_type(bool nullable) {
  return DataType{TypeId::Float32, nullable, {}, {}};
}
DataType float64_type(bool nullable) {
  return DataType{TypeId::Float64, nullable, {}, {}};
}
DataType string_type(bool nullable) {
  return DataType{TypeId::String, nullable, {}, {}};
}
DataType date32_type(bool nullable) {
  return DataType{TypeId::Date32, nullable, {}, {}};
}
DataType timestamp_type(bool nullable) {
  return DataType{TypeId::Timestamp, nullable, {}, {}};
}

DataType decimal_type(std::int32_t precision, std::int32_t scale, bool nullable) {
  return DataType{TypeId::Decimal, nullable, precision, scale};
}

Schema::Schema(std::vector<Field> fields) : fields_(std::move(fields)) {}

std::optional<std::size_t> Schema::find_field(std::string_view name) const {
  const auto it =
      std::find_if(fields_.begin(), fields_.end(), [&](const Field& f) { return f.name == name; });
  if (it == fields_.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(std::distance(fields_.begin(), it));
}

bool Schema::equals(const Schema& other) const noexcept {
  return fields_ == other.fields_;
}

}  // namespace kernellake
