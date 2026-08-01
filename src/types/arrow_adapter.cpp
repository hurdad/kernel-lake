#include "kernellake/types/arrow_adapter.hpp"

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {
constexpr arrow::TimeUnit::type kTimestampUnit = arrow::TimeUnit::MICRO;
}  // namespace

DataType from_arrow_type(const std::shared_ptr<arrow::DataType>& type, bool nullable) {
  switch (type->id()) {
    case arrow::Type::BOOL:
      return boolean_type(nullable);
    case arrow::Type::INT32:
      return int32_type(nullable);
    case arrow::Type::INT64:
      return int64_type(nullable);
    case arrow::Type::UINT32:
      return uint32_type(nullable);
    case arrow::Type::UINT64:
      return uint64_type(nullable);
    case arrow::Type::FLOAT:
      return float32_type(nullable);
    case arrow::Type::DOUBLE:
      return float64_type(nullable);
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
      return string_type(nullable);
    case arrow::Type::DATE32:
      return date32_type(nullable);
    case arrow::Type::TIMESTAMP:
      return timestamp_type(nullable);
    case arrow::Type::DECIMAL128: {
      const auto& decimal = static_cast<const arrow::Decimal128Type&>(*type);
      return decimal_type(decimal.precision(), decimal.scale(), nullable);
    }
    default:
      throw PlanningError("unsupported Arrow type '" + type->ToString() +
                           "': KernelLake's type system does not yet cover this type");
  }
}

std::shared_ptr<arrow::DataType> to_arrow_type(const DataType& type) {
  switch (type.id) {
    case TypeId::Boolean:
      return arrow::boolean();
    case TypeId::Int32:
      return arrow::int32();
    case TypeId::Int64:
      return arrow::int64();
    case TypeId::UInt32:
      return arrow::uint32();
    case TypeId::UInt64:
      return arrow::uint64();
    case TypeId::Float32:
      return arrow::float32();
    case TypeId::Float64:
      return arrow::float64();
    case TypeId::String:
      return arrow::utf8();
    case TypeId::Date32:
      return arrow::date32();
    case TypeId::Timestamp:
      return arrow::timestamp(kTimestampUnit);
    case TypeId::Decimal:
      return arrow::decimal128(type.precision.value_or(38), type.scale.value_or(0));
  }
  throw PlanningError("unreachable: unknown KernelLake TypeId");
}

Schema from_arrow_schema(const std::shared_ptr<arrow::Schema>& schema) {
  std::vector<Field> fields;
  fields.reserve(static_cast<std::size_t>(schema->num_fields()));
  for (const auto& arrow_field : schema->fields()) {
    fields.push_back(Field{arrow_field->name(), from_arrow_type(arrow_field->type(),
                                                                  arrow_field->nullable())});
  }
  return Schema(std::move(fields));
}

std::shared_ptr<arrow::Schema> to_arrow_schema(const Schema& schema) {
  std::vector<std::shared_ptr<arrow::Field>> arrow_fields;
  arrow_fields.reserve(schema.field_count());
  for (const Field& field : schema.fields()) {
    arrow_fields.push_back(arrow::field(field.name, to_arrow_type(field.type), field.type.nullable));
  }
  return arrow::schema(arrow_fields);
}

}  // namespace kernellake
