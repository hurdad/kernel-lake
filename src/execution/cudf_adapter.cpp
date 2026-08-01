#include "kernellake/execution/cudf_adapter.hpp"

#include "kernellake/common/errors.hpp"

namespace kernellake {

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

}  // namespace kernellake
