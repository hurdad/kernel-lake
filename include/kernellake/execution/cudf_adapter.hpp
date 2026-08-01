#pragma once

#include <cudf/types.hpp>

#include "kernellake/types/schema.hpp"

namespace kernellake {

// Throws PlanningError for TypeId::Decimal (not yet supported for GPU
// execution -- cudf's fixed_point types need precision/scale plumbing this
// MVP doesn't do yet).
[[nodiscard]] cudf::type_id to_cudf_type_id(TypeId id);

[[nodiscard]] cudf::data_type to_cudf_type(const DataType& type);

}  // namespace kernellake
