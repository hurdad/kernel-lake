#pragma once

#include <arrow/api.h>

#include <memory>

#include "kernellake/types/schema.hpp"

namespace kernellake {

// Explicit, one-directional adapters between the internal KernelLake type
// system and Apache Arrow. Kept in one place so type-conversion logic is not
// scattered across operators (per the KernelLake architecture spec).
//
// Timestamps are treated as microsecond-precision, UTC-naive, matching the
// most common Parquet TIMESTAMP encoding; sub-microsecond or timezone-aware
// timestamps are out of scope for the MVP type system.

// Throws PlanningError if the Arrow type has no KernelLake equivalent
// (e.g. nested/list/struct/map types, which are not yet supported).
[[nodiscard]] DataType from_arrow_type(const std::shared_ptr<arrow::DataType>& type, bool nullable);

[[nodiscard]] std::shared_ptr<arrow::DataType> to_arrow_type(const DataType& type);

[[nodiscard]] Schema from_arrow_schema(const std::shared_ptr<arrow::Schema>& schema);

[[nodiscard]] std::shared_ptr<arrow::Schema> to_arrow_schema(const Schema& schema);

}  // namespace kernellake
