#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <memory>

#include "kernellake/expression/expression.hpp"
#include "kernellake/types/schema.hpp"

namespace kernellake {

// Throws PlanningError for TypeId::Decimal (not yet supported for GPU
// execution -- cudf's fixed_point types need precision/scale plumbing this
// MVP doesn't do yet).
[[nodiscard]] cudf::type_id to_cudf_type_id(TypeId id);

[[nodiscard]] cudf::data_type to_cudf_type(const DataType& type);

// Converts a literal directly to a cudf::scalar, for operators that need to
// materialize a plain literal as a standalone column (via
// cudf::make_column_from_scalar) rather than through cudf::ast::compute_column
// -- cudf::ast can only produce a fixed-width *output* column, so a CASE
// branch whose result is a STRING literal (e.g. `THEN 'high'`) cannot go
// through the AST path at all, even though a string literal is perfectly
// valid as an intermediate AST node (e.g. one side of `region = 'A'`). See
// docs/ARCHITECTURE.md.
[[nodiscard]] std::unique_ptr<cudf::scalar> literal_to_scalar(const LiteralExpression& expr);

}  // namespace kernellake
