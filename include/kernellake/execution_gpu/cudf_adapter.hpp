#pragma once

#include <cudf/datetime.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <memory>

#include "kernellake/execution_gpu/execution_context.hpp"
#include "kernellake/expression/expression.hpp"
#include "kernellake/types/schema.hpp"

namespace kernellake {

// Shared by every GPU operator that materializes an ExtractExpression
// directly (ProjectionOperator, HashAggregateOperator,
// ScalarAggregateOperator -- cudf::ast has no datetime-extraction operator,
// so none of them can go through cudf::ast::compute_column for this).
[[nodiscard]] cudf::datetime::datetime_component to_cudf_datetime_component(DatePart part);

// Throws PlanningError for TypeId::Decimal -- picking DECIMAL32/64/128
// needs the type's precision, which a bare TypeId doesn't carry; use
// to_cudf_type(DataType) for Decimal instead.
[[nodiscard]] cudf::type_id to_cudf_type_id(TypeId id);

// For TypeId::Decimal, picks DECIMAL32 (precision <= 9), DECIMAL64 (<= 18),
// or DECIMAL128 (<= 38), with cudf's scale set to the *negative* of
// DataType::scale -- cudf's fixed_point scale is the exponent applied to
// the stored integer (value = raw * 10^scale), so KernelLake's "N digits
// after the decimal point" convention (matching Arrow/Parquet) is
// cudf_scale = -N. See docs/ARCHITECTURE.md.
[[nodiscard]] cudf::data_type to_cudf_type(const DataType& type);

// Converts a literal directly to a cudf::scalar, for operators that need to
// materialize a plain literal as a standalone column (via
// cudf::make_column_from_scalar) rather than through cudf::ast::compute_column
// -- cudf::ast can only produce a fixed-width *output* column, so a CASE
// branch whose result is a STRING literal (e.g. `THEN 'high'`) cannot go
// through the AST path at all, even though a string literal is perfectly
// valid as an intermediate AST node (e.g. one side of `region = 'A'`). See
// docs/ARCHITECTURE.md.
//
// `context` is threaded through to every cudf::scalar constructor call
// (context.stream/context.memory_resource) rather than letting them fall
// back to their own defaults (cudf::get_default_stream()/
// cudf::get_current_device_resource_ref()) -- those defaults mean "the
// process-wide ambient stream/resource," not "this query's own," which
// only happened to be harmless while KernelLake only ever ran one query at
// a time (see GpuExecutionCoordinator's own comment on why that's no
// longer true).
[[nodiscard]] std::unique_ptr<cudf::scalar> literal_to_scalar(const LiteralExpression& expr,
                                                               ExecutionContext& context);

// Builds a cudf::fixed_point_scalar<decimal32/64/128> (width chosen by
// `type.precision`) from a literal's underlying double/int64 value, shifted
// to `type.scale` digits after the decimal point. Used by
// literal_to_scalar() (materializing a DECIMAL literal directly) for the
// exact raw-value/scale construction a DECIMAL literal needs. The literal's
// value only ever has double precision to begin with (see LiteralStorage),
// so this cannot represent more significant digits than a double can; a
// documented limitation, not a bug. Same `context` rationale as
// literal_to_scalar() above.
[[nodiscard]] std::unique_ptr<cudf::scalar> make_decimal_scalar(const DataType& type,
                                                                const LiteralStorage& value, bool is_valid,
                                                                ExecutionContext& context);

// The raw (already-shifted) integer representation and cudf type_id
// (DECIMAL32/64/128) make_decimal_scalar() would build a scalar from --
// exposed separately because cudf::ast::literal's constructor is templated
// on the *concrete* fixed_point_scalar<T> type (no type-erased overload
// exists), so a caller building an AST literal node needs these raw
// ingredients to construct the concrete scalar itself, unlike
// make_decimal_scalar()'s type-erased `unique_ptr<cudf::scalar>` (fine for
// callers like cudf::make_column_from_scalar that only need the base
// interface).
struct DecimalRawValue {
  __int128_t raw;
  std::int32_t cudf_scale;
  cudf::type_id type_id;
};
[[nodiscard]] DecimalRawValue decimal_raw_value(const DataType& type, const LiteralStorage& value);

}  // namespace kernellake
