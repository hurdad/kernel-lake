#pragma once

#include <arrow/compute/expression.h>
#include <arrow/datum.h>

#include "kernellake/expression/expression.hpp"

namespace kernellake {

// Converts a LiteralExpression into an arrow::Datum-wrapped scalar of the
// matching Arrow type. Exposed (not just compile_expression_cpu()'s own
// internal use for AST literal nodes) so other CPU-backend code needing a
// one-off constant Arrow value from a KernelLake literal -- e.g. materializing
// a Hive partition column's per-fragment constant value, see
// acero_query_executor.cpp -- doesn't duplicate this type-mapping table.
// Throws ExecutionError for TypeId::Decimal (not yet supported on this
// backend, same restriction as compile_expression_cpu()'s CAST handling).
[[nodiscard]] arrow::Datum literal_to_arrow_datum(const LiteralExpression& literal);

// Compiles a kernellake::Expression tree into an arrow::compute::Expression
// for Arrow Acero's Filter/Project nodes -- the CPU execution backend's
// equivalent of kernellake/execution/expression_compiler.hpp's cudf::ast
// compiler (see docs/ARCHITECTURE.md's CPU backend section). Structurally
// simpler than that one: arrow::compute::Expression is a plain, cheap-to-
// copy value type (built via field_ref()/literal()/call()), unlike
// cudf::ast's tree-node-reference model, so no owned tree/scalar arena is
// needed here -- this is a pure function, not a stateful compiler object.
//
// Scope for this phase: plain columns, numeric/string/boolean/date/
// timestamp literals, arithmetic, comparisons, AND/OR/NOT/IS [NOT] NULL,
// BETWEEN, and numeric/date/timestamp CAST (needed even though explicit
// SQL `CAST` is out of this phase's stated scope -- the binder's own safe
// implicit numeric promotion, e.g. comparing an INT32 column against an
// integer literal, silently inserts a CastExpression for ordinary
// comparisons, so this compiler must handle it or reject far more queries
// than intended). LIKE/IN/CASE/CAST-to-DECIMAL/CAST-to-STRING are not yet
// supported -- throws ExecutionError naming the unsupported node/target
// rather than silently miscompiling.
[[nodiscard]] arrow::compute::Expression compile_expression_cpu(const Expression& expr);

}  // namespace kernellake
