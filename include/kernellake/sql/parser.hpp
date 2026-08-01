#pragma once

#include <string_view>

#include "kernellake/sql/ast.hpp"

namespace kernellake::sql {

// Parses one SQL statement into KernelLake's parser-independent AST.
//
// Supported grammar (see docs/ARCHITECTURE.md for the authoritative list):
//   SELECT <items> FROM read_parquet('path' [, 'path2', ...])
//     [WHERE <expr>] [GROUP BY <cols>] [ORDER BY <cols>] [LIMIT <n>]
// with column references, aliases, numeric/string/boolean/date literals,
// comparison and arithmetic operators, AND/OR/NOT, BETWEEN, IS [NOT] NULL,
// and SUM/COUNT/MIN/MAX/AVG aggregates.
//
// Anything outside that grammar (joins, subqueries, DISTINCT, HAVING, set
// operations, CTEs, window functions, OFFSET, LIKE/IN/CASE, functions other
// than the five aggregates above) throws SqlError with a message identifying
// the unsupported construct rather than silently reinterpreting it.
[[nodiscard]] AstSelectStatement parse_sql(std::string_view sql);

}  // namespace kernellake::sql
