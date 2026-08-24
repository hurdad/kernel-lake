#pragma once

#include <cstdint>
#include <string_view>

namespace kernellake {

// Shared by sql::AstJoinStep, BoundJoinStep, LogicalJoin, and HashJoinNode/
// HashJoinOperator/SemiAntiJoinOperator -- kept in kernellake_types (like
// PartitionColumn) rather than any one of those layers, since it threads
// through all of them unchanged. `LeftOuter` is the only outer-join kind
// supported so far (no RIGHT/FULL -- RIGHT is expressible by swapping
// sides at the SQL level, FULL is unimplemented); see
// docs/ARCHITECTURE.md's "Hash joins" section.
//
// `LeftSemi`/`LeftAnti` never appear directly in SQL syntax -- there is no
// `LEFT SEMI JOIN`/`LEFT ANTI JOIN` keyword this project's grammar
// accepts. They're produced only by the `EXISTS`/`NOT EXISTS` ->
// join-step rewrite (see sql::rewrite_exists_subqueries(), the same
// "sugar" relationship `IN (SELECT ...)`/HAVING's own scalar subquery
// already have to their own resolved forms -- see ast.hpp's own comments
// on those). Like `LeftOuter`, `left` is the preserved/probe side; unlike
// every other JoinType, the *output* contributes zero columns from the
// right/build side at all (`LeftSemi`: left rows with >=1 match; `LeftAnti`:
// left rows with none) -- see LogicalJoin::build_schema()'s own comment for
// why this changes how a join step's own combined-field-count accounting
// works for every step after it in a chain.
enum class JoinType : std::uint8_t {
  Inner,
  LeftOuter,
  LeftSemi,
  LeftAnti,
};

[[nodiscard]] constexpr std::string_view to_string(JoinType type) noexcept {
  switch (type) {
    case JoinType::Inner:
      return "INNER";
    case JoinType::LeftOuter:
      return "LEFT OUTER";
    case JoinType::LeftSemi:
      return "LEFT SEMI";
    case JoinType::LeftAnti:
      return "LEFT ANTI";
  }
  return "UNKNOWN";
}

}  // namespace kernellake
