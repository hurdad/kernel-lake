#pragma once

#include <cstdint>
#include <string_view>

namespace kernellake {

// Shared by sql::AstJoinStep, BoundJoinStep, LogicalJoin, and HashJoinNode/
// HashJoinOperator -- kept in kernellake_types (like PartitionColumn) rather
// than any one of those layers, since it threads through all of them
// unchanged. `LeftOuter` is the only outer-join kind supported so far (no
// RIGHT/FULL -- RIGHT is expressible by swapping sides at the SQL level,
// FULL is unimplemented); see docs/ARCHITECTURE.md's "Hash joins" section.
enum class JoinType : std::uint8_t {
  Inner,
  LeftOuter,
};

[[nodiscard]] constexpr std::string_view to_string(JoinType type) noexcept {
  switch (type) {
    case JoinType::Inner:
      return "INNER";
    case JoinType::LeftOuter:
      return "LEFT OUTER";
  }
  return "UNKNOWN";
}

}  // namespace kernellake
