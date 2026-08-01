#include "kernellake/io/parquet_pruning.hpp"

#include <optional>
#include <sstream>

namespace kernellake {

namespace {

std::optional<double> as_double(const LiteralStorage& value) {
  if (std::holds_alternative<std::int64_t>(value)) {
    return static_cast<double>(std::get<std::int64_t>(value));
  }
  if (std::holds_alternative<double>(value)) {
    return std::get<double>(value);
  }
  return std::nullopt;
}

// Returns -1/0/1, or nullopt if the two values are not comparable (which
// KernelLake treats as "cannot prune", never as a guessed ordering).
std::optional<int> compare_literals(const LiteralStorage& a, const LiteralStorage& b) {
  if (std::holds_alternative<std::string>(a) && std::holds_alternative<std::string>(b)) {
    const std::string& sa = std::get<std::string>(a);
    const std::string& sb = std::get<std::string>(b);
    return sa < sb ? -1 : (sa > sb ? 1 : 0);
  }
  if (std::holds_alternative<bool>(a) && std::holds_alternative<bool>(b)) {
    const bool ba = std::get<bool>(a);
    const bool bb = std::get<bool>(b);
    return ba == bb ? 0 : (bb ? -1 : 1);
  }
  const std::optional<double> da = as_double(a);
  const std::optional<double> db = as_double(b);
  if (da.has_value() && db.has_value()) {
    return *da < *db ? -1 : (*da > *db ? 1 : 0);
  }
  return std::nullopt;
}

std::string literal_to_string(const LiteralStorage& value) {
  if (std::holds_alternative<std::string>(value)) return "'" + std::get<std::string>(value) + "'";
  if (std::holds_alternative<std::int64_t>(value)) return std::to_string(std::get<std::int64_t>(value));
  if (std::holds_alternative<double>(value)) return std::to_string(std::get<double>(value));
  if (std::holds_alternative<bool>(value)) return std::get<bool>(value) ? "TRUE" : "FALSE";
  return "NULL";
}

// Returns a reason string if `predicate` can be proven impossible against
// `stats`, or nullopt if the row group must still be scanned for it.
std::optional<std::string> predicate_proves_empty(const PushablePredicate& predicate,
                                                    const ColumnStatistics& stats) {
  if (!stats.has_min_max) return std::nullopt;
  const auto* literal_expr = dynamic_cast<const LiteralExpression*>(predicate.literal.get());
  if (literal_expr == nullptr || literal_expr->is_null()) return std::nullopt;
  const LiteralStorage& literal = literal_expr->value();

  const std::optional<int> cmp_min = compare_literals(stats.min_value, literal);
  const std::optional<int> cmp_max = compare_literals(stats.max_value, literal);
  if (!cmp_min.has_value() || !cmp_max.has_value()) return std::nullopt;

  std::ostringstream reason;
  reason << predicate.column_name << " " << to_string(predicate.op) << " "
         << literal_to_string(literal) << " vs [" << literal_to_string(stats.min_value) << ", "
         << literal_to_string(stats.max_value) << "]";

  switch (predicate.op) {
    case BinaryOperator::Equal:
      if (*cmp_min > 0 || *cmp_max < 0) return reason.str() + ": literal outside range";
      return std::nullopt;
    case BinaryOperator::NotEqual:
      // Only provably empty when every value in the group equals the
      // literal (min == max == literal): then NotEqual rejects every row.
      if (*cmp_min == 0 && *cmp_max == 0) return reason.str() + ": all values equal literal";
      return std::nullopt;
    case BinaryOperator::Less:
      if (*cmp_min >= 0) return reason.str() + ": min >= literal";
      return std::nullopt;
    case BinaryOperator::LessEqual:
      if (*cmp_min > 0) return reason.str() + ": min > literal";
      return std::nullopt;
    case BinaryOperator::Greater:
      if (*cmp_max <= 0) return reason.str() + ": max <= literal";
      return std::nullopt;
    case BinaryOperator::GreaterEqual:
      if (*cmp_max < 0) return reason.str() + ": max < literal";
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

}  // namespace

ScanDecision evaluate_pruning(const FileMetadata& file,
                               const std::vector<PushablePredicate>& predicates) {
  ScanDecision decision;
  decision.file = file.path;

  for (const RowGroupMetadata& row_group : file.row_groups) {
    std::optional<std::string> skip_reason;
    for (const PushablePredicate& predicate : predicates) {
      const auto it = row_group.column_statistics.find(predicate.column_name);
      if (it == row_group.column_statistics.end()) continue;
      if (std::optional<std::string> reason = predicate_proves_empty(predicate, it->second)) {
        skip_reason = "row_group " + std::to_string(row_group.index) + " skipped: " + *reason;
        break;
      }
    }
    if (skip_reason.has_value()) {
      decision.skipped_row_groups.push_back(row_group.index);
      decision.reasons.push_back(*skip_reason);
    } else {
      decision.selected_row_groups.push_back(row_group.index);
    }
  }

  if (decision.skipped_row_groups.empty() && !predicates.empty()) {
    decision.reasons.push_back("no row groups could be proven safe to skip from available statistics");
  }

  return decision;
}

}  // namespace kernellake
