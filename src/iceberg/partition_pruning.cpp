#include "kernellake/iceberg/partition_pruning.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "kernellake/expression/expression.hpp"

namespace kernellake::iceberg {

namespace {

// ---------------------------------------------------------------------------
// Calendar math for the year/month/day/hour transforms. Deliberately
// duplicated rather than reusing kernellake/common/date_util.cpp's private
// is_leap_year()/days_in_month() (not exported) -- same "small helper,
// small enough to just duplicate" convention this codebase already
// follows elsewhere (see e.g. iceberg/schema_translation.cpp's and
// delta/schema_translation.cpp's own duplicated parse_decimal()).
// ---------------------------------------------------------------------------

bool is_leap_year(int year) {
  return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
}

int days_in_month(int year, int month) {
  static constexpr int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month == 2 && is_leap_year(year)) {
    return 29;
  }
  return kDays[month - 1];
}

struct CivilDate {
  int year;
  int month;
};

// The exact inverse of date_util.cpp's parse_iso_date() forward
// (calendar -> days) computation, written the same way (a plain
// year-then-month scan) so the two are easy to eyeball-verify against
// each other as inverses.
CivilDate civil_from_days(std::int64_t days_since_epoch) {
  int year = 1970;
  std::int64_t remaining = days_since_epoch;
  if (remaining >= 0) {
    while (true) {
      const std::int64_t year_days = is_leap_year(year) ? 366 : 365;
      if (remaining < year_days) {
        break;
      }
      remaining -= year_days;
      ++year;
    }
  } else {
    while (remaining < 0) {
      --year;
      remaining += is_leap_year(year) ? 366 : 365;
    }
  }
  int month = 1;
  while (true) {
    const int month_days = days_in_month(year, month);
    if (remaining < month_days) {
      break;
    }
    remaining -= month_days;
    ++month;
  }
  return CivilDate{year, month};
}

constexpr std::int64_t kMicrosPerDay = 86'400'000'000;
constexpr std::int64_t kMicrosPerHour = 3'600'000'000;

// Floor division (C++'s built-in `/` truncates toward zero, which is wrong
// for negative micros -- a pre-1970 timestamp must still floor toward the
// earlier day/hour, not round toward epoch).
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
  const std::int64_t q = a / b;
  const std::int64_t r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

// Applies year/month/day/hour to a Date32 (days since epoch) or Timestamp
// (microseconds since epoch) literal. nullopt when the literal's type
// doesn't match what the transform needs (e.g. `hour` against a Date32
// column, which the Iceberg spec doesn't define -- a date has no
// time-of-day component) or isn't a date/timestamp at all.
std::optional<std::int64_t> apply_time_transform(const std::string& transform,
                                                 const LiteralExpression& literal) {
  if (!std::holds_alternative<std::int64_t>(literal.value())) {
    return std::nullopt;
  }
  const std::int64_t raw = std::get<std::int64_t>(literal.value());
  const TypeId type_id = literal.result_type().id;

  if (transform == "hour") {
    return type_id == TypeId::Timestamp ? std::optional(floor_div(raw, kMicrosPerHour)) : std::nullopt;
  }

  std::int64_t days = 0;
  if (type_id == TypeId::Date32) {
    days = raw;
  } else if (type_id == TypeId::Timestamp) {
    days = floor_div(raw, kMicrosPerDay);
  } else {
    return std::nullopt;
  }

  if (transform == "day") {
    return days;
  }
  const CivilDate civil = civil_from_days(days);
  if (transform == "month") {
    return static_cast<std::int64_t>(civil.year - 1970) * 12 + (civil.month - 1);
  }
  if (transform == "year") {
    return static_cast<std::int64_t>(civil.year - 1970);
  }
  return std::nullopt;
}

// Applies `transform` to a predicate's literal, producing a value directly
// comparable to the manifest's own already-transformed partition value.
// nullopt for bucket[N]/truncate[W]/void, or any transform this function
// doesn't recognize -- see partition_pruning.hpp's own doc comment for
// why those are left unevaluated rather than guessed at.
std::optional<LiteralStorage> apply_transform(const std::string& transform,
                                              const LiteralExpression& literal) {
  if (transform == "identity") {
    return literal.value();
  }
  if (transform == "year" || transform == "month" || transform == "day" || transform == "hour") {
    const std::optional<std::int64_t> value = apply_time_transform(transform, literal);
    if (!value.has_value()) {
      return std::nullopt;
    }
    return *value;
  }
  return std::nullopt;
}

std::optional<LiteralStorage> to_literal_storage(const PartitionFieldValue& value) {
  if (std::holds_alternative<std::int64_t>(value)) {
    return std::get<std::int64_t>(value);
  }
  if (std::holds_alternative<std::string>(value)) {
    return std::get<std::string>(value);
  }
  return std::nullopt;  // monostate: a null partition value -- can't compare, never prune.
}

std::optional<double> as_double(const LiteralStorage& value) {
  if (std::holds_alternative<std::int64_t>(value)) {
    return static_cast<double>(std::get<std::int64_t>(value));
  }
  if (std::holds_alternative<double>(value)) {
    return std::get<double>(value);
  }
  return std::nullopt;
}

// Same shape as kernellake::(anonymous)::compare_literals in
// parquet_pruning.cpp -- duplicated rather than shared, same reasoning as
// this file's own civil_from_days() above.
std::optional<int> compare_literals(const LiteralStorage& a, const LiteralStorage& b) {
  if (std::holds_alternative<std::string>(a) && std::holds_alternative<std::string>(b)) {
    const std::string& sa = std::get<std::string>(a);
    const std::string& sb = std::get<std::string>(b);
    return sa < sb ? -1 : (sa > sb ? 1 : 0);
  }
  const std::optional<double> da = as_double(a);
  const std::optional<double> db = as_double(b);
  if (da.has_value() && db.has_value()) {
    return *da < *db ? -1 : (*da > *db ? 1 : 0);
  }
  return std::nullopt;
}

// `cmp` is compare(partition_value, transformed_literal): -1/0/1. Under a
// coarsening transform (day/month/year/hour, is_identity false), the
// transformed literal only pins down *which bucket* the literal's real
// value falls in, not where within that bucket -- so cmp == 0 means "the
// file's bucket and the literal's bucket are the same", which is
// ambiguous for any operator whose emptiness proof would otherwise rely
// on that exact boundary: NotEqual (a file whose bucket matches the
// literal's bucket can still contain rows unequal to the literal -- see
// partition_pruning.hpp's own doc comment) and the two *strict*
// inequalities Less/Greater (a file whose bucket matches the literal's
// bucket can still contain rows on the correct side of the literal
// within that same bucket, e.g. `ts < '2024-01-15 12:00:00'` against a
// file whose day-partition is also 2024-01-15 -- rows earlier that same
// day still satisfy the predicate, so cmp == 0 must NOT prove empty
// unless is_identity). Equal/LessEqual/GreaterEqual need no such gate:
// their emptiness proofs only ever fire on a strict cmp != 0 (Equal) or
// cmp on the far side of the boundary (cmp > 0 / cmp < 0), never on the
// ambiguous cmp == 0 case itself, so they're already sound under
// coarsening regardless of is_identity -- the same way evaluate_pruning()
// treats a real [min, max] row-group range, here degenerated to a single
// point [V, V].
bool proves_empty(BinaryOperator op, int cmp, bool is_identity) {
  switch (op) {
    case BinaryOperator::Equal:
      return cmp != 0;
    case BinaryOperator::NotEqual:
      return is_identity && cmp == 0;
    case BinaryOperator::Less:
      return is_identity ? cmp >= 0 : cmp > 0;
    case BinaryOperator::LessEqual:
      return cmp > 0;
    case BinaryOperator::Greater:
      return is_identity ? cmp <= 0 : cmp < 0;
    case BinaryOperator::GreaterEqual:
      return cmp < 0;
    default:
      return false;
  }
}

std::string source_column_name(const IcebergTableMetadata& table_metadata, std::int32_t source_id) {
  for (const IcebergSchemaField& field : table_metadata.schema_fields) {
    if (field.id == source_id) {
      return field.name;
    }
  }
  return {};
}

}  // namespace

bool partition_values_prove_empty(const IcebergTableMetadata& table_metadata,
                                  const IcebergPartitionSpec& spec,
                                  const std::vector<PartitionFieldValue>& partition_values,
                                  const std::vector<PushablePredicate>& predicates) {
  const std::size_t field_count = std::min(spec.fields.size(), partition_values.size());
  for (std::size_t i = 0; i < field_count; ++i) {
    const IcebergPartitionField& field = spec.fields[i];
    const std::string column_name = source_column_name(table_metadata, field.source_id);
    if (column_name.empty()) {
      continue;
    }
    const std::optional<LiteralStorage> partition_value = to_literal_storage(partition_values[i]);
    if (!partition_value.has_value()) {
      continue;
    }

    for (const PushablePredicate& predicate : predicates) {
      if (predicate.column_name != column_name) {
        continue;
      }
      const auto* literal_expr = dynamic_cast<const LiteralExpression*>(predicate.literal.get());
      if (literal_expr == nullptr || literal_expr->is_null()) {
        continue;
      }
      const std::optional<LiteralStorage> transformed_literal =
          apply_transform(field.transform, *literal_expr);
      if (!transformed_literal.has_value()) {
        continue;
      }
      // has_value() checked above (both partition_value, before this
      // nested loop, and transformed_literal, just above) -- same
      // false-positive shape as binder.cpp's/result_formatter.cpp's own
      // suppressions of this check.
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      const std::optional<int> cmp = compare_literals(*partition_value, *transformed_literal);
      if (!cmp.has_value()) {
        continue;
      }
      if (proves_empty(predicate.op, *cmp, field.transform == "identity")) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace kernellake::iceberg
