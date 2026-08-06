#include "kernellake/iceberg/schema_translation.hpp"

#include <fmt/format.h>

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake::iceberg {

namespace {

// Parses "decimal(P,S)" -> (P, S). Iceberg's decimal type string is always
// exactly this shape (spec-fixed, no whitespace variants to tolerate) --
// anything else falls through to the caller's "unsupported type" error
// rather than this function guessing at a looser grammar.
std::optional<std::pair<std::int32_t, std::int32_t>> parse_decimal(const std::string& type) {
  constexpr std::string_view kPrefix = "decimal(";
  if (type.size() < kPrefix.size() + 2 || type.compare(0, kPrefix.size(), kPrefix) != 0 ||
      type.back() != ')') {
    return std::nullopt;
  }
  const std::string_view inside(type.data() + kPrefix.size(), type.size() - kPrefix.size() - 1);
  const std::size_t comma = inside.find(',');
  if (comma == std::string_view::npos) {
    return std::nullopt;
  }
  const std::string_view precision_str = inside.substr(0, comma);
  const std::string_view scale_str = inside.substr(comma + 1);

  std::int32_t precision = 0;
  std::int32_t scale = 0;
  const auto precision_result =
      std::from_chars(precision_str.data(), precision_str.data() + precision_str.size(), precision);
  const auto scale_result = std::from_chars(scale_str.data(), scale_str.data() + scale_str.size(), scale);
  if (precision_result.ec != std::errc{} || precision_result.ptr != precision_str.data() + precision_str.size() ||
      scale_result.ec != std::errc{} || scale_result.ptr != scale_str.data() + scale_str.size()) {
    return std::nullopt;
  }
  return std::make_pair(precision, scale);
}

DataType translate_type(const IcebergSchemaField& field) {
  const bool nullable = !field.required;
  const std::string& type = field.type;

  if (type == "boolean") return boolean_type(nullable);
  if (type == "int") return int32_type(nullable);
  if (type == "long") return int64_type(nullable);
  if (type == "float") return float32_type(nullable);
  if (type == "double") return float64_type(nullable);
  if (type == "date") return date32_type(nullable);
  if (type == "timestamp" || type == "timestamptz") return timestamp_type(nullable);
  if (type == "string") return string_type(nullable);

  if (const std::optional<std::pair<std::int32_t, std::int32_t>> decimal = parse_decimal(type);
      decimal.has_value()) {
    return decimal_type(decimal->first, decimal->second, nullable);
  }

  throw StorageError(fmt::format(
      "iceberg schema: column '{}' has unsupported Iceberg type '{}' (time/uuid/fixed/binary and nested "
      "list/map/struct types aren't supported yet)",
      field.name, type));
}

}  // namespace

Schema iceberg_schema_to_kernellake_schema(const std::vector<IcebergSchemaField>& fields) {
  std::vector<Field> kernellake_fields;
  kernellake_fields.reserve(fields.size());
  for (const IcebergSchemaField& field : fields) {
    kernellake_fields.push_back(Field{field.name, translate_type(field)});
  }
  return Schema(std::move(kernellake_fields));
}

}  // namespace kernellake::iceberg
