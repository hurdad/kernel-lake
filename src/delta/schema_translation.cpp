#include "kernellake/delta/schema_translation.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake::delta {

namespace {

// Parses "decimal(P,S)" -> (P, S). Delta's decimal type string is always
// exactly this shape (Spark-schema-compatible, no whitespace variants to
// tolerate) -- anything else falls through to the caller's "unsupported
// type" error rather than this function guessing at a looser grammar. Same
// helper shape as kernellake::iceberg's own parse_decimal (deliberately
// duplicated, not shared -- this project doesn't factor out a
// cross-table-format utility target for a single small parser).
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
  if (precision_result.ec != std::errc{} ||
      precision_result.ptr != precision_str.data() + precision_str.size() || scale_result.ec != std::errc{} ||
      scale_result.ptr != scale_str.data() + scale_str.size()) {
    return std::nullopt;
  }
  return std::make_pair(precision, scale);
}

DataType translate_type(const std::string& name, const std::string& type, bool nullable) {
  if (type == "boolean") {
    return boolean_type(nullable);
  }
  if (type == "byte" || type == "short" || type == "integer") {
    return int32_type(nullable);
  }
  if (type == "long") {
    return int64_type(nullable);
  }
  if (type == "float") {
    return float32_type(nullable);
  }
  if (type == "double") {
    return float64_type(nullable);
  }
  if (type == "date") {
    return date32_type(nullable);
  }
  if (type == "timestamp" || type == "timestamp_ntz") {
    return timestamp_type(nullable);
  }
  if (type == "string") {
    return string_type(nullable);
  }

  if (const std::optional<std::pair<std::int32_t, std::int32_t>> decimal = parse_decimal(type);
      decimal.has_value()) {
    return decimal_type(decimal->first, decimal->second, nullable);
  }

  throw StorageError(
      fmt::format("delta schema: column '{}' has unsupported Delta type '{}' (binary/void and nested "
                  "struct/array/map types aren't supported yet)",
                  name, type));
}

}  // namespace

Schema delta_schema_to_kernellake_schema(const std::string& schema_json) {
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(schema_json);
  } catch (const nlohmann::json::exception& e) {
    throw StorageError(fmt::format("delta schema: malformed schema_string JSON: {}", e.what()));
  }

  if (!parsed.contains("fields")) {
    throw StorageError("delta schema: schema_string has no top-level 'fields' array");
  }

  std::vector<Field> fields;
  for (const nlohmann::json& field_json : parsed.at("fields")) {
    try {
      const std::string name = field_json.at("name").get<std::string>();
      const bool nullable = field_json.value("nullable", true);
      const nlohmann::json& type_json = field_json.at("type");
      // A primitive type is a bare JSON string ("long", "decimal(10,2)",
      // ...); a nested type (struct/array/map) is a JSON object -- dumping
      // it back to text here just gives translate_type() something
      // printable to name in its "unsupported type" error, the same
      // fallback kernellake::iceberg's own field-type parsing uses.
      const std::string type = type_json.is_string() ? type_json.get<std::string>() : type_json.dump();
      fields.push_back(Field{name, translate_type(name, type, nullable)});
    } catch (const nlohmann::json::exception& e) {
      throw StorageError(fmt::format("delta schema: malformed field entry: {}", e.what()));
    }
  }
  return Schema(std::move(fields));
}

}  // namespace kernellake::delta
