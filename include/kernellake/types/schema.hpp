#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kernellake {

enum class TypeId : std::uint8_t {
  Boolean,
  Int32,
  Int64,
  UInt32,
  UInt64,
  Float32,
  Float64,
  String,
  Date32,
  Timestamp,
  Decimal,
};

[[nodiscard]] std::string_view to_string(TypeId id) noexcept;

struct DataType {
  TypeId id;
  bool nullable = true;
  std::optional<std::int32_t> precision;
  std::optional<std::int32_t> scale;

  [[nodiscard]] bool operator==(const DataType& other) const noexcept {
    return id == other.id && nullable == other.nullable && precision == other.precision &&
           scale == other.scale;
  }

  [[nodiscard]] std::string to_string() const;
};

// Convenience factories for the common (non-decimal) types.
[[nodiscard]] DataType boolean_type(bool nullable = true);
[[nodiscard]] DataType int32_type(bool nullable = true);
[[nodiscard]] DataType int64_type(bool nullable = true);
[[nodiscard]] DataType uint32_type(bool nullable = true);
[[nodiscard]] DataType uint64_type(bool nullable = true);
[[nodiscard]] DataType float32_type(bool nullable = true);
[[nodiscard]] DataType float64_type(bool nullable = true);
[[nodiscard]] DataType string_type(bool nullable = true);
[[nodiscard]] DataType date32_type(bool nullable = true);
[[nodiscard]] DataType timestamp_type(bool nullable = true);
[[nodiscard]] DataType decimal_type(std::int32_t precision, std::int32_t scale, bool nullable = true);

struct Field {
  std::string name;
  DataType type;

  [[nodiscard]] bool operator==(const Field& other) const noexcept {
    return name == other.name && type == other.type;
  }
};

// An ordered, immutable list of named, typed fields. Independent of Arrow,
// Parquet, and libcudf; see arrow_adapter.hpp for conversions to/from Arrow.
class Schema {
 public:
  explicit Schema(std::vector<Field> fields);

  [[nodiscard]] const std::vector<Field>& fields() const noexcept { return fields_; }

  [[nodiscard]] std::optional<std::size_t> find_field(std::string_view name) const;

  [[nodiscard]] const Field& field(std::size_t index) const { return fields_.at(index); }

  [[nodiscard]] std::size_t field_count() const noexcept { return fields_.size(); }

  // Returns true if `other` has the same field names and types in the same
  // order. Used to validate schema compatibility across multiple Parquet
  // files that make up one logical dataset.
  [[nodiscard]] bool equals(const Schema& other) const noexcept;

 private:
  std::vector<Field> fields_;
};

}  // namespace kernellake
