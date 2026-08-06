#include "kernellake/io/table_resolution.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <utility>

#include "kernellake/common/date_util.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/storage/file_discovery.hpp"

namespace kernellake {

namespace {

std::vector<std::string> split_path_segments(const std::string& path) {
  std::vector<std::string> segments;
  std::string current;
  for (const char c : path) {
    if (c == '/') {
      if (!current.empty()) {
        segments.push_back(current);
      }
      current.clear();
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    segments.push_back(current);
  }
  return segments;
}

std::optional<std::pair<std::string, std::string>> parse_key_value_segment(const std::string& segment) {
  const std::size_t eq = segment.find('=');
  if (eq == std::string::npos || eq == 0) {
    return std::nullopt;
  }
  const std::string key = segment.substr(0, eq);
  if (std::isdigit(static_cast<unsigned char>(key[0]))) {
    return std::nullopt;
  }
  for (const char c : key) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      return std::nullopt;
    }
  }
  return std::make_pair(key, segment.substr(eq + 1));
}

// Scans path segments from the filename backward, stopping at the first
// segment that isn't `key=value`-shaped, then reverses the result so it
// reads root-to-leaf (outermost partition directory first) -- matching the
// order a `SELECT region, date, ...` query would expect the appended
// partition columns to appear in.
std::vector<std::pair<std::string, std::string>> extract_partition_segments(const std::string& path) {
  std::vector<std::string> segments = split_path_segments(path);
  if (segments.empty()) {
    return {};
  }
  segments.pop_back();  // the filename itself is never a partition segment.

  std::vector<std::pair<std::string, std::string>> pairs;
  for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
    std::optional<std::pair<std::string, std::string>> parsed = parse_key_value_segment(*it);
    if (!parsed) {
      break;
    }
    pairs.push_back(std::move(*parsed));
  }
  std::reverse(pairs.begin(), pairs.end());
  return pairs;
}

bool looks_like_int64(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  std::size_t start = 0;
  if (value[0] == '-') {
    start = 1;
  }
  if (start >= value.size()) {
    return false;
  }
  for (std::size_t i = start; i < value.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(value[i]))) {
      return false;
    }
  }
  return true;
}

bool looks_like_iso_date(const std::string& value) {
  if (value.size() != 10) {
    return false;
  }
  for (std::size_t i = 0; i < value.size(); ++i) {
    if (i == 4 || i == 7) {
      if (value[i] != '-') {
        return false;
      }
    } else if (!std::isdigit(static_cast<unsigned char>(value[i]))) {
      return false;
    }
  }
  return true;
}

DataType infer_partition_type(const std::vector<std::string>& values) {
  bool all_int = true;
  bool all_date = true;
  for (const std::string& value : values) {
    if (!looks_like_int64(value)) {
      all_int = false;
    }
    if (!looks_like_iso_date(value)) {
      all_date = false;
    }
  }
  if (all_int) {
    return int64_type(false);
  }
  if (all_date) {
    // Digit-shaped isn't the same as a real calendar date ("2026-13-40"
    // matches looks_like_iso_date's shape check but isn't a valid date) --
    // fall back to string rather than let parse_iso_date throw later.
    for (const std::string& value : values) {
      try {
        (void)parse_iso_date(value);
      } catch (const SqlError&) {
        return string_type(false);
      }
    }
    return date32_type(false);
  }
  return string_type(false);
}

LiteralStorage parse_partition_value(const std::string& value, const DataType& type) {
  switch (type.id) {
    case TypeId::Int64:
      return static_cast<std::int64_t>(std::stoll(value));
    case TypeId::Date32:
      return static_cast<std::int64_t>(parse_iso_date(value));
    default:
      return value;
  }
}

}  // namespace

ResolvedTable resolve_table(ObjectStore& store, const std::vector<std::string>& sources) {
  const std::vector<ObjectInfo> files = discover_parquet_files_recursive(store, sources);

  std::vector<FileMetadata> metadata;
  metadata.reserve(files.size());
  for (const ObjectInfo& file : files) {
    metadata.push_back(inspect_parquet_file(store, file.uri));
  }
  validate_schema_compatibility(metadata);

  // Detect Hive-style partitioning: every file's path must yield the exact
  // same sequence of `key=value` directory segments immediately above the
  // file itself, or none at all. A mix (some files partitioned, some not,
  // or a different key sequence) is a broken/ambiguous layout, rejected
  // outright rather than guessed at.
  std::vector<std::vector<std::pair<std::string, std::string>>> per_file_segments;
  per_file_segments.reserve(files.size());
  for (const ObjectInfo& file : files) {
    per_file_segments.push_back(extract_partition_segments(file.uri.value()));
  }
  std::vector<std::string> partition_keys;
  if (!per_file_segments.empty()) {
    for (const auto& [key, value] : per_file_segments.front()) {
      partition_keys.push_back(key);
    }
  }
  for (std::size_t i = 0; i < per_file_segments.size(); ++i) {
    const std::vector<std::pair<std::string, std::string>>& segments = per_file_segments[i];
    if (segments.size() != partition_keys.size()) {
      throw StorageError(fmt::format(
          "inconsistent Hive-style partition layout: '{}' has {} partition directory level(s), expected {} "
          "(from '{}')",
          files[i].uri.value(), segments.size(), partition_keys.size(), files.front().uri.value()));
    }
    for (std::size_t k = 0; k < segments.size(); ++k) {
      if (segments[k].first != partition_keys[k]) {
        throw StorageError(fmt::format(
            "inconsistent Hive-style partition layout: '{}' has partition key '{}' at level {}, expected "
            "'{}' (from '{}')",
            files[i].uri.value(), segments[k].first, k, partition_keys[k], files.front().uri.value()));
      }
    }
  }

  std::vector<PartitionColumn> partition_columns;
  partition_columns.reserve(partition_keys.size());
  for (std::size_t k = 0; k < partition_keys.size(); ++k) {
    if (metadata.front().schema.find_field(partition_keys[k])) {
      throw StorageError(
          fmt::format("Hive partition column '{}' collides with an existing column of the same name in "
                      "the Parquet file itself",
                      partition_keys[k]));
    }
    std::vector<std::string> values;
    values.reserve(per_file_segments.size());
    for (const std::vector<std::pair<std::string, std::string>>& segments : per_file_segments) {
      values.push_back(segments[k].second);
    }
    partition_columns.push_back(PartitionColumn{partition_keys[k], infer_partition_type(values)});
  }

  std::vector<Field> fields = metadata.front().schema.fields();
  for (const PartitionColumn& column : partition_columns) {
    fields.push_back(Field{column.name, column.type});
  }
  Schema schema(std::move(fields));

  std::vector<ResolvedFile> resolved_files;
  resolved_files.reserve(files.size());
  for (std::size_t i = 0; i < files.size(); ++i) {
    std::vector<LiteralStorage> values;
    values.reserve(partition_columns.size());
    for (std::size_t k = 0; k < partition_columns.size(); ++k) {
      values.push_back(parse_partition_value(per_file_segments[i][k].second, partition_columns[k].type));
    }
    resolved_files.push_back(ResolvedFile{metadata[i], std::move(values)});
  }

  return ResolvedTable{std::move(resolved_files), std::move(schema), std::move(partition_columns)};
}

ResolvedTable resolve_table_or_delegate(ObjectStore& store, const std::vector<std::string>& sources,
                                        TableSourceResolver* extra_resolver) {
  if (extra_resolver != nullptr && extra_resolver->can_resolve(sources)) {
    return extra_resolver->resolve(store, sources);
  }
  return resolve_table(store, sources);
}

}  // namespace kernellake
