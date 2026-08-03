#include "commands.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <sstream>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/storage/file_discovery.hpp"
#include "kernellake/storage/object_store_registry.hpp"

namespace kernellake::cli {

namespace {

std::string literal_to_string(const LiteralStorage& value) {
  if (std::holds_alternative<std::string>(value)) {
    return "'" + std::get<std::string>(value) + "'";
  }
  if (std::holds_alternative<std::int64_t>(value)) {
    return std::to_string(std::get<std::int64_t>(value));
  }
  if (std::holds_alternative<double>(value)) {
    return std::to_string(std::get<double>(value));
  }
  if (std::holds_alternative<bool>(value)) {
    return std::get<bool>(value) ? "TRUE" : "FALSE";
  }
  return "NULL";
}

void print_text(const FileMetadata& meta) {
  std::printf("file: %s\n", meta.path.value().c_str());
  std::printf("rows: %lld\n", static_cast<long long>(meta.row_count));
  std::printf("row_groups: %zu\n", meta.row_groups.size());
  std::printf("schema:\n");
  for (const Field& field : meta.schema.fields()) {
    std::printf("  - %s: %s\n", field.name.c_str(), field.type.to_string().c_str());
  }
  for (const RowGroupMetadata& rg : meta.row_groups) {
    std::printf("row_group %d:\n", rg.index);
    std::printf("  rows: %lld\n", static_cast<long long>(rg.row_count));
    std::printf("  compressed_bytes: %lld\n", static_cast<long long>(rg.compressed_size_bytes));
    std::printf("  uncompressed_bytes: %lld\n", static_cast<long long>(rg.uncompressed_size_bytes));
    std::printf("  columns:\n");
    for (const Field& field : meta.schema.fields()) {
      const auto it = rg.column_statistics.find(field.name);
      std::printf("    - %s:", field.name.c_str());
      if (it != rg.column_statistics.end() && it->second.has_min_max) {
        std::printf(" min=%s max=%s", literal_to_string(it->second.min_value).c_str(),
                    literal_to_string(it->second.max_value).c_str());
      } else {
        std::printf(" min=unknown max=unknown");
      }
      if (it != rg.column_statistics.end() && it->second.has_null_count) {
        std::printf(" null_count=%lld", static_cast<long long>(it->second.null_count));
      } else {
        std::printf(" null_count=unknown");
      }
      std::printf("\n");
    }
  }
}

nlohmann::json to_json(const FileMetadata& meta) {
  nlohmann::json j;
  j["file"] = meta.path.value();
  j["rows"] = meta.row_count;
  nlohmann::json schema = nlohmann::json::array();
  for (const Field& field : meta.schema.fields()) {
    schema.push_back({{"name", field.name}, {"type", field.type.to_string()}});
  }
  j["schema"] = std::move(schema);

  nlohmann::json row_groups = nlohmann::json::array();
  for (const RowGroupMetadata& rg : meta.row_groups) {
    nlohmann::json rg_json;
    rg_json["index"] = rg.index;
    rg_json["rows"] = rg.row_count;
    rg_json["compressed_bytes"] = rg.compressed_size_bytes;
    rg_json["uncompressed_bytes"] = rg.uncompressed_size_bytes;
    nlohmann::json columns;
    for (const Field& field : meta.schema.fields()) {
      const auto it = rg.column_statistics.find(field.name);
      nlohmann::json col;
      if (it != rg.column_statistics.end() && it->second.has_min_max) {
        col["min"] = literal_to_string(it->second.min_value);
        col["max"] = literal_to_string(it->second.max_value);
      } else {
        col["min"] = nullptr;
        col["max"] = nullptr;
      }
      if (it != rg.column_statistics.end() && it->second.has_null_count) {
        col["null_count"] = it->second.null_count;
      } else {
        col["null_count"] = nullptr;
      }
      columns[field.name] = std::move(col);
    }
    rg_json["columns"] = std::move(columns);
    row_groups.push_back(std::move(rg_json));
  }
  j["row_groups"] = std::move(row_groups);
  return j;
}

}  // namespace

int run_inspect_parquet(const std::vector<std::string_view>& args, const EngineConfig& config) {
  std::string path;
  std::string format = "text";
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--path" && i + 1 < args.size()) {
      path = args[++i];
    } else if (args[i] == "--format" && i + 1 < args.size()) {
      format = args[++i];
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, "kernellake inspect-parquet: --path is required\n");
    return 1;
  }
  if (format != "text" && format != "json") {
    std::fprintf(stderr, "kernellake inspect-parquet: --format must be 'text' or 'json', got '%s'\n",
                 format.c_str());
    return 1;
  }

  try {
    ObjectStoreRegistry store(config.storage);
    const std::vector<ObjectInfo> files = discover_parquet_files(store, {path});
    std::vector<FileMetadata> metadata;
    metadata.reserve(files.size());
    for (const ObjectInfo& file : files) {
      metadata.push_back(inspect_parquet_file(store, file.uri));
    }
    validate_schema_compatibility(metadata);

    if (format == "json") {
      nlohmann::json array = nlohmann::json::array();
      for (const FileMetadata& meta : metadata) {
        array.push_back(to_json(meta));
      }
      std::printf("%s\n", array.dump(2).c_str());
    } else {
      for (const FileMetadata& meta : metadata) {
        print_text(meta);
      }
    }
  } catch (const KernelLakeError& e) {
    std::fprintf(stderr, "kernellake inspect-parquet: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace kernellake::cli
