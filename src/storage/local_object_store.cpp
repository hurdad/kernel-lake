#include "kernellake/storage/local_object_store.hpp"

#include <arrow/io/file.h>

#include <algorithm>
#include <filesystem>
#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

namespace fs = std::filesystem;

// Classic iterative wildcard matcher: '*' matches any sequence (including
// empty), '?' matches exactly one character. Every other character must
// match literally.
bool glob_match(std::string_view pattern, std::string_view text) {
  std::size_t p = 0, t = 0;
  std::size_t star_p = std::string_view::npos, star_t = 0;
  while (t < text.size()) {
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
      ++p;
      ++t;
    } else if (p < pattern.size() && pattern[p] == '*') {
      star_p = p++;
      star_t = t;
    } else if (star_p != std::string_view::npos) {
      p = star_p + 1;
      t = ++star_t;
    } else {
      return false;
    }
  }
  while (p < pattern.size() && pattern[p] == '*') ++p;
  return p == pattern.size();
}

bool has_glob_chars(std::string_view text) {
  return text.find('*') != std::string_view::npos || text.find('?') != std::string_view::npos;
}

[[noreturn]] void fail_missing(const std::string& path) {
  throw StorageError("path does not exist: '" + path +
                      "' (check the path and that KernelLake has read access)");
}

std::vector<ObjectInfo> list_directory_matching(const fs::path& dir, std::string_view pattern) {
  std::vector<ObjectInfo> results;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string filename = entry.path().filename().string();
    if (!glob_match(pattern, filename)) continue;
    results.push_back(ObjectInfo{Uri(entry.path().string()),
                                  static_cast<std::uint64_t>(entry.file_size())});
  }
  std::sort(results.begin(), results.end(),
            [](const ObjectInfo& a, const ObjectInfo& b) { return a.uri < b.uri; });
  return results;
}

class LocalRandomAccessObject final : public RandomAccessObject {
public:
  explicit LocalRandomAccessObject(std::shared_ptr<arrow::io::ReadableFile> file)
      : file_(std::move(file)) {}

  [[nodiscard]] std::uint64_t size() const override {
    const arrow::Result<std::int64_t> result = file_->GetSize();
    if (!result.ok()) {
      throw StorageError("failed to stat opened file: " + result.status().ToString());
    }
    return static_cast<std::uint64_t>(*result);
  }

  [[nodiscard]] std::shared_ptr<arrow::io::RandomAccessFile> as_arrow_file() const override {
    return file_;
  }

private:
  std::shared_ptr<arrow::io::ReadableFile> file_;
};

}  // namespace

std::vector<ObjectInfo> LocalObjectStore::list(const Uri& prefix) {
  const fs::path path(prefix.value());

  if (has_glob_chars(path.filename().string())) {
    const fs::path dir = path.parent_path().empty() ? fs::path(".") : path.parent_path();
    if (!fs::exists(dir) || !fs::is_directory(dir)) fail_missing(prefix.value());
    std::vector<ObjectInfo> results = list_directory_matching(dir, path.filename().string());
    if (results.empty()) {
      throw StorageError("no files matched pattern '" + prefix.value() + "'");
    }
    return results;
  }

  std::error_code ec;
  const bool exists = fs::exists(path, ec);
  if (!exists) fail_missing(prefix.value());

  if (fs::is_directory(path, ec)) {
    std::vector<ObjectInfo> results = list_directory_matching(path, "*.parquet");
    if (results.empty()) {
      throw StorageError("directory contains no Parquet files: '" + prefix.value() + "'");
    }
    return results;
  }

  return {ObjectInfo{Uri(fs::absolute(path).string()),
                      static_cast<std::uint64_t>(fs::file_size(path))}};
}

std::unique_ptr<RandomAccessObject> LocalObjectStore::open(const Uri& uri) {
  arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> result =
      arrow::io::ReadableFile::Open(uri.value());
  if (!result.ok()) {
    throw StorageError("failed to open '" + uri.value() + "': " + result.status().ToString());
  }
  return std::make_unique<LocalRandomAccessObject>(*result);
}

}  // namespace kernellake
