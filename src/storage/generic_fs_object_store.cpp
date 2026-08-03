#include "generic_fs_object_store.hpp"

#include <algorithm>
#include <filesystem>

#include "kernellake/common/errors.hpp"

namespace kernellake::detail {

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

std::string strip_scheme(const Uri& uri) {
  const std::string_view scheme = uri.scheme();
  if (scheme == "file") return uri.value();
  // scheme + "://" -- Uri::scheme() finds the text before "://" itself, so
  // this is always a valid offset for a non-"file" scheme.
  return uri.value().substr(scheme.size() + 3);
}

namespace {

class GenericRandomAccessObject final : public RandomAccessObject {
 public:
  explicit GenericRandomAccessObject(std::shared_ptr<arrow::io::RandomAccessFile> file)
      : file_(std::move(file)) {}

  [[nodiscard]] std::uint64_t size() const override {
    const arrow::Result<std::int64_t> result = file_->GetSize();
    if (!result.ok()) {
      throw StorageError("failed to stat opened file: " + result.status().ToString());
    }
    return static_cast<std::uint64_t>(*result);
  }

  [[nodiscard]] std::shared_ptr<arrow::io::RandomAccessFile> as_arrow_file() const override { return file_; }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
};

// Splits "bucket/prefix/*.parquet" into ("bucket/prefix", "*.parquet") --
// the directory to list and the glob pattern to filter its immediate
// children by. `path` has no leading slash and no scheme (already stripped
// by strip_scheme).
std::pair<std::string, std::string> split_last_component(const std::string& path) {
  const std::size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return {"", path};
  return {path.substr(0, slash), path.substr(slash + 1)};
}

}  // namespace

std::vector<ObjectInfo> generic_fs_list(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                        std::string_view backend_label, const Uri& prefix) {
  const std::string path = strip_scheme(prefix);
  const auto [dir, last_component] = split_last_component(path);

  if (has_glob_chars(last_component)) {
    arrow::fs::FileSelector selector;
    selector.base_dir = dir;
    selector.allow_not_found = true;
    const arrow::Result<arrow::fs::FileInfoVector> result = fs->GetFileInfo(selector);
    if (!result.ok()) {
      throw StorageError(std::string(backend_label) + ": failed to list '" + prefix.value() +
                         "': " + result.status().ToString());
    }
    std::vector<ObjectInfo> results;
    for (const arrow::fs::FileInfo& info : *result) {
      if (!info.IsFile()) continue;
      const auto [entry_dir, entry_name] = split_last_component(info.path());
      if (!glob_match(last_component, entry_name)) continue;
      results.push_back(ObjectInfo{Uri(std::string(prefix.scheme()) + "://" + info.path()),
                                   static_cast<std::uint64_t>(info.size())});
    }
    if (results.empty()) {
      throw StorageError(std::string(backend_label) + ": no files matched pattern '" + prefix.value() + "'");
    }
    std::sort(results.begin(), results.end(),
              [](const ObjectInfo& a, const ObjectInfo& b) { return a.uri < b.uri; });
    return results;
  }

  const arrow::Result<arrow::fs::FileInfo> info_result = fs->GetFileInfo(path);
  if (!info_result.ok()) {
    throw StorageError(std::string(backend_label) + ": failed to inspect '" + prefix.value() +
                       "': " + info_result.status().ToString());
  }
  const arrow::fs::FileInfo& info = *info_result;
  if (info.type() == arrow::fs::FileType::NotFound) {
    throw StorageError(std::string(backend_label) + ": path does not exist: '" + prefix.value() + "'");
  }

  if (info.IsDirectory()) {
    arrow::fs::FileSelector selector;
    selector.base_dir = path;
    selector.allow_not_found = true;
    const arrow::Result<arrow::fs::FileInfoVector> dir_result = fs->GetFileInfo(selector);
    if (!dir_result.ok()) {
      throw StorageError(std::string(backend_label) + ": failed to list '" + prefix.value() +
                         "': " + dir_result.status().ToString());
    }
    std::vector<ObjectInfo> results;
    for (const arrow::fs::FileInfo& entry : *dir_result) {
      if (!entry.IsFile()) continue;
      if (entry.extension() != "parquet") continue;
      results.push_back(ObjectInfo{Uri(std::string(prefix.scheme()) + "://" + entry.path()),
                                   static_cast<std::uint64_t>(entry.size())});
    }
    if (results.empty()) {
      throw StorageError(std::string(backend_label) + ": directory contains no Parquet files: '" +
                         prefix.value() + "'");
    }
    std::sort(results.begin(), results.end(),
              [](const ObjectInfo& a, const ObjectInfo& b) { return a.uri < b.uri; });
    return results;
  }

  return {ObjectInfo{prefix, static_cast<std::uint64_t>(info.size())}};
}

std::unique_ptr<RandomAccessObject> generic_fs_open(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                    std::string_view backend_label, const Uri& uri) {
  const std::string path = strip_scheme(uri);
  const arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> result = fs->OpenInputFile(path);
  if (!result.ok()) {
    throw StorageError(std::string(backend_label) + ": failed to open '" + uri.value() +
                       "': " + result.status().ToString());
  }
  return std::make_unique<GenericRandomAccessObject>(*result);
}

}  // namespace kernellake::detail
