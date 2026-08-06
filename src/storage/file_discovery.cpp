#include "kernellake/storage/file_discovery.hpp"

#include <fmt/format.h>

#include <algorithm>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

bool has_parquet_extension(const std::string& path) {
  static constexpr std::string_view kExtension = ".parquet";
  return path.size() >= kExtension.size() &&
         path.compare(path.size() - kExtension.size(), kExtension.size(), kExtension) == 0;
}

bool has_glob_chars(const std::string& path) {
  return path.find('*') != std::string::npos || path.find('?') != std::string::npos;
}

std::vector<ObjectInfo> sorted_deduped(std::vector<ObjectInfo> files) {
  std::sort(files.begin(), files.end(),
            [](const ObjectInfo& a, const ObjectInfo& b) { return a.uri < b.uri; });
  files.erase(std::unique(files.begin(), files.end(),
                          [](const ObjectInfo& a, const ObjectInfo& b) { return a.uri == b.uri; }),
              files.end());
  return files;
}

std::vector<ObjectInfo> check_all_parquet(std::vector<ObjectInfo> files) {
  for (const ObjectInfo& info : files) {
    if (!has_parquet_extension(info.uri.value())) {
      throw StorageError(fmt::format("not a Parquet file (unsupported format): '{}'", info.uri.value()));
    }
  }
  return files;
}

}  // namespace

std::vector<ObjectInfo> discover_parquet_files(ObjectStore& store, const std::vector<std::string>& sources) {
  if (sources.empty()) {
    throw StorageError("no data source given (expected FROM read_parquet('path'))");
  }

  std::vector<ObjectInfo> all_files;
  for (const std::string& source : sources) {
    std::vector<ObjectInfo> listed = check_all_parquet(store.list(Uri(source)));
    all_files.insert(all_files.end(), std::make_move_iterator(listed.begin()),
                     std::make_move_iterator(listed.end()));
  }
  return sorted_deduped(std::move(all_files));
}

std::vector<ObjectInfo> discover_parquet_files_recursive(ObjectStore& store,
                                                         const std::vector<std::string>& sources) {
  if (sources.empty()) {
    throw StorageError("no data source given (expected FROM read_parquet('path'))");
  }

  std::vector<ObjectInfo> all_files;
  for (const std::string& source : sources) {
    std::vector<ObjectInfo> listed;
    if (has_glob_chars(source)) {
      listed = store.list(Uri(source));
    } else {
      try {
        listed = store.list_recursive(Uri(source));
      } catch (const StorageError&) {
        // `source` isn't actually a directory (an exact single file) --
        // list_recursive() rejects that outright; fall back to list()'s
        // existing exact-file handling. A real directory always succeeds
        // above (recursive listing is a strict superset of a flat one), so
        // this fallback only ever fires for the single-file case, never
        // masking a genuine listing failure with a different one.
        listed = store.list(Uri(source));
      }
    }
    listed = check_all_parquet(std::move(listed));
    all_files.insert(all_files.end(), std::make_move_iterator(listed.begin()),
                     std::make_move_iterator(listed.end()));
  }
  return sorted_deduped(std::move(all_files));
}

}  // namespace kernellake
