#include "kernellake/storage/file_discovery.hpp"

#include <algorithm>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

bool has_parquet_extension(const std::string& path) {
  static constexpr std::string_view kExtension = ".parquet";
  return path.size() >= kExtension.size() &&
         path.compare(path.size() - kExtension.size(), kExtension.size(), kExtension) == 0;
}

}  // namespace

std::vector<ObjectInfo> discover_parquet_files(ObjectStore& store, const std::vector<std::string>& sources) {
  if (sources.empty()) {
    throw StorageError("no data source given (expected FROM read_parquet('path'))");
  }

  std::vector<ObjectInfo> all_files;
  for (const std::string& source : sources) {
    std::vector<ObjectInfo> listed = store.list(Uri(source));
    for (ObjectInfo& info : listed) {
      if (!has_parquet_extension(info.uri.value())) {
        throw StorageError("not a Parquet file (unsupported format): '" + info.uri.value() + "'");
      }
      all_files.push_back(std::move(info));
    }
  }

  std::sort(all_files.begin(), all_files.end(),
            [](const ObjectInfo& a, const ObjectInfo& b) { return a.uri < b.uri; });
  all_files.erase(std::unique(all_files.begin(), all_files.end(),
                              [](const ObjectInfo& a, const ObjectInfo& b) { return a.uri == b.uri; }),
                  all_files.end());
  return all_files;
}

}  // namespace kernellake
