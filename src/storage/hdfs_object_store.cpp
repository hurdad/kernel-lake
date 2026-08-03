#include "kernellake/storage/hdfs_object_store.hpp"

#include <arrow/filesystem/hdfs.h>
#include <fmt/format.h>

#include "generic_fs_object_store.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

// Unlike S3/GCS/Azure ("scheme://bucket/key", where the first path
// component is itself the addressed resource), an "hdfs://namenode:port/path"
// URI's authority names a namenode connection that arrow::fs::HadoopFileSystem
// already has from HdfsOptions.connection_config (storage.hdfs.connection_config
// in config) -- passing it through as part of the path argument to
// GetFileInfo()/OpenInputFile() would be wrong. generic_fs_list()/
// generic_fs_open() (shared with S3/GCS/Azure, via detail::strip_scheme())
// only strip the "hdfs://" prefix, so this additionally strips the
// authority before delegating, leaving just the real path.
std::shared_ptr<arrow::fs::FileSystem> make_hdfs_filesystem(const HdfsSection& config) {
  const arrow::Result<std::shared_ptr<arrow::fs::HadoopFileSystem>> result =
      arrow::fs::HadoopFileSystem::Make(config.options);
  if (!result.ok()) {
    throw StorageError(fmt::format("hdfs: failed to construct filesystem: {}", result.status().ToString()));
  }
  return *result;
}

Uri strip_authority(const Uri& uri) {
  const std::string path = detail::strip_scheme(uri);
  const std::size_t slash = path.find('/');
  if (slash == std::string::npos) {
    return Uri("hdfs://" + path);
  }
  return Uri("hdfs://" + path.substr(slash));
}

}  // namespace

HdfsObjectStore::HdfsObjectStore(const HdfsSection& config) : fs_(make_hdfs_filesystem(config)) {}

std::vector<ObjectInfo> HdfsObjectStore::list(const Uri& prefix) {
  return detail::generic_fs_list(fs_, "hdfs", strip_authority(prefix));
}

std::unique_ptr<RandomAccessObject> HdfsObjectStore::open(const Uri& uri) {
  return detail::generic_fs_open(fs_, "hdfs", strip_authority(uri));
}

}  // namespace kernellake
