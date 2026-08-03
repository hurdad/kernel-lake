#include "kernellake/storage/object_store_registry.hpp"

#include <fmt/format.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/azure_object_store.hpp"
#include "kernellake/storage/gcs_object_store.hpp"
#include "kernellake/storage/hdfs_object_store.hpp"
#include "kernellake/storage/s3_object_store.hpp"

namespace kernellake {

ObjectStore& ObjectStoreRegistry::backend_for(std::string_view scheme) {
  if (scheme == "file") {
    return local_;
  }
  if (scheme == "s3") {
    if (!s3_) {
      s3_ = std::make_unique<S3ObjectStore>(config_.s3);
    }
    return *s3_;
  }
  if (scheme == "gs" || scheme == "gcs") {
    if (!gcs_) {
      gcs_ = std::make_unique<GcsObjectStore>(config_.gcs);
    }
    return *gcs_;
  }
  if (scheme == "abfs" || scheme == "abfss" || scheme == "az") {
    if (!azure_) {
      azure_ = std::make_unique<AzureObjectStore>(config_.azure);
    }
    return *azure_;
  }
  if (scheme == "hdfs") {
    if (!hdfs_) {
      hdfs_ = std::make_unique<HdfsObjectStore>(config_.hdfs);
    }
    return *hdfs_;
  }
  throw StorageError(
      fmt::format("unsupported URI scheme '{}' (expected a local path, or 's3://', 'gs://'/'gcs://', "
                  "'abfs://'/'abfss://'/'az://', or 'hdfs://')",
                  scheme));
}

}  // namespace kernellake
