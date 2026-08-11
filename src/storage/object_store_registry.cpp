#include "kernellake/storage/object_store_registry.hpp"

#include <fmt/format.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/azure_object_store.hpp"
#include "kernellake/storage/gcs_object_store.hpp"
#include "kernellake/storage/hdfs_object_store.hpp"
#include "kernellake/storage/s3_object_store.hpp"

namespace kernellake {

// Each branch's check-then-construct used to be a plain `if (!s3_) { s3_ =
// ...; }` with no synchronization -- a real data race under
// kernellake-server's concurrent Flight SQL RPCs (see the *_once_ members'
// own comment in object_store_registry.hpp): two threads racing to touch
// the same not-yet-constructed backend for the first time could both
// observe the null unique_ptr and both construct+assign, corrupting it.
// std::call_once gives each backend exactly-once construction with a
// proper happens-before guarantee for every caller (including ones that
// don't run the constructing call themselves), without needing a lock held
// across the backend's actual (potentially slow, network-bound) I/O
// afterward.
ObjectStore& ObjectStoreRegistry::backend_for(std::string_view scheme) {
  if (scheme == "file") {
    return local_;
  }
  if (scheme == "s3") {
    std::call_once(s3_once_, [this] { s3_ = std::make_unique<S3ObjectStore>(config_.s3); });
    return *s3_;
  }
  if (scheme == "gs" || scheme == "gcs") {
    std::call_once(gcs_once_, [this] { gcs_ = std::make_unique<GcsObjectStore>(config_.gcs); });
    return *gcs_;
  }
  if (scheme == "abfs" || scheme == "abfss" || scheme == "az") {
    std::call_once(azure_once_, [this] { azure_ = std::make_unique<AzureObjectStore>(config_.azure); });
    return *azure_;
  }
  if (scheme == "hdfs") {
    std::call_once(hdfs_once_, [this] { hdfs_ = std::make_unique<HdfsObjectStore>(config_.hdfs); });
    return *hdfs_;
  }
  throw StorageError(
      fmt::format("unsupported URI scheme '{}' (expected a local path, or 's3://', 'gs://'/'gcs://', "
                  "'abfs://'/'abfss://'/'az://', or 'hdfs://')",
                  scheme));
}

}  // namespace kernellake
