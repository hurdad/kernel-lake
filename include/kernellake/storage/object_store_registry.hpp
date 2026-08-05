#pragma once

#include <memory>

#include "kernellake/common/config.hpp"
#include "kernellake/storage/local_object_store.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// A scheme-dispatching ObjectStore: owns a LocalObjectStore plus lazily-
// constructed S3/GCS/Azure/HdfsObjectStore backends, keyed by Uri::scheme(),
// and implements ObjectStore itself so it's a drop-in replacement for
// LocalObjectStore at every existing call site (QueryEngine's `store_`
// member, the CLI's standalone uses). A backend is constructed on the first
// call whose Uri names its scheme; construction failures (bad credentials,
// unreachable endpoint, etc.) surface as StorageError from that call, not
// eagerly at ObjectStoreRegistry construction time.
class ObjectStoreRegistry final : public ObjectStore {
 public:
  explicit ObjectStoreRegistry(const StorageSection& config) : config_(config), local_(config_.local_root) {}

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override {
    return backend_for(prefix.scheme()).list(prefix);
  }

  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override {
    return backend_for(prefix.scheme()).list_recursive(prefix);
  }

  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override {
    return backend_for(uri.scheme()).open(uri);
  }

 private:
  [[nodiscard]] ObjectStore& backend_for(std::string_view scheme);

  const StorageSection& config_;
  LocalObjectStore local_;
  std::unique_ptr<ObjectStore> s3_;
  std::unique_ptr<ObjectStore> gcs_;
  std::unique_ptr<ObjectStore> azure_;
  std::unique_ptr<ObjectStore> hdfs_;
};

}  // namespace kernellake
