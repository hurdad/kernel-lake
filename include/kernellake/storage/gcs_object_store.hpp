#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// GCS-backed ObjectStore ("gs://bucket/key"), wrapping arrow::fs::GcsFileSystem.
// Same list()/open() contract as LocalObjectStore.
class GcsObjectStore final : public ObjectStore {
 public:
  explicit GcsObjectStore(const GcsSection& config);

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

}  // namespace kernellake
