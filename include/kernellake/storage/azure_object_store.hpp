#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Azure Blob-backed ObjectStore ("abfs://container/key"), wrapping
// arrow::fs::AzureFileSystem. Same list()/open() contract as
// LocalObjectStore.
class AzureObjectStore final : public ObjectStore {
 public:
  explicit AzureObjectStore(const AzureSection& config);

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

}  // namespace kernellake
