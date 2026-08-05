#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// S3-backed ObjectStore ("s3://bucket/key"), wrapping arrow::fs::S3FileSystem.
// Same list()/open() contract as LocalObjectStore (glob-in-final-component,
// directory listing, StorageError on anything missing -- never a silent
// empty result). Constructing this calls arrow::fs::EnsureS3Initialized()
// (idempotent, AWS SDK global init) the first time; callers only need to
// make sure arrow::fs::FinalizeS3() runs once at process shutdown (see
// src/cli/main.cpp / src/server/main.cpp).
class S3ObjectStore final : public ObjectStore {
 public:
  explicit S3ObjectStore(const S3Section& config);

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

}  // namespace kernellake
