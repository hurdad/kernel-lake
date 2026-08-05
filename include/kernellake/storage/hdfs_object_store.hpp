#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// HDFS-backed ObjectStore ("hdfs://namenode:port/path"), wrapping
// arrow::fs::HadoopFileSystem. Same list()/open() contract as
// LocalObjectStore.
//
// Unlike S3ObjectStore/GcsObjectStore/AzureObjectStore, this is compiled
// and verified to link/run against a *simulated* absence of libhdfs (see
// docs/ARCHITECTURE.md's "Cloud object storage" section) but not against
// a real Hadoop cluster: arrow::fs::HadoopFileSystem dlopen()s libhdfs.so
// lazily at runtime (a standard Arrow design, not a build-time link
// dependency of libarrow itself), so this class compiles and links
// cleanly with no Hadoop installed at all -- it will only fail, with a
// clear StorageError, the first time something actually tries to use it
// without a real libhdfs.so/JAVA_HOME/CLASSPATH environment present.
class HdfsObjectStore final : public ObjectStore {
 public:
  explicit HdfsObjectStore(const HdfsSection& config);

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

}  // namespace kernellake
