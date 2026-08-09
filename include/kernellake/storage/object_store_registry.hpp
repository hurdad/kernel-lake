#pragma once

#include <memory>
#include <optional>

#include "kernellake/common/config.hpp"
#include "kernellake/storage/local_object_store.hpp"
#include "kernellake/storage/nvme_object_cache.hpp"
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
//
// When config.cache.enabled, open() for any *non*-"file" scheme is routed
// through an owned NvmeObjectCache instead of the raw backend -- see that
// class's own comment and docs/ARCHITECTURE.md's "NVMe cache tier" section.
//
// list() also consults the cache first, for one specific, narrow reason
// found by a real end-to-end test (see docs/ARCHITECTURE.md): every
// read_parquet(...) resolves its source through
// discover_parquet_files[_recursive]() (src/storage/file_discovery.cpp)
// *before* any open() call happens, so even a fully-cached repeat query
// would still require the backend reachable just to re-discover a file it
// already has a local copy of -- entirely defeating the point for a
// backend that's since gone offline. NvmeObjectCache::cached_info() only
// ever matches a URI string previously passed to open() (i.e. a real,
// exact, non-glob, non-directory source) -- a glob pattern or directory
// prefix is never cached under its own string, so falls through to the
// real backend exactly as before. list_recursive() is deliberately left
// as a live-only call: it's how a directory's file *set* is discovered in
// the first place, which a same-string cache lookup fundamentally can't
// answer (new files could always have appeared) -- its own StorageError
// catch-and-retry-via-list() fallback in discover_parquet_files_recursive()
// is what reaches this list() cache check for the common single-explicit-
// file source case anyway.
class ObjectStoreRegistry final : public ObjectStore {
 public:
  explicit ObjectStoreRegistry(const StorageSection& config) : config_(config), local_(config_.local_root) {
    if (config_.cache.enabled) {
      cache_.emplace(config_.cache);
    }
  }

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override {
    if (cache_.has_value() && prefix.scheme() != "file") {
      if (const std::optional<ObjectInfo> cached = cache_->cached_info(prefix)) {
        return {*cached};
      }
    }
    return backend_for(prefix.scheme()).list(prefix);
  }

  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override {
    return backend_for(prefix.scheme()).list_recursive(prefix);
  }

  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override {
    const std::string_view scheme = uri.scheme();
    ObjectStore& backend = backend_for(scheme);
    if (cache_.has_value() && scheme != "file") {
      return cache_->get_or_populate(uri, backend);
    }
    return backend.open(uri);
  }

  // nullopt when storage.cache.enabled is false -- see NvmeObjectCache::
  // snapshot() for field semantics. Safe to call at any time, including
  // concurrently with in-flight queries.
  [[nodiscard]] std::optional<NvmeCacheMetricsSnapshot> cache_metrics() const {
    if (!cache_.has_value()) {
      return std::nullopt;
    }
    return cache_->snapshot();
  }

 private:
  [[nodiscard]] ObjectStore& backend_for(std::string_view scheme);

  const StorageSection& config_;
  LocalObjectStore local_;
  std::unique_ptr<ObjectStore> s3_;
  std::unique_ptr<ObjectStore> gcs_;
  std::unique_ptr<ObjectStore> azure_;
  std::unique_ptr<ObjectStore> hdfs_;
  std::optional<NvmeObjectCache> cache_;
};

}  // namespace kernellake
