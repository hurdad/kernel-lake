#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "kernellake/common/config.hpp"
#include "kernellake/storage/local_object_store.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Point-in-time read of one NvmeObjectCache's counters (NvmeObjectCache::
// snapshot()). hits/misses/evictions are cumulative since this cache
// instance was constructed (which, for kernellake-server, means since
// process startup -- see docs/ARCHITECTURE.md); current_bytes/
// current_entries are live gauges, not cumulative. A "hit" counts both a
// get_or_populate() call that found an existing entry and a successful
// cached_info() lookup (both represent a real backend call avoided); a
// "miss" only counts get_or_populate() calls that had to populate a new
// entry -- cached_info() failing is not counted as a miss, since nothing
// upstream of it treats that as a completed cache decision the way a
// get_or_populate() miss is (see cached_info()'s own comment).
struct NvmeCacheMetricsSnapshot {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t evictions = 0;
  std::uint64_t current_bytes = 0;
  std::uint64_t current_entries = 0;
};

// A local-NVMe, whole-object, read-through cache sitting in front of a
// remote ObjectStore backend (S3/GCS/Azure/HDFS) -- never used for "file"
// scheme URIs, since those are already local. On a cache miss, the entire
// remote object is streamed to `directory` once and served from there for
// every later open() of the same URI (repeat queries against overlapping
// data become a local disk read instead of a network fetch) -- see
// docs/ARCHITECTURE.md's "NVMe cache tier" section for the full design
// rationale and docs/ROADMAP.md's matching entry for why whole-object
// (rather than byte-range) caching was chosen.
//
// The cache key is derived from the URI string alone, not re-verified
// against the backend on every open() -- deliberately, on the same
// write-once assumption this project's own TPC-H/benchmark tooling already
// relies on for Parquet files in object storage (see the class comment on
// docs/ROADMAP.md's "NVMe cache tier" entry). This means a cache *hit*
// makes zero calls to the remote backend, not even a metadata HEAD -- that
// is the whole point, since even a metadata-only round trip would undercut
// what this class exists to avoid. An overwrite-in-place workload is not
// supported by this cache (nor, today, by anything else in this project).
//
// Safe to call from multiple threads concurrently (kernellake-server
// serves concurrent queries): per-cache-key locking serializes population
// of the *same* object without blocking unrelated cache misses/hits.
class NvmeObjectCache {
 public:
  explicit NvmeObjectCache(CacheSection config);

  // `backend` is whichever ObjectStore actually owns `uri` (already
  // resolved by the caller, e.g. ObjectStoreRegistry::backend_for()) --
  // this class does no scheme dispatch of its own. `backend` is only ever
  // called on a cache miss.
  [[nodiscard]] std::unique_ptr<RandomAccessObject> get_or_populate(const Uri& uri, ObjectStore& backend);

  // A pure local stat, no network call: returns this URI's cached size if
  // (and only if) it's already cached under its *exact* string -- lets a
  // caller resolving file metadata (ObjectStoreRegistry::list(), used by
  // read_parquet(...)'s file-discovery path before any open() happens)
  // skip the backend entirely for an already-cached, non-glob source. A
  // glob or directory prefix never matches here, since nothing ever
  // populates the cache under a glob/directory string -- only an exact
  // single-object URI, from get_or_populate() -- so no separate glob
  // detection is needed to make this safe.
  [[nodiscard]] std::optional<ObjectInfo> cached_info(const Uri& uri) const;

  // Safe to call from any thread at any time, including concurrently with
  // get_or_populate() -- individual fields may not represent one atomic
  // instant together, the same tradeoff GpuMemoryMetricsRegistry::
  // snapshot() makes for its own counters.
  [[nodiscard]] NvmeCacheMetricsSnapshot snapshot() const noexcept;

 private:
  [[nodiscard]] std::string cache_file_name(const Uri& uri) const;
  void populate(const std::string& cache_path, RandomAccessObject& remote, std::uint64_t size_bytes);
  // `protected_key` (the cache file name -- see cache_file_name() -- this
  // populate()/get_or_populate() call is currently serving) is never
  // evicted, even if it's the oldest/only candidate and the cache is over
  // budget without it: the caller is about to open exactly this file, and
  // evicting the entry a call is in the middle of serving defeats the
  // point of populating it in the first place (see this method's own
  // comment for the real bug that shipped before this parameter existed).
  void evict_if_over_budget(const std::string& protected_key);
  [[nodiscard]] std::shared_ptr<std::recursive_mutex> lock_for(const std::string& cache_key);
  // One-time scan of any cache entries already on disk at construction
  // time (a warm restart against a directory a previous process
  // populated) -- seeds current_bytes_/current_entries_ so they reflect
  // reality immediately, not just growth since this instance started.
  void seed_metrics_from_existing_directory();

  CacheSection config_;
  LocalObjectStore cache_store_;
  std::mutex keys_mutex_;
  // recursive_mutex, not plain mutex: evict_if_over_budget() (see its own
  // comment) try_locks each candidate entry's own key lock before removing
  // it, including -- when populate() just ran over budget -- the very key
  // get_or_populate() is already holding this lock for on the same thread's
  // call stack; try_lock() on a plain mutex already held by this thread
  // would report "busy" and wrongly skip evicting its own just-written
  // entry.
  std::unordered_map<std::string, std::shared_ptr<std::recursive_mutex>> key_locks_;
  // Serializes evict_if_over_budget() sweeps against each other (only, not
  // against get_or_populate()/populate() -- see that method's own comment
  // for why it uses try_lock, never a blocking lock, on each candidate
  // entry's own key lock instead of taking this mutex plus that one
  // together, which is exactly the lock-order pattern that can deadlock
  // two threads evicting each other's in-progress key).
  std::mutex eviction_mutex_;

  // mutable: cached_info() is logically read-only (const) but still counts
  // toward the same hit counter get_or_populate() uses.
  mutable std::atomic<std::uint64_t> hits_{0};
  std::atomic<std::uint64_t> misses_{0};
  std::atomic<std::uint64_t> evictions_{0};
  std::atomic<std::uint64_t> current_bytes_{0};
  std::atomic<std::uint64_t> current_entries_{0};
};

}  // namespace kernellake
