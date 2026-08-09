#include "kernellake/storage/nvme_object_cache.hpp"

#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <string_view>
#include <vector>

#include <unistd.h>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

namespace fs = std::filesystem;

// FNV-1a 64-bit -- deterministic across processes and builds (unlike
// std::hash<std::string>, whose exact algorithm is implementation-defined),
// which matters here since the cache directory is meant to survive process
// restarts: a hash that changed between runs would silently orphan every
// previously-cached entry.
std::uint64_t fnv1a_64(std::string_view data) {
  constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::uint64_t hash = kOffsetBasis;
  for (const unsigned char byte : data) {
    hash ^= byte;
    hash *= kPrime;
  }
  return hash;
}

// Distinct extension from the ".tmp-<pid>-<n>" suffix populate() writes
// under, so evict_if_over_budget()'s directory scan never counts (or
// deletes) a temp file that a concurrent populate() is still writing.
constexpr std::string_view kCacheFileExtension = ".cache";

}  // namespace

NvmeObjectCache::NvmeObjectCache(CacheSection config) : config_(std::move(config)), cache_store_(config_.directory) {
  seed_metrics_from_existing_directory();
}

std::string NvmeObjectCache::cache_file_name(const Uri& uri) const {
  return fmt::format("{:016x}{}", fnv1a_64(uri.value()), kCacheFileExtension);
}

std::shared_ptr<std::mutex> NvmeObjectCache::lock_for(const std::string& cache_key) {
  std::lock_guard<std::mutex> guard(keys_mutex_);
  std::shared_ptr<std::mutex>& slot = key_locks_[cache_key];
  if (!slot) {
    slot = std::make_shared<std::mutex>();
  }
  return slot;
}

std::unique_ptr<RandomAccessObject> NvmeObjectCache::get_or_populate(const Uri& uri, ObjectStore& backend) {
  const std::string file_name = cache_file_name(uri);
  const fs::path cache_path = fs::path(config_.directory) / file_name;

  // Serializes population of *this* key only -- an unrelated key's miss
  // proceeds concurrently. key_locks_ grows by one small entry per distinct
  // object ever cached and is never pruned; acceptable for the working-set
  // sizes this cache targets (see docs/ARCHITECTURE.md).
  const std::shared_ptr<std::mutex> key_lock = lock_for(file_name);
  const std::lock_guard<std::mutex> populate_guard(*key_lock);

  std::error_code exists_ec;
  if (!fs::exists(cache_path, exists_ec)) {
    misses_.fetch_add(1, std::memory_order_relaxed);
    const std::unique_ptr<RandomAccessObject> remote = backend.open(uri);
    populate(cache_path.string(), *remote, remote->size());
  } else {
    hits_.fetch_add(1, std::memory_order_relaxed);
  }

  // Bumps mtime as an access-time proxy for LRU eviction below -- POSIX
  // atime updates aren't guaranteed on read() (many real deployments mount
  // with noatime), so mtime-on-touch is the standard workaround.
  std::error_code touch_ec;
  fs::last_write_time(cache_path, fs::file_time_type::clock::now(), touch_ec);

  return cache_store_.open(Uri(cache_path.string()));
}

std::optional<ObjectInfo> NvmeObjectCache::cached_info(const Uri& uri) const {
  const fs::path cache_path = fs::path(config_.directory) / cache_file_name(uri);
  std::error_code size_ec;
  const std::uintmax_t size = fs::file_size(cache_path, size_ec);
  if (size_ec) {
    return std::nullopt;  // Not cached (or a concurrent evict raced us) -- an ordinary miss, not an error.
  }
  hits_.fetch_add(1, std::memory_order_relaxed);
  return ObjectInfo{uri, static_cast<std::uint64_t>(size)};
}

NvmeCacheMetricsSnapshot NvmeObjectCache::snapshot() const noexcept {
  return NvmeCacheMetricsSnapshot{
      .hits = hits_.load(std::memory_order_relaxed),
      .misses = misses_.load(std::memory_order_relaxed),
      .evictions = evictions_.load(std::memory_order_relaxed),
      .current_bytes = current_bytes_.load(std::memory_order_relaxed),
      .current_entries = current_entries_.load(std::memory_order_relaxed),
  };
}

void NvmeObjectCache::seed_metrics_from_existing_directory() {
  std::uint64_t bytes = 0;
  std::uint64_t count = 0;
  std::error_code iter_ec;
  for (const fs::directory_entry& entry :
       fs::directory_iterator(config_.directory, fs::directory_options::skip_permission_denied, iter_ec)) {
    if (!entry.is_regular_file() || entry.path().extension() != kCacheFileExtension) {
      continue;
    }
    std::error_code size_ec;
    const std::uintmax_t size = entry.file_size(size_ec);
    if (size_ec) {
      continue;
    }
    bytes += size;
    ++count;
  }
  current_bytes_.store(bytes, std::memory_order_relaxed);
  current_entries_.store(count, std::memory_order_relaxed);
}

void NvmeObjectCache::populate(const std::string& cache_path, RandomAccessObject& remote,
                               std::uint64_t size_bytes) {
  std::error_code mkdir_ec;
  fs::create_directories(config_.directory, mkdir_ec);

  static std::atomic<std::uint64_t> temp_counter{0};
  const std::string tmp_path =
      fmt::format("{}.tmp-{}-{}", cache_path, static_cast<long>(::getpid()), temp_counter.fetch_add(1));

  const arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> out_result =
      arrow::io::FileOutputStream::Open(tmp_path);
  if (!out_result.ok()) {
    throw StorageError(
        fmt::format("failed to open NVMe cache temp file '{}': {}", tmp_path, out_result.status().ToString()));
  }
  const std::shared_ptr<arrow::io::FileOutputStream>& out = *out_result;
  const std::shared_ptr<arrow::io::RandomAccessFile> arrow_remote = remote.as_arrow_file();

  constexpr std::int64_t kChunkBytes = 64LL * 1024 * 1024;
  const std::int64_t total = static_cast<std::int64_t>(size_bytes);
  std::int64_t offset = 0;
  while (offset < total) {
    const std::int64_t want = std::min<std::int64_t>(kChunkBytes, total - offset);
    const arrow::Result<std::shared_ptr<arrow::Buffer>> chunk = arrow_remote->ReadAt(offset, want);
    if (!chunk.ok()) {
      static_cast<void>(out->Close());
      fs::remove(tmp_path);
      throw StorageError(fmt::format("failed to read {} bytes at offset {} while populating NVMe cache: {}", want,
                                     offset, chunk.status().ToString()));
    }
    const std::shared_ptr<arrow::Buffer>& buffer = *chunk;
    if (buffer->size() == 0) {
      break;  // Short read: the remote object is smaller than size_bytes claimed.
    }
    const arrow::Status write_status = out->Write(buffer->data(), buffer->size());
    if (!write_status.ok()) {
      static_cast<void>(out->Close());
      fs::remove(tmp_path);
      throw StorageError(
          fmt::format("failed to write NVMe cache temp file '{}': {}", tmp_path, write_status.ToString()));
    }
    offset += buffer->size();
  }

  const arrow::Status close_status = out->Close();
  if (!close_status.ok()) {
    fs::remove(tmp_path);
    throw StorageError(
        fmt::format("failed to close NVMe cache temp file '{}': {}", tmp_path, close_status.ToString()));
  }

  // Atomic on POSIX (same filesystem, since tmp_path and cache_path share a
  // directory): a concurrent reader can never observe a partially-written
  // cache_path.
  std::error_code rename_ec;
  fs::rename(tmp_path, cache_path, rename_ec);
  if (rename_ec) {
    fs::remove(tmp_path);
    throw StorageError(
        fmt::format("failed to finalize NVMe cache entry '{}': {}", cache_path, rename_ec.message()));
  }

  // `offset`, the actual bytes written, not the caller-claimed size_bytes:
  // they differ on the short-read path above, and these counters should
  // reflect what's really on disk.
  current_bytes_.fetch_add(static_cast<std::uint64_t>(offset), std::memory_order_relaxed);
  current_entries_.fetch_add(1, std::memory_order_relaxed);

  evict_if_over_budget();
}

void NvmeObjectCache::evict_if_over_budget() {
  if (config_.max_size_bytes == 0) {
    return;  // 0 == unbounded, matches this project's existing convention.
  }
  // Common case: current_bytes_ (maintained incrementally by populate()/
  // eviction itself, seeded from disk at construction) is already known to
  // be under budget, so no directory walk is needed at all. This can drift
  // from the real on-disk total if something outside this class ever
  // touches the cache directory directly -- an accepted, undefended
  // scenario, same as this project's general stance of trusting its own
  // invariants rather than defending against out-of-band interference.
  if (current_bytes_.load(std::memory_order_relaxed) <= config_.max_size_bytes) {
    return;
  }

  struct Entry {
    fs::path path;
    fs::file_time_type mtime;
    std::uintmax_t size = 0;
  };
  std::vector<Entry> entries;
  std::uintmax_t total = 0;

  std::error_code iter_ec;
  for (const fs::directory_entry& entry :
       fs::directory_iterator(config_.directory, fs::directory_options::skip_permission_denied, iter_ec)) {
    if (!entry.is_regular_file() || entry.path().extension() != kCacheFileExtension) {
      continue;  // Skips ".tmp-*" files a concurrent populate() may still be writing.
    }
    std::error_code size_ec;
    const std::uintmax_t size = entry.file_size(size_ec);
    if (size_ec) {
      continue;
    }
    std::error_code mtime_ec;
    const fs::file_time_type mtime = fs::last_write_time(entry.path(), mtime_ec);
    entries.push_back(Entry{entry.path(), mtime, size});
    total += size;
  }

  if (total <= config_.max_size_bytes) {
    return;
  }

  std::sort(entries.begin(), entries.end(),
           [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
  // Unlinking a file a concurrent reader still has open is safe on POSIX:
  // the inode stays valid (and readable) until every open file descriptor
  // against it is closed, only the directory entry disappears -- an
  // in-flight query reading an evicted entry is never disrupted.
  for (const Entry& entry : entries) {
    if (total <= config_.max_size_bytes) {
      break;
    }
    std::error_code remove_ec;
    if (fs::remove(entry.path, remove_ec) && !remove_ec) {
      total -= entry.size;
      current_bytes_.fetch_sub(entry.size, std::memory_order_relaxed);
      current_entries_.fetch_sub(1, std::memory_order_relaxed);
      evictions_.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

}  // namespace kernellake
