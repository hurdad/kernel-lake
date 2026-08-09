#include "kernellake/storage/nvme_object_cache.hpp"

#include <arrow/buffer.h>
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

// Wraps a real ObjectStore (a LocalObjectStore rooted at a scratch "remote"
// directory, standing in for a real S3/GCS/Azure backend) and counts open()
// calls -- lets tests assert exactly how many times the "remote" was
// actually touched, the thing this whole cache exists to minimize.
class CountingObjectStore final : public ObjectStore {
 public:
  explicit CountingObjectStore(ObjectStore& delegate) : delegate_(delegate) {}

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override { return delegate_.list(prefix); }
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override {
    return delegate_.list_recursive(prefix);
  }
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override {
    ++open_count;
    return delegate_.open(uri);
  }

  std::atomic<int> open_count{0};

 private:
  ObjectStore& delegate_;
};

// Always throws -- used to prove a code path never actually reaches the
// backend (a cache hit should make zero calls to it).
class ThrowingObjectStore final : public ObjectStore {
 public:
  [[nodiscard]] std::vector<ObjectInfo> list(const Uri&) override {
    throw StorageError("ThrowingObjectStore::list should not be called");
  }
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri&) override {
    throw StorageError("ThrowingObjectStore::list_recursive should not be called");
  }
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri&) override {
    throw StorageError("ThrowingObjectStore::open should not be called");
  }
};

std::string read_all(RandomAccessObject& object) {
  const arrow::Result<std::shared_ptr<arrow::Buffer>> result =
      object.as_arrow_file()->ReadAt(0, static_cast<std::int64_t>(object.size()));
  return std::string(reinterpret_cast<const char*>(result.ValueOrDie()->data()),
                     static_cast<std::size_t>(result.ValueOrDie()->size()));
}

std::size_t count_cache_files(const fs::path& dir) {
  std::size_t count = 0;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".cache") {
      ++count;
    }
  }
  return count;
}

class NvmeObjectCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string unique = std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                               ::testing::UnitTest::GetInstance()->current_test_info()->name();
    remote_dir_ = fs::temp_directory_path() / fs::path("kernellake_nvme_cache_remote_" + unique);
    cache_dir_ = fs::temp_directory_path() / fs::path("kernellake_nvme_cache_dir_" + unique);
    fs::create_directories(remote_dir_);
    // Deliberately not pre-creating cache_dir_ -- populate() must create it
    // itself on first use.
  }

  void TearDown() override {
    fs::remove_all(remote_dir_);
    fs::remove_all(cache_dir_);
  }

  void write_remote_file(const std::string& name, const std::string& content) const {
    std::ofstream out(remote_dir_ / name, std::ios::binary);
    out << content;
  }

  [[nodiscard]] CacheSection cache_config(std::uint64_t max_size_bytes = 0) const {
    CacheSection config;
    config.enabled = true;
    config.directory = cache_dir_.string();
    config.max_size_bytes = max_size_bytes;
    return config;
  }

  fs::path remote_dir_;
  fs::path cache_dir_;
};

TEST_F(NvmeObjectCacheTest, MissPopulatesCacheAndReturnsCorrectContent) {
  write_remote_file("data.parquet", "hello world");
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());

  const Uri uri((remote_dir_ / "data.parquet").string());
  const std::unique_ptr<RandomAccessObject> object = cache.get_or_populate(uri, remote_store);

  EXPECT_EQ(read_all(*object), "hello world");
  EXPECT_EQ(count_cache_files(cache_dir_), 1u);
}

// Regression test for a real bug found via a manual end-to-end smoke test
// against MinIO (see docs/ARCHITECTURE.md's "NVMe cache tier" section):
// read_parquet(...)'s file-discovery path calls ObjectStore::list() before
// any open(), so without this, a fully-cached repeat query still required
// the backend reachable just to re-resolve a file it already had locally
// -- confirmed for real by stopping the MinIO container and watching an
// otherwise-cached query fail at the listing step. cached_info() is what
// lets ObjectStoreRegistry::list() answer that without touching the
// backend at all.
TEST_F(NvmeObjectCacheTest, CachedInfoIsNulloptBeforePopulationAndCorrectSizeAfter) {
  write_remote_file("data.parquet", "hello world");
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());
  const Uri uri((remote_dir_ / "data.parquet").string());

  EXPECT_FALSE(cache.cached_info(uri).has_value());

  static_cast<void>(cache.get_or_populate(uri, remote_store));

  const std::optional<ObjectInfo> info = cache.cached_info(uri);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->uri.value(), uri.value());
  EXPECT_EQ(info->size_bytes, 11u);  // strlen("hello world")
}

TEST_F(NvmeObjectCacheTest, CachedInfoIsNulloptForADifferentUnrelatedUri) {
  write_remote_file("data.parquet", "hello world");
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "data.parquet").string()), remote_store));

  // A glob pattern or an entirely different path was never passed to
  // get_or_populate(), so it must never spuriously match a cached entry.
  EXPECT_FALSE(cache.cached_info(Uri((remote_dir_ / "*.parquet").string())).has_value());
  EXPECT_FALSE(cache.cached_info(Uri((remote_dir_ / "other.parquet").string())).has_value());
}

TEST_F(NvmeObjectCacheTest, HitMakesZeroCallsToBackend) {
  write_remote_file("data.parquet", "hello world");
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());
  const Uri uri((remote_dir_ / "data.parquet").string());

  {
    const std::unique_ptr<RandomAccessObject> first = cache.get_or_populate(uri, remote_store);
    EXPECT_EQ(read_all(*first), "hello world");
  }

  // A second get_or_populate() for the same URI, backed by a store that
  // throws on any call -- must succeed and return correct content purely
  // from the local cache, proving the backend was never touched.
  ThrowingObjectStore throwing_store;
  const std::unique_ptr<RandomAccessObject> second = cache.get_or_populate(uri, throwing_store);
  EXPECT_EQ(read_all(*second), "hello world");
}

TEST_F(NvmeObjectCacheTest, DifferentUrisGetDistinctCacheEntries) {
  write_remote_file("a.parquet", "aaa");
  write_remote_file("b.parquet", "bbbbb");
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());

  const std::unique_ptr<RandomAccessObject> a =
      cache.get_or_populate(Uri((remote_dir_ / "a.parquet").string()), remote_store);
  const std::unique_ptr<RandomAccessObject> b =
      cache.get_or_populate(Uri((remote_dir_ / "b.parquet").string()), remote_store);

  EXPECT_EQ(read_all(*a), "aaa");
  EXPECT_EQ(read_all(*b), "bbbbb");
  EXPECT_EQ(count_cache_files(cache_dir_), 2u);
}

TEST_F(NvmeObjectCacheTest, ConcurrentPopulateOfSameKeyFetchesBackendExactlyOnce) {
  write_remote_file("data.parquet", std::string(1024, 'x'));
  LocalObjectStore local_store(remote_dir_.string());
  CountingObjectStore counting_store(local_store);
  NvmeObjectCache cache(cache_config());
  const Uri uri((remote_dir_ / "data.parquet").string());

  constexpr int kThreads = 8;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&] {
      const std::unique_ptr<RandomAccessObject> object = cache.get_or_populate(uri, counting_store);
      EXPECT_EQ(object->size(), 1024u);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(counting_store.open_count.load(), 1);
  EXPECT_EQ(count_cache_files(cache_dir_), 1u);
}

TEST_F(NvmeObjectCacheTest, EvictsLeastRecentlyUsedEntryWhenOverBudget) {
  write_remote_file("a.parquet", std::string(100, 'a'));
  write_remote_file("b.parquet", std::string(100, 'b'));
  LocalObjectStore remote_store(remote_dir_.string());
  // Budget fits exactly one 100-byte entry -- populating a second must
  // evict the first.
  NvmeObjectCache cache(cache_config(/*max_size_bytes=*/100));

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "a.parquet").string()), remote_store));
  ASSERT_EQ(count_cache_files(cache_dir_), 1u);

  // Back-date the first entry's mtime so recency ordering is deterministic
  // regardless of filesystem mtime resolution, rather than relying on a
  // real-time sleep between the two populate() calls below.
  for (const auto& entry : fs::directory_iterator(cache_dir_)) {
    fs::last_write_time(entry.path(), fs::file_time_type::clock::now() - std::chrono::hours(1));
  }

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "b.parquet").string()), remote_store));

  EXPECT_EQ(count_cache_files(cache_dir_), 1u);
  std::uintmax_t total = 0;
  for (const auto& entry : fs::directory_iterator(cache_dir_)) {
    total += entry.file_size();
  }
  EXPECT_LE(total, 100u);

  // The survivor must be "b" (more recently populated), "a" must be gone.
  ThrowingObjectStore throwing_store;
  EXPECT_NO_THROW((void)(cache.get_or_populate(Uri((remote_dir_ / "b.parquet").string()), throwing_store)));
}

TEST_F(NvmeObjectCacheTest, ZeroMaxSizeBytesMeansUnbounded) {
  write_remote_file("a.parquet", std::string(1000, 'a'));
  write_remote_file("b.parquet", std::string(1000, 'b'));
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config(/*max_size_bytes=*/0));

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "a.parquet").string()), remote_store));
  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "b.parquet").string()), remote_store));

  EXPECT_EQ(count_cache_files(cache_dir_), 2u);
}

TEST_F(NvmeObjectCacheTest, SnapshotStartsAtAllZeros) {
  NvmeObjectCache cache(cache_config());
  const NvmeCacheMetricsSnapshot snapshot = cache.snapshot();
  EXPECT_EQ(snapshot.hits, 0u);
  EXPECT_EQ(snapshot.misses, 0u);
  EXPECT_EQ(snapshot.evictions, 0u);
  EXPECT_EQ(snapshot.current_bytes, 0u);
  EXPECT_EQ(snapshot.current_entries, 0u);
}

TEST_F(NvmeObjectCacheTest, SnapshotTracksMissThenHitsFromGetOrPopulate) {
  write_remote_file("data.parquet", "hello world");
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());
  const Uri uri((remote_dir_ / "data.parquet").string());

  static_cast<void>(cache.get_or_populate(uri, remote_store));
  EXPECT_EQ(cache.snapshot().misses, 1u);
  EXPECT_EQ(cache.snapshot().hits, 0u);

  static_cast<void>(cache.get_or_populate(uri, remote_store));
  static_cast<void>(cache.get_or_populate(uri, remote_store));
  const NvmeCacheMetricsSnapshot snapshot = cache.snapshot();
  EXPECT_EQ(snapshot.misses, 1u);
  EXPECT_EQ(snapshot.hits, 2u);
}

TEST_F(NvmeObjectCacheTest, SnapshotCountsCachedInfoSuccessAsAHit) {
  write_remote_file("data.parquet", "hello world");
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());
  const Uri uri((remote_dir_ / "data.parquet").string());

  // A miss doesn't touch hits at all -- neither get_or_populate() nor a
  // failed cached_info() lookup before population counts as one.
  EXPECT_FALSE(cache.cached_info(uri).has_value());
  EXPECT_EQ(cache.snapshot().hits, 0u);

  static_cast<void>(cache.get_or_populate(uri, remote_store));
  EXPECT_EQ(cache.snapshot().hits, 0u);  // the populating call itself is a miss, not a hit

  ASSERT_TRUE(cache.cached_info(uri).has_value());
  EXPECT_EQ(cache.snapshot().hits, 1u);
}

TEST_F(NvmeObjectCacheTest, SnapshotTracksCurrentBytesAndEntriesAcrossMultipleObjects) {
  write_remote_file("a.parquet", std::string(100, 'a'));
  write_remote_file("b.parquet", std::string(250, 'b'));
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config());

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "a.parquet").string()), remote_store));
  const NvmeCacheMetricsSnapshot after_first = cache.snapshot();
  EXPECT_EQ(after_first.current_bytes, 100u);
  EXPECT_EQ(after_first.current_entries, 1u);

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "b.parquet").string()), remote_store));
  const NvmeCacheMetricsSnapshot after_second = cache.snapshot();
  EXPECT_EQ(after_second.current_bytes, 350u);
  EXPECT_EQ(after_second.current_entries, 2u);
}

TEST_F(NvmeObjectCacheTest, SnapshotTracksEvictionsAndShrinksCurrentBytes) {
  write_remote_file("a.parquet", std::string(100, 'a'));
  write_remote_file("b.parquet", std::string(100, 'b'));
  LocalObjectStore remote_store(remote_dir_.string());
  NvmeObjectCache cache(cache_config(/*max_size_bytes=*/100));

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "a.parquet").string()), remote_store));
  ASSERT_EQ(cache.snapshot().current_entries, 1u);

  for (const auto& entry : fs::directory_iterator(cache_dir_)) {
    fs::last_write_time(entry.path(), fs::file_time_type::clock::now() - std::chrono::hours(1));
  }

  static_cast<void>(cache.get_or_populate(Uri((remote_dir_ / "b.parquet").string()), remote_store));

  const NvmeCacheMetricsSnapshot snapshot = cache.snapshot();
  EXPECT_EQ(snapshot.evictions, 1u);
  EXPECT_EQ(snapshot.current_entries, 1u);
  EXPECT_EQ(snapshot.current_bytes, 100u);
}

// Cache metrics must reflect reality immediately after construction, not
// just growth since that particular NvmeObjectCache instance started --
// otherwise a kernellake-server restart against an already-populated cache
// directory would report current_bytes/current_entries as 0 despite real
// cached data already sitting on disk.
TEST_F(NvmeObjectCacheTest, ConstructorSeedsMetricsFromPreExistingCacheDirectory) {
  write_remote_file("a.parquet", std::string(100, 'a'));
  write_remote_file("b.parquet", std::string(250, 'b'));
  LocalObjectStore remote_store(remote_dir_.string());

  {
    NvmeObjectCache first_process_cache(cache_config());
    static_cast<void>(
        first_process_cache.get_or_populate(Uri((remote_dir_ / "a.parquet").string()), remote_store));
    static_cast<void>(
        first_process_cache.get_or_populate(Uri((remote_dir_ / "b.parquet").string()), remote_store));
  }
  ASSERT_EQ(count_cache_files(cache_dir_), 2u);

  // A fresh instance over the same directory, simulating a process restart.
  NvmeObjectCache second_process_cache(cache_config());
  const NvmeCacheMetricsSnapshot snapshot = second_process_cache.snapshot();
  EXPECT_EQ(snapshot.current_bytes, 350u);
  EXPECT_EQ(snapshot.current_entries, 2u);
  EXPECT_EQ(snapshot.hits, 0u);
  EXPECT_EQ(snapshot.misses, 0u);
}

}  // namespace
}  // namespace kernellake
