#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <thread>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/memory/gpu_memory_metrics.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

// GpuMemoryMetricsRegistry is a process-wide singleton, and every test in
// this binary (rmm_environment_test.cpp's tests included -- every
// RmmEnvironment now routes through TrackingMemoryResource, see
// rmm_environment.cpp) shares it. Tests here that exercise the Registry's
// own bookkeeping directly use a dedicated device_id per test (this binary
// never targets more than one real device, so any id other than 0 is
// otherwise untouched) to avoid cross-test interference; tests that go
// through a real RmmEnvironment (which always targets device_id 0, the
// EngineConfig default) compare before/after deltas instead of absolute
// values, since other tests sharing device_id 0 may have already
// contributed to its counters.

TEST(GpuMemoryMetricsRegistryTest, RecordAllocationIncreasesCurrentBytes) {
  constexpr int kDeviceId = 1;
  const std::uint64_t before = GpuMemoryMetricsRegistry::snapshot(kDeviceId).current_bytes;
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 1024);
  EXPECT_EQ(GpuMemoryMetricsRegistry::snapshot(kDeviceId).current_bytes, before + 1024);
}

TEST(GpuMemoryMetricsRegistryTest, RecordDeallocationDecreasesCurrentBytes) {
  constexpr int kDeviceId = 2;
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 4096);
  const std::uint64_t after_alloc = GpuMemoryMetricsRegistry::snapshot(kDeviceId).current_bytes;
  GpuMemoryMetricsRegistry::record_deallocation(kDeviceId, 4096);
  EXPECT_EQ(GpuMemoryMetricsRegistry::snapshot(kDeviceId).current_bytes, after_alloc - 4096);
}

TEST(GpuMemoryMetricsRegistryTest, MultipleAllocationsAccumulate) {
  constexpr int kDeviceId = 3;
  const GpuMemoryMetricsSnapshot before = GpuMemoryMetricsRegistry::snapshot(kDeviceId);
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 100);
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 200);
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 300);
  const GpuMemoryMetricsSnapshot after = GpuMemoryMetricsRegistry::snapshot(kDeviceId);
  EXPECT_EQ(after.current_bytes, before.current_bytes + 600);
  EXPECT_EQ(after.allocations, before.allocations + 3);
}

TEST(GpuMemoryMetricsRegistryTest, PeakStaysAtMaximumAfterDeallocation) {
  constexpr int kDeviceId = 4;
  const std::uint64_t base_peak = GpuMemoryMetricsRegistry::snapshot(kDeviceId).peak_bytes;
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 10'000);
  const std::uint64_t peak_after_alloc = GpuMemoryMetricsRegistry::snapshot(kDeviceId).peak_bytes;
  EXPECT_GE(peak_after_alloc, base_peak + 10'000);

  GpuMemoryMetricsRegistry::record_deallocation(kDeviceId, 10'000);
  const GpuMemoryMetricsSnapshot after_free = GpuMemoryMetricsRegistry::snapshot(kDeviceId);
  // The point of this test: peak does not drop back down just because
  // current usage did.
  EXPECT_EQ(after_free.peak_bytes, peak_after_alloc);
}

TEST(GpuMemoryMetricsRegistryTest, AllocationsCounterIncrements) {
  constexpr int kDeviceId = 5;
  const std::uint64_t before = GpuMemoryMetricsRegistry::snapshot(kDeviceId).allocations;
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 1);
  EXPECT_EQ(GpuMemoryMetricsRegistry::snapshot(kDeviceId).allocations, before + 1);
}

TEST(GpuMemoryMetricsRegistryTest, DeallocationsCounterIncrements) {
  constexpr int kDeviceId = 6;
  const std::uint64_t before = GpuMemoryMetricsRegistry::snapshot(kDeviceId).deallocations;
  GpuMemoryMetricsRegistry::record_deallocation(kDeviceId, 1);
  EXPECT_EQ(GpuMemoryMetricsRegistry::snapshot(kDeviceId).deallocations, before + 1);
}

// The exact example from this feature's own spec: allocating and freeing a
// 1 GiB buffer 10 times should show ~10 GiB cumulative allocated bytes even
// though current allocated memory is back to (whatever it was before this
// test ran, i.e. a net-zero delta).
TEST(GpuMemoryMetricsRegistryTest, AllocatedTotalBytesIsCumulativeAcrossFreedMemory) {
  constexpr int kDeviceId = 7;
  constexpr std::uint64_t kOneGiB = 1ULL * 1024 * 1024 * 1024;
  const GpuMemoryMetricsSnapshot before = GpuMemoryMetricsRegistry::snapshot(kDeviceId);

  for (int i = 0; i < 10; ++i) {
    GpuMemoryMetricsRegistry::record_allocation(kDeviceId, kOneGiB);
    GpuMemoryMetricsRegistry::record_deallocation(kDeviceId, kOneGiB);
  }

  const GpuMemoryMetricsSnapshot after = GpuMemoryMetricsRegistry::snapshot(kDeviceId);
  EXPECT_EQ(after.allocated_total_bytes, before.allocated_total_bytes + 10 * kOneGiB);
  EXPECT_EQ(after.current_bytes, before.current_bytes);  // net zero: every allocation was freed.
}

TEST(GpuMemoryMetricsRegistryTest, AllocationFailureIncrementsFailureCount) {
  constexpr int kDeviceId = 8;
  const GpuMemoryMetricsSnapshot before = GpuMemoryMetricsRegistry::snapshot(kDeviceId);
  GpuMemoryMetricsRegistry::record_allocation_failure(kDeviceId);
  const GpuMemoryMetricsSnapshot after = GpuMemoryMetricsRegistry::snapshot(kDeviceId);
  EXPECT_EQ(after.allocation_failures, before.allocation_failures + 1);
  // A failed allocation was never actually allocated -- no other counter
  // should move.
  EXPECT_EQ(after.current_bytes, before.current_bytes);
  EXPECT_EQ(after.allocations, before.allocations);
}

TEST(GpuMemoryMetricsRegistryTest, DevicesAreIndependent) {
  constexpr int kDeviceA = 9;
  constexpr int kDeviceB = 10;
  const std::uint64_t a_before = GpuMemoryMetricsRegistry::snapshot(kDeviceA).current_bytes;
  const std::uint64_t b_before = GpuMemoryMetricsRegistry::snapshot(kDeviceB).current_bytes;

  GpuMemoryMetricsRegistry::record_allocation(kDeviceA, 500);

  EXPECT_EQ(GpuMemoryMetricsRegistry::snapshot(kDeviceA).current_bytes, a_before + 500);
  EXPECT_EQ(GpuMemoryMetricsRegistry::snapshot(kDeviceB).current_bytes, b_before);
}

TEST(GpuMemoryMetricsRegistryTest, KnownDeviceIdsIncludesEveryDeviceThatHasRecordedActivity) {
  constexpr int kDeviceId = 12;
  GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 1);
  const std::vector<int> known = GpuMemoryMetricsRegistry::known_device_ids();
  EXPECT_NE(std::find(known.begin(), known.end(), kDeviceId), known.end());
}

TEST(GpuMemoryMetricsRegistryTest, OutOfRangeDeviceIdIsSilentlyIgnoredNotCrashed) {
  EXPECT_NO_THROW(GpuMemoryMetricsRegistry::record_allocation(999, 1));
  EXPECT_NO_THROW(GpuMemoryMetricsRegistry::record_deallocation(999, 1));
  EXPECT_NO_THROW(GpuMemoryMetricsRegistry::record_allocation_failure(999));
  EXPECT_NO_THROW((void)GpuMemoryMetricsRegistry::snapshot(999));
  EXPECT_NO_THROW(GpuMemoryMetricsRegistry::record_allocation(-1, 1));
}

TEST(GpuMemoryMetricsRegistryTest, ConcurrentAllocationsAndDeallocationsDoNotCorruptState) {
  constexpr int kDeviceId = 11;
  constexpr int kThreads = 8;
  constexpr int kOpsPerThread = 5000;
  const GpuMemoryMetricsSnapshot before = GpuMemoryMetricsRegistry::snapshot(kDeviceId);

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([kDeviceId] {
      for (int i = 0; i < kOpsPerThread; ++i) {
        GpuMemoryMetricsRegistry::record_allocation(kDeviceId, 1);
        GpuMemoryMetricsRegistry::record_deallocation(kDeviceId, 1);
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  const GpuMemoryMetricsSnapshot after = GpuMemoryMetricsRegistry::snapshot(kDeviceId);
  constexpr std::uint64_t kExpectedOps = static_cast<std::uint64_t>(kThreads) * kOpsPerThread;
  EXPECT_EQ(after.allocations, before.allocations + kExpectedOps);
  EXPECT_EQ(after.deallocations, before.deallocations + kExpectedOps);
  EXPECT_EQ(after.allocated_total_bytes, before.allocated_total_bytes + kExpectedOps);
  EXPECT_EQ(after.current_bytes, before.current_bytes);  // every allocation was matched by a free.
}

// --- Integration: a real RmmEnvironment/TrackingMemoryResource, real GPU
// allocations, going through the actual production resource stack. Always
// targets device_id 0 (RmmEnvironment's default constructor argument),
// same as every other GPU test in this binary -- compares before/after
// deltas, not absolute values, since that device_id's counters are shared
// process-wide state.

TEST(GpuMemoryMetricsTrackingResourceTest, AllocatingRealDeviceBufferIncreasesCurrentBytes) {
  EngineConfig config = default_config();
  RmmEnvironment env(config);

  const std::uint64_t before = GpuMemoryMetricsRegistry::snapshot(0).current_bytes;
  constexpr std::size_t kBytes = 1024 * 1024;  // 1 MiB
  rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{});
  const std::uint64_t during = GpuMemoryMetricsRegistry::snapshot(0).current_bytes;

  EXPECT_GE(during, before + kBytes);
}

TEST(GpuMemoryMetricsTrackingResourceTest, FreeingRealDeviceBufferDecreasesCurrentBytesBackToBaseline) {
  EngineConfig config = default_config();
  RmmEnvironment env(config);

  const std::uint64_t before = GpuMemoryMetricsRegistry::snapshot(0).current_bytes;
  const std::uint64_t deallocations_before = GpuMemoryMetricsRegistry::snapshot(0).deallocations;
  { rmm::device_buffer buffer(1024 * 1024, rmm::cuda_stream_view{}); }
  const GpuMemoryMetricsSnapshot after = GpuMemoryMetricsRegistry::snapshot(0);

  EXPECT_EQ(after.current_bytes, before);
  EXPECT_GT(after.deallocations, deallocations_before);
}

TEST(GpuMemoryMetricsTrackingResourceTest, PeakRemainsElevatedAfterFreeingRealAllocation) {
  EngineConfig config = default_config();
  RmmEnvironment env(config);

  constexpr std::size_t kBytes = 2 * 1024 * 1024;  // 2 MiB
  std::uint64_t peak_during = 0;
  {
    rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{});
    peak_during = GpuMemoryMetricsRegistry::snapshot(0).peak_bytes;
  }
  const std::uint64_t peak_after = GpuMemoryMetricsRegistry::snapshot(0).peak_bytes;

  EXPECT_EQ(peak_after, peak_during);
}

TEST(GpuMemoryMetricsTrackingResourceTest, ExceedingQueryMemoryLimitIncrementsFailureCount) {
  EngineConfig config = default_config();
  config.engine.query_memory_limit_bytes = 1024;  // Deliberately tiny -- matches
                                                  // RmmEnvironment.RespectsConfiguredQueryMemoryLimit's
                                                  // own setup (rmm_environment_test.cpp).
  RmmEnvironment env(config);

  const std::uint64_t before = GpuMemoryMetricsRegistry::snapshot(0).allocation_failures;
  EXPECT_THROW((void)({
                 rmm::device_buffer buffer(64 * 1024 * 1024, rmm::cuda_stream_view{});  // 64 MiB > limit
               }),
               std::exception);
  const std::uint64_t after = GpuMemoryMetricsRegistry::snapshot(0).allocation_failures;

  EXPECT_EQ(after, before + 1);
}

// Telemetry (observability.enabled) is false in default_config() -- every
// test above already exercises this implicitly, but this asserts the
// property directly per this feature's own spec: a normal allocate/
// deallocate cycle must not throw regardless of telemetry state.
TEST(GpuMemoryMetricsTrackingResourceTest, TelemetryDisabledDoesNotBreakAllocation) {
  EngineConfig config = default_config();
  ASSERT_FALSE(config.observability.enabled);
  RmmEnvironment env(config);

  EXPECT_NO_THROW((void)({ rmm::device_buffer buffer(1024, rmm::cuda_stream_view{}); }));
}

}  // namespace
}  // namespace kernellake
