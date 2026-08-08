#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kernellake {

// Upper bound on distinct CUDA device ordinals tracked -- generous for any
// real single-node GPU count (nodes with >16 GPUs are not a configuration
// KernelLake targets; see EngineSection::device_id, always a small ordinal
// in practice). A fixed-size array, not a map, so record_allocation()/
// record_deallocation() on the allocation hot path never touches a mutex or
// allocates -- see GpuMemoryMetricsRegistry's own comment.
inline constexpr int kMaxTrackedGpuDevices = 16;

// Point-in-time read of one device's counters -- see
// GpuMemoryMetricsRegistry::snapshot(). Cumulative fields (allocations,
// deallocations, allocated_total_bytes, allocation_failures) are monotonic
// since process startup; current_bytes/peak_bytes are live state.
struct GpuMemoryMetricsSnapshot {
  int device_id = 0;
  std::uint64_t current_bytes = 0;
  std::uint64_t peak_bytes = 0;
  std::uint64_t allocations = 0;
  std::uint64_t deallocations = 0;
  std::uint64_t allocated_total_bytes = 0;
  std::uint64_t allocation_failures = 0;
};

// Process-wide, per-CUDA-device GPU allocation counters, fed by
// TrackingMemoryResource (below) from inside RmmEnvironment's resource
// stack. Deliberately a separate, plain atomic struct rather than reading
// rmm::mr::statistics_resource_adaptor directly (KernelLake's
// RmmEnvironment already uses one, for a different purpose -- see its own
// comment): that adaptor's counters are scoped to *its own instance's*
// lifetime, which is the *query's* lifetime for the CLI (a fresh
// RmmEnvironment, and thus a fresh statistics_resource_adaptor, gets built
// and torn down per query -- see query_engine_execute_gpu.cpp). Counters
// meant to survive that churn and represent the whole process's history
// (matching what an external metrics consumer expects -- "peak since
// process startup", cumulative allocated bytes across every query so far)
// need storage that outlives any single RmmEnvironment instance, hence this
// static/global registry instead.
//
// Every method here is safe to call from the CUDA allocation hot path:
// fixed-size array indexing plus a handful of atomic ops, no locks, no
// allocation, no logging, no OTel SDK calls (the OTel-facing side --
// gpu_memory_metrics_otel.cpp -- only ever *reads* a snapshot, on its own
// periodic export thread, never from here).
class GpuMemoryMetricsRegistry {
 public:
  // Records one successful allocation of `bytes` on `device_id`. Updates
  // current/peak/allocations/allocated_total_bytes.
  static void record_allocation(int device_id, std::uint64_t bytes) noexcept;

  // Records one deallocation of `bytes` on `device_id`. Updates
  // current_bytes/deallocations. Does not touch peak_bytes (peak only ever
  // moves up -- see record_allocation()).
  static void record_deallocation(int device_id, std::uint64_t bytes) noexcept;

  // Records one failed allocation attempt on `device_id` (RMM bad_alloc/
  // out_of_memory, or any other exception TrackingMemoryResource's
  // allocate() catches and rethrows). Does not touch any other counter --
  // a failed allocation was never actually allocated.
  static void record_allocation_failure(int device_id) noexcept;

  // Point-in-time read of `device_id`'s counters. Safe to call from any
  // thread at any time (including concurrently with the hot-path methods
  // above) -- individual fields may not represent one atomic instant
  // together, same tradeoff rmm::mr::statistics_resource_adaptor itself
  // makes for its own counter reads.
  [[nodiscard]] static GpuMemoryMetricsSnapshot snapshot(int device_id) noexcept;

  // Every device_id that has recorded at least one allocation attempt
  // (success or failure) since process startup -- what the OTel
  // ObservableGauge/ObservableCounter callbacks iterate over, so a device
  // KernelLake has never touched doesn't show up as a spurious all-zero
  // series. Allocates (a std::vector) -- never called from the hot path,
  // only from the periodic OTel export callback.
  [[nodiscard]] static std::vector<int> known_device_ids();

 private:
  struct DeviceState {
    std::atomic<bool> active{false};
    std::atomic<std::uint64_t> current_bytes{0};
    std::atomic<std::uint64_t> peak_bytes{0};
    std::atomic<std::uint64_t> allocations{0};
    std::atomic<std::uint64_t> deallocations{0};
    std::atomic<std::uint64_t> allocated_total_bytes{0};
    std::atomic<std::uint64_t> allocation_failures{0};
  };

  // std::array<DeviceState, N>, not std::vector -- fixed at compile time so
  // there is no dynamic allocation anywhere in this class, including at
  // first use. Indexed directly by device_id (checked against
  // kMaxTrackedGpuDevices; out-of-range device_ids are silently dropped --
  // see .cpp -- since this is a metrics side channel, not something a
  // caller depends on for correctness).
  static std::array<DeviceState, kMaxTrackedGpuDevices>& devices();
};

}  // namespace kernellake
