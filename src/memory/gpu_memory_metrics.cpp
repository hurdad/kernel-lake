#include "kernellake/memory/gpu_memory_metrics.hpp"

namespace kernellake {

std::array<GpuMemoryMetricsRegistry::DeviceState, kMaxTrackedGpuDevices>&
GpuMemoryMetricsRegistry::devices() {
  static std::array<DeviceState, kMaxTrackedGpuDevices> instances;
  return instances;
}

namespace {
// Out-of-range device_ids are dropped rather than asserted/thrown: this is a
// metrics side channel (see the header's own comment), and a config value
// KernelLake itself validated (EngineSection::device_id) should never
// realistically exceed kMaxTrackedGpuDevices, but a metrics bug must never
// be the reason a query fails.
bool in_range(int device_id) {
  return device_id >= 0 && device_id < kMaxTrackedGpuDevices;
}
}  // namespace

void GpuMemoryMetricsRegistry::record_allocation(int device_id, std::uint64_t bytes) noexcept {
  if (!in_range(device_id)) {
    return;
  }
  DeviceState& state = devices()[static_cast<std::size_t>(device_id)];
  state.active.store(true, std::memory_order_relaxed);
  state.allocations.fetch_add(1, std::memory_order_relaxed);
  state.allocated_total_bytes.fetch_add(bytes, std::memory_order_relaxed);

  const std::uint64_t current = state.current_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  std::uint64_t peak = state.peak_bytes.load(std::memory_order_relaxed);
  while (current > peak &&
         !state.peak_bytes.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
  }
}

void GpuMemoryMetricsRegistry::record_deallocation(int device_id, std::uint64_t bytes) noexcept {
  if (!in_range(device_id)) {
    return;
  }
  DeviceState& state = devices()[static_cast<std::size_t>(device_id)];
  state.deallocations.fetch_add(1, std::memory_order_relaxed);
  state.current_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

void GpuMemoryMetricsRegistry::record_allocation_failure(int device_id) noexcept {
  if (!in_range(device_id)) {
    return;
  }
  DeviceState& state = devices()[static_cast<std::size_t>(device_id)];
  state.active.store(true, std::memory_order_relaxed);
  state.allocation_failures.fetch_add(1, std::memory_order_relaxed);
}

GpuMemoryMetricsSnapshot GpuMemoryMetricsRegistry::snapshot(int device_id) noexcept {
  GpuMemoryMetricsSnapshot result;
  result.device_id = device_id;
  if (!in_range(device_id)) {
    return result;
  }
  const DeviceState& state = devices()[static_cast<std::size_t>(device_id)];
  result.current_bytes = state.current_bytes.load(std::memory_order_relaxed);
  result.peak_bytes = state.peak_bytes.load(std::memory_order_relaxed);
  result.allocations = state.allocations.load(std::memory_order_relaxed);
  result.deallocations = state.deallocations.load(std::memory_order_relaxed);
  result.allocated_total_bytes = state.allocated_total_bytes.load(std::memory_order_relaxed);
  result.allocation_failures = state.allocation_failures.load(std::memory_order_relaxed);
  return result;
}

std::vector<int> GpuMemoryMetricsRegistry::known_device_ids() {
  std::vector<int> result;
  const auto& all = devices();
  for (int i = 0; i < kMaxTrackedGpuDevices; ++i) {
    if (all[static_cast<std::size_t>(i)].active.load(std::memory_order_relaxed)) {
      result.push_back(i);
    }
  }
  return result;
}

}  // namespace kernellake
