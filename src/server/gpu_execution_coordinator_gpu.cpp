// Provides GpuExecutionCoordinator for KERNELLAKE_WITH_CUDA=ON builds.
// Mutually exclusive with gpu_execution_coordinator_stub.cpp -- see that
// file's comment.
#include "kernellake/server/gpu_execution_coordinator.hpp"

#include <fmt/format.h>

#include <atomic>
#include <cstddef>
#include <semaphore>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {

namespace {

// Minimal RAII guard for std::counting_semaphore -- the standard library
// has no built-in equivalent of std::lock_guard for semaphores. Acquires
// on construction, releases on destruction (including when execute()
// below exits via an exception), matching the exception-safety the old
// std::lock_guard<std::mutex> gave for free.
class SemaphoreGuard {
 public:
  explicit SemaphoreGuard(std::counting_semaphore<>& semaphore) : semaphore_(semaphore) {
    semaphore_.acquire();
  }
  ~SemaphoreGuard() { semaphore_.release(); }

  SemaphoreGuard(const SemaphoreGuard&) = delete;
  SemaphoreGuard& operator=(const SemaphoreGuard&) = delete;

 private:
  std::counting_semaphore<>& semaphore_;
};

}  // namespace

// Tier 1 of docs/MULTI_GPU_SCALING.md ("concurrent queries, one GPU each,
// single node"): one RmmEnvironment per GPU this server is configured to
// use instead of one shared process-wide instance pinned to
// config.engine.device_id, and a per-device semaphore instead of the
// single semaphore opt #2 (docs/GPU_OPTIMIZATIONS.md) introduced to
// replace the original single-flight mutex. execute() below round-robins
// across devices, so an N-GPU configuration gets up to
// N x ServerConfig::max_concurrent_gpu_queries total query throughput
// instead of that same cap applied to one device while the rest sit idle.
// A single query still runs entirely on one GPU -- spanning one query
// across multiple devices is tier 2, not attempted here.
struct GpuExecutionCoordinator::Impl {
  explicit Impl(const ServerConfig& config) {
    int real_device_count = 0;
    check_cuda(cudaGetDeviceCount(&real_device_count), "cudaGetDeviceCount for GpuExecutionCoordinator");
    if (real_device_count <= 0) {
      throw ConfigurationError(
          "server.engine.backend 'gpu' requires at least one visible CUDA device, but "
          "cudaGetDeviceCount() reported none");
    }

    // Empty gpu_device_ids means "every visible device" -- Tier 1's
    // original behavior; see ServerConfig::gpu_device_ids's own comment
    // for why a non-empty list (pinning to a subset) is also supported.
    std::vector<int> device_ids = config.gpu_device_ids;
    if (device_ids.empty()) {
      device_ids.reserve(static_cast<std::size_t>(real_device_count));
      for (int i = 0; i < real_device_count; ++i) {
        device_ids.push_back(i);
      }
    } else {
      for (const int device_id : device_ids) {
        if (device_id >= real_device_count) {
          throw ConfigurationError(fmt::format(
              "engine.gpu_device_ids names device {}, but cudaGetDeviceCount() reports only {} visible "
              "device(s) (valid range: 0-{})",
              device_id, real_device_count, real_device_count - 1));
        }
      }
    }

    environments.reserve(device_ids.size());
    execute_semaphores.reserve(device_ids.size());
    for (const int device_id : device_ids) {
      // set_current_device_resource() (inside RmmEnvironment's constructor)
      // installs into whichever device is current *at construction time* --
      // one slot per device, confirmed from RMM source during the
      // 2026-08-17 mutex investigation (see docs/MULTI_GPU_SCALING.md) --
      // so each instance needs its own device selected first via this
      // guard, restored again once construction finishes.
      const CudaDeviceGuard device_guard(device_id);
      environments.push_back(std::make_unique<RmmEnvironment>(config.engine_config, device_id));
      execute_semaphores.push_back(
          std::make_unique<std::counting_semaphore<>>(config.max_concurrent_gpu_queries));
    }
  }

  // One entry per configured device (environments.size() ==
  // execute_semaphores.size() == config.gpu_device_ids.size(), or
  // cudaGetDeviceCount() if that list was empty at construction).
  std::vector<std::unique_ptr<RmmEnvironment>> environments;
  // Bounds how many queries run concurrently against each device's own
  // shared RmmEnvironment -- see EngineSection::max_concurrent_gpu_queries's
  // own doc comment for why a semaphore (bounded concurrency) rather than
  // either a mutex (unconditional serialization, opt #2's whole reason for
  // existing) or no gate at all. This cap now applies per device, not
  // process-wide: an N-device node allows up to N times as many queries
  // running at once as a single-device one did.
  std::vector<std::unique_ptr<std::counting_semaphore<>>> execute_semaphores;
  // Round-robin device selection: fetch_add-and-mod, no locking needed.
  std::atomic<std::uint64_t> next_device{0};
};

GpuExecutionCoordinator::GpuExecutionCoordinator(const ServerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

GpuExecutionCoordinator::~GpuExecutionCoordinator() = default;

QueryResult GpuExecutionCoordinator::execute(const QueryEngine& engine, const PhysicalPlanPtr& physical) {
  const std::size_t device_count = impl_->environments.size();
  const std::size_t device_index =
      static_cast<std::size_t>(impl_->next_device.fetch_add(1, std::memory_order_relaxed) % device_count);
  const SemaphoreGuard guard(*impl_->execute_semaphores[device_index]);
  return engine.execute(physical, *impl_->environments[device_index]);
}

}  // namespace kernellake
