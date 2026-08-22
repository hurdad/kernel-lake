// Provides GpuExecutionCoordinator for KERNELLAKE_WITH_CUDA=ON builds.
// Mutually exclusive with gpu_execution_coordinator_stub.cpp -- see that
// file's comment.
#include "kernellake/server/gpu_execution_coordinator.hpp"

#include <semaphore>

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

struct GpuExecutionCoordinator::Impl {
  explicit Impl(const EngineConfig& config)
      : rmm_environment(config), execute_semaphore(config.engine.max_concurrent_gpu_queries) {}

  RmmEnvironment rmm_environment;
  // Bounds how many queries run concurrently against the shared GPU --
  // see EngineSection::max_concurrent_gpu_queries's own doc comment for
  // why this is a semaphore (bounded concurrency) rather than either the
  // old std::mutex (unconditional serialization, opt #2's whole reason
  // for existing) or no gate at all (real oversubscription/contention
  // risks, see that same comment). RmmEnvironment's own resource stack
  // (limiter shared/thread-safe, per-query statistics via
  // make_query_tracker()) is what makes running more than one query at a
  // time through the same rmm_environment safe at all -- see that class's
  // own comments.
  std::counting_semaphore<> execute_semaphore;
};

GpuExecutionCoordinator::GpuExecutionCoordinator(const EngineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

GpuExecutionCoordinator::~GpuExecutionCoordinator() = default;

QueryResult GpuExecutionCoordinator::execute(const QueryEngine& engine, const PhysicalPlanPtr& physical) {
  const SemaphoreGuard guard(impl_->execute_semaphore);
  return engine.execute(physical, impl_->rmm_environment);
}

}  // namespace kernellake
