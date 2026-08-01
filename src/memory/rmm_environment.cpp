#include "kernellake/memory/rmm_environment.hpp"

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include "kernellake/common/errors.hpp"

namespace kernellake {

struct RmmEnvironment::Impl {
  cuda::mr::any_resource<cuda::mr::device_accessible> base_resource;
  rmm::mr::statistics_resource_adaptor stats;
  rmm::mr::limiting_resource_adaptor limiter;
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_resource;

  Impl(const EngineConfig& config, cuda::mr::any_resource<cuda::mr::device_accessible> base,
       cuda::mr::any_resource<cuda::mr::device_accessible> previous)
      : base_resource(std::move(base)),
        stats(base_resource),
        limiter(stats, config.engine.query_memory_limit_bytes),
        previous_resource(std::move(previous)) {}
};

namespace {

cuda::mr::any_resource<cuda::mr::device_accessible> build_base_resource(
    const MemorySection& memory) {
  if (memory.use_async_allocator) {
    return cuda::mr::any_resource<cuda::mr::device_accessible>{
        rmm::mr::cuda_async_memory_resource{}};
  }
  return cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::pool_memory_resource{
      cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
      memory.pool_initial_bytes, memory.pool_max_bytes}};
}

}  // namespace

RmmEnvironment::RmmEnvironment(const EngineConfig& config) {
  cuda::mr::any_resource<cuda::mr::device_accessible> base = build_base_resource(config.memory);
  // set_current_device_resource returns the *previous* resource so it can
  // be restored on destruction; construct our Impl in two steps since the
  // limiter needs `stats`, which needs `base`, all before we know what the
  // previous global resource was.
  RmmEnvironment::Impl* raw =
      new Impl(config, base, cuda::mr::any_resource<cuda::mr::device_accessible>{});
  impl_.reset(raw);
  impl_->previous_resource = rmm::mr::set_current_device_resource(impl_->limiter);
}

RmmEnvironment::~RmmEnvironment() {
  if (impl_) {
    rmm::mr::set_current_device_resource(impl_->previous_resource);
  }
}

MemoryUsage RmmEnvironment::track_query(const std::function<void()>& query) {
  impl_->stats.push_counters();
  query();
  const auto [bytes, allocations] = impl_->stats.pop_counters();
  (void)allocations;
  return MemoryUsage{bytes.value, bytes.peak};
}

MemoryUsage RmmEnvironment::current_usage() const {
  const rmm::mr::statistics_resource_adaptor::counter bytes = impl_->stats.get_bytes_counter();
  return MemoryUsage{bytes.value, bytes.peak};
}

}  // namespace kernellake
