#include "kernellake/memory/rmm_environment.hpp"

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"

namespace kernellake {

std::uint64_t resolve_query_memory_limit_bytes(const EngineConfig& config) {
  if (config.engine.query_memory_limit_bytes != 0) {
    return config.engine.query_memory_limit_bytes;
  }
  // CudaDeviceGuard: cudaMemGetInfo() reports the *current* device's
  // memory, which may not be config.engine.device_id yet at whatever point
  // a caller invokes this from (RmmEnvironment's own constructor, in
  // particular, runs before query_engine_execute_gpu.cpp's own
  // CudaDeviceGuard for the query itself -- see that file's comment on
  // construction order).
  const CudaDeviceGuard device_guard(config.engine.device_id);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
             "cudaMemGetInfo for query_memory_limit_bytes auto-detect");
  // free_bytes, not total_bytes -- confirmed a real, not hypothetical,
  // reason this matters: this project's own dev GPU (WSL2, RTX 5060 Ti,
  // also used interactively -- e.g. gaming -- not a dedicated headless
  // card) can have several GiB of its 16 GiB total held by something else
  // entirely, fluctuating in real time, unrelated to anything kernellake
  // does. Sizing off total_bytes there would set a limit the allocator
  // can never actually satisfy in full, reproducing the exact "Exceeded
  // memory limit" failure this auto-detection exists to avoid, just with
  // a subtler cause (nothing obviously wrong, yet still OOM).
  //
  // 90%, not a more conservative fraction: an empirical real-GPU
  // comparison (RTX 5060 Ti, TPC-H Q3's 3-way join at SF10) found 75% of
  // free (a first attempt) too tight -- it failed with "Exceeded memory
  // limit", needing noticeably more -- while a manually configured ~12
  // GiB ceiling succeeded even when free VRAM was reportedly lower than
  // that at measurement time (the query's actual peak usage turned out to
  // fit regardless, since this ceiling only *permits* allocation up to
  // that amount rather than reserving it upfront; the real hardware
  // ceiling is still enforced independently by the CUDA allocator itself).
  // Being too conservative here has a worse failure mode than being too
  // generous: an over-tight auto-detected limit fails a query that could
  // have actually fit, while an over-generous one, in the rare case it's
  // still not enough, produces the exact same clean, already-handled
  // "Exceeded memory limit" error a manually-misconfigured value would.
  return static_cast<std::uint64_t>(free_bytes) * 9 / 10;
}

struct RmmEnvironment::Impl {
  cuda::mr::any_resource<cuda::mr::device_accessible> base_resource;
  rmm::mr::statistics_resource_adaptor stats;
  rmm::mr::limiting_resource_adaptor limiter;
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_resource;

  Impl(const EngineConfig& config, cuda::mr::any_resource<cuda::mr::device_accessible> base,
       cuda::mr::any_resource<cuda::mr::device_accessible> previous)
      : base_resource(std::move(base)),
        stats(base_resource),
        limiter(stats, resolve_query_memory_limit_bytes(config)),
        previous_resource(std::move(previous)) {}
};

namespace {

cuda::mr::any_resource<cuda::mr::device_accessible> build_base_resource(const MemorySection& memory) {
  if (memory.use_async_allocator) {
    return cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::cuda_async_memory_resource{}};
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
  RmmEnvironment::Impl* raw = new Impl(config, base, cuda::mr::any_resource<cuda::mr::device_accessible>{});
  impl_.reset(raw);
  impl_->previous_resource = rmm::mr::set_current_device_resource(impl_->limiter);
}

RmmEnvironment::~RmmEnvironment() {
  if (impl_) {
    // impl_'s pool (or async-allocator) resource is about to be torn down
    // and its underlying device memory returned to the driver. Without this
    // sync, any GPU work still in flight on the device (e.g. a kernel
    // launched via cudf against a batch that hasn't been awaited yet) can
    // read or write into memory that gets freed and reissued to a *later*
    // RmmEnvironment's pool -- corrupting unrelated, much later work instead
    // of failing at the source. Caught via a full-suite run that crashed
    // deep inside an unrelated later test with a driver-level GPF; every
    // test passes individually or in small combinations because there is no
    // still-in-flight work left to race against.
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before tearing down RmmEnvironment");
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
