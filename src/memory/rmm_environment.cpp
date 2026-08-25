#include "kernellake/memory/rmm_environment.hpp"

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <spdlog/spdlog.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/memory/gpu_memory_otel.hpp"
#include "kernellake/memory/gpu_memory_tracking_resource.hpp"

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
  int device_id;
  std::uint64_t query_memory_limit_bytes;
  cuda::mr::any_resource<cuda::mr::device_accessible> base_resource;
  // SHARED across every concurrently in-flight query -- deliberately not
  // one per query. This is the real, physical memory ceiling for the whole
  // device; splitting it per-query would let N concurrent queries each
  // think they have the *full* ceiling to themselves, collectively
  // oversubscribing one real GPU's memory (query_engine_execute_gpu.cpp's
  // pass_read_limit_bytes/build_side_budget_bytes are each sized as a
  // fraction of this value for exactly one query's use). Already
  // thread-safe (atomics-based, per RMM's own header) -- safe to share.
  rmm::mr::limiting_resource_adaptor limiter;
  // Outermost layer: every allocation actually issued through
  // set_current_device_resource() below passes through this first (and its
  // deallocations last) -- see TrackingMemoryResource's own doc comment for
  // why that position matters (it sees limiter rejections, not just genuine
  // CUDA OOM). Feeds GpuMemoryMetricsRegistry, a process-wide counter store
  // independent of this Impl's own lifetime.
  TrackingMemoryResource tracking;
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_resource;

  Impl(const EngineConfig& config, cuda::mr::any_resource<cuda::mr::device_accessible> base,
       cuda::mr::any_resource<cuda::mr::device_accessible> previous)
      : device_id(config.engine.device_id),
        query_memory_limit_bytes(resolve_query_memory_limit_bytes(config)),
        base_resource(std::move(base)),
        limiter(base_resource, query_memory_limit_bytes),
        tracking(cuda::mr::any_resource<cuda::mr::device_accessible>{limiter}, config.engine.device_id),
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
  // Idempotent (internally std::call_once-guarded) -- registers the GPU
  // memory OTel instruments against whatever global MeterProvider
  // observability::init() already set up, the first time any
  // RmmEnvironment is ever constructed in this process. A no-op when
  // observability is disabled or this isn't a KERNELLAKE_ENABLE_OTEL build
  // -- see gpu_memory_otel.hpp.
  register_gpu_memory_otel_instruments();

  cuda::mr::any_resource<cuda::mr::device_accessible> base = build_base_resource(config.memory);
  // set_current_device_resource returns the *previous* resource so it can
  // be restored on destruction; construct our Impl in two steps since the
  // limiter needs `base`, all before we know what the previous global
  // resource was.
  RmmEnvironment::Impl* raw = new Impl(config, base, cuda::mr::any_resource<cuda::mr::device_accessible>{});
  impl_.reset(raw);
  impl_->previous_resource = rmm::mr::set_current_device_resource(impl_->tracking);
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
    //
    // check_cuda() throws on failure; a destructor is implicitly noexcept,
    // so letting that propagate would call std::terminate() and crash the
    // whole process (kernellake-server keeps one long-lived RmmEnvironment
    // for its entire lifetime, so this runs at shutdown/restart, not just
    // in a short-lived CLI process) instead of just failing to restore the
    // previous resource cleanly. Caught and logged instead -- if the CUDA
    // context is already in a broken enough state for this to fail, the
    // set_current_device_resource() call below is unlikely to succeed
    // either, but a log line beats a hard crash.
    try {
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before tearing down RmmEnvironment");
    } catch (const CudaError& e) {
      spdlog::error(
          "RmmEnvironment teardown: {} -- device memory may not be safe to reuse; continuing anyway rather "
          "than crashing the process",
          e.what());
    }
    rmm::mr::set_current_device_resource(impl_->previous_resource);
  }
}

QueryMemoryTracker RmmEnvironment::make_query_tracker() {
  // Wraps the shared limiter, not base_resource directly -- every query's
  // allocations must still count against the one real, shared device
  // ceiling (see Impl::limiter's own comment), just reported through this
  // call's own independent statistics_resource_adaptor instance rather
  // than a stack shared with any other concurrently-running query.
  return QueryMemoryTracker(cuda::mr::any_resource<cuda::mr::device_accessible>{impl_->limiter});
}

std::uint64_t RmmEnvironment::query_memory_limit_bytes() const {
  return impl_->query_memory_limit_bytes;
}

int RmmEnvironment::device_id() const {
  return impl_->device_id;
}

}  // namespace kernellake
