// Provides the real register_gpu_memory_otel_instruments() for
// KERNELLAKE_ENABLE_OTEL=ON builds. Mutually exclusive with
// gpu_memory_metrics_otel_stub.cpp -- see src/memory/CMakeLists.txt.
//
// Deliberately talks to opentelemetry-cpp's global Provider directly
// (opentelemetry::metrics::Provider::GetMeterProvider()) rather than going
// through kernellake::observability's own API: that module has no CUDA/RMM
// dependency and must stay that way (it's built in every preset, including
// CPU-only ones -- see kernellake_observability's own CMakeLists.txt
// comment), so it can't expose a GPU-specific registration hook. The
// Provider itself is process-wide and already holds whatever MeterProvider
// observability::init() installed by the time any RmmEnvironment is
// constructed (query_engine_execute_gpu.cpp/GpuExecutionCoordinator always
// run after main()'s own observability::init() call) -- any translation
// unit linking opentelemetry-cpp can read it, regardless of which module
// set it.
#include "kernellake/memory/gpu_memory_otel.hpp"

#include <opentelemetry/metrics/observer_result.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/nostd/variant.h>

#include <mutex>

#include "kernellake/memory/gpu_memory_metrics.hpp"

namespace kernellake {

namespace {

namespace metrics_api = opentelemetry::metrics;

using ObserverResultI64 = opentelemetry::nostd::shared_ptr<metrics_api::ObserverResultT<int64_t>>;

// Every callback below shares this shape: iterate every device that has
// ever recorded activity, observe one GpuMemoryMetricsSnapshot field per
// device, tagged with a gpu.device.id attribute. gpu.device.id (not
// query.id/request.id/trace.id/etc.) is the only attribute -- see
// docs/OBSERVABILITY.md's own note on why query-scoped identifiers never
// become metric dimensions here (unbounded cardinality). `result` is
// always the int64 variant alternative: every instrument this file creates
// is an Int64Observable*, never a double one. GpuMemoryMetricsSnapshot's
// own fields are std::uint64_t (byte counts and monotonic counts are never
// negative); OTel's int64 observable instrument API takes int64_t --
// static_cast per-call here, not by changing the registry's own field
// types (see its header for why they're unsigned).
void observe_current_bytes(metrics_api::ObserverResult result, void*) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  for (const int device_id : GpuMemoryMetricsRegistry::known_device_ids()) {
    const GpuMemoryMetricsSnapshot snapshot = GpuMemoryMetricsRegistry::snapshot(device_id);
    observer->Observe(static_cast<std::int64_t>(snapshot.current_bytes),
                      {{"gpu.device.id", static_cast<std::int64_t>(device_id)}});
  }
}

void observe_peak_bytes(metrics_api::ObserverResult result, void*) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  for (const int device_id : GpuMemoryMetricsRegistry::known_device_ids()) {
    const GpuMemoryMetricsSnapshot snapshot = GpuMemoryMetricsRegistry::snapshot(device_id);
    observer->Observe(static_cast<std::int64_t>(snapshot.peak_bytes),
                      {{"gpu.device.id", static_cast<std::int64_t>(device_id)}});
  }
}

void observe_allocations(metrics_api::ObserverResult result, void*) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  for (const int device_id : GpuMemoryMetricsRegistry::known_device_ids()) {
    const GpuMemoryMetricsSnapshot snapshot = GpuMemoryMetricsRegistry::snapshot(device_id);
    observer->Observe(static_cast<std::int64_t>(snapshot.allocations),
                      {{"gpu.device.id", static_cast<std::int64_t>(device_id)}});
  }
}

void observe_deallocations(metrics_api::ObserverResult result, void*) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  for (const int device_id : GpuMemoryMetricsRegistry::known_device_ids()) {
    const GpuMemoryMetricsSnapshot snapshot = GpuMemoryMetricsRegistry::snapshot(device_id);
    observer->Observe(static_cast<std::int64_t>(snapshot.deallocations),
                      {{"gpu.device.id", static_cast<std::int64_t>(device_id)}});
  }
}

void observe_allocated_total(metrics_api::ObserverResult result, void*) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  for (const int device_id : GpuMemoryMetricsRegistry::known_device_ids()) {
    const GpuMemoryMetricsSnapshot snapshot = GpuMemoryMetricsRegistry::snapshot(device_id);
    observer->Observe(static_cast<std::int64_t>(snapshot.allocated_total_bytes),
                      {{"gpu.device.id", static_cast<std::int64_t>(device_id)}});
  }
}

void observe_allocation_failures(metrics_api::ObserverResult result, void*) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  for (const int device_id : GpuMemoryMetricsRegistry::known_device_ids()) {
    const GpuMemoryMetricsSnapshot snapshot = GpuMemoryMetricsRegistry::snapshot(device_id);
    observer->Observe(static_cast<std::int64_t>(snapshot.allocation_failures),
                      {{"gpu.device.id", static_cast<std::int64_t>(device_id)}});
  }
}

// Kept alive for the process lifetime -- opentelemetry-cpp's own examples
// and Meter::CreateXxxObservable*() contract both expect the returned
// shared_ptr<ObservableInstrument> to outlive AddCallback() registration;
// letting it go out of scope would be equivalent to RemoveCallback().
struct Instruments {
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> allocated;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> peak;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> allocations;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> deallocations;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> allocated_total;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> allocation_failures;
};

Instruments g_instruments;  // NOLINT(cert-err58-cpp) -- default-constructed, no throwing init.

// A plain mutex-guarded bool, not std::once_flag: reset_gpu_memory_otel_
// instruments_for_testing() needs to clear it back to "not yet
// registered", which std::once_flag has no supported way to do. Real
// (non-test) callers only ever call register_gpu_memory_otel_instruments(),
// which behaves exactly like the call_once it replaces.
std::mutex g_registration_mutex;
bool g_registered = false;

void create_instruments() {
  const opentelemetry::nostd::shared_ptr<metrics_api::MeterProvider> provider =
      metrics_api::Provider::GetMeterProvider();
  const opentelemetry::nostd::shared_ptr<metrics_api::Meter> meter = provider->GetMeter("kernellake.gpu");

  g_instruments.allocated = meter->CreateInt64ObservableGauge(
      "kernellake.gpu.memory.allocated",
      "Current GPU bytes allocated through KernelLake's RMM resource stack", "By");
  g_instruments.allocated->AddCallback(&observe_current_bytes, nullptr);

  g_instruments.peak = meter->CreateInt64ObservableGauge(
      "kernellake.gpu.memory.peak", "Highest observed current-allocated-bytes value since process startup",
      "By");
  g_instruments.peak->AddCallback(&observe_peak_bytes, nullptr);

  // Reported as ObservableCounters (async, read-current-cumulative-value),
  // not synchronous Counters -- a synchronous Counter::Add() call would
  // have to happen from inside TrackingMemoryResource::allocate() itself,
  // on the GPU allocation hot path, which is exactly what this whole
  // design avoids (see GpuMemoryMetricsRegistry's own comment: the hot
  // path only ever touches plain atomics). The OTel SDK's periodic
  // exporter is what turns these into the cumulative counter semantics
  // OTLP expects, from a callback it invokes on its own export thread.
  g_instruments.allocations = meter->CreateInt64ObservableCounter(
      "kernellake.gpu.memory.allocations", "Cumulative successful GPU allocation count", "{allocation}");
  g_instruments.allocations->AddCallback(&observe_allocations, nullptr);

  g_instruments.deallocations = meter->CreateInt64ObservableCounter(
      "kernellake.gpu.memory.deallocations", "Cumulative GPU deallocation count", "{allocation}");
  g_instruments.deallocations->AddCallback(&observe_deallocations, nullptr);

  g_instruments.allocated_total = meter->CreateInt64ObservableCounter(
      "kernellake.gpu.memory.allocated_total", "Cumulative bytes successfully allocated over time", "By");
  g_instruments.allocated_total->AddCallback(&observe_allocated_total, nullptr);

  g_instruments.allocation_failures = meter->CreateInt64ObservableCounter(
      "kernellake.gpu.memory.allocation_failures",
      "Cumulative failed GPU allocation attempts (RMM bad_alloc/out_of_memory, including "
      "query_memory_limit_bytes rejections)",
      "{allocation}");
  g_instruments.allocation_failures->AddCallback(&observe_allocation_failures, nullptr);
}

}  // namespace

void register_gpu_memory_otel_instruments() {
  const std::lock_guard<std::mutex> lock(g_registration_mutex);
  if (g_registered) {
    return;
  }
  create_instruments();
  g_registered = true;
}

void reset_gpu_memory_otel_instruments_for_testing() {
  const std::lock_guard<std::mutex> lock(g_registration_mutex);
  g_instruments = Instruments{};
  g_registered = false;
}

}  // namespace kernellake
