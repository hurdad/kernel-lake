// Provides the real register_nvme_cache_otel_instruments() for
// KERNELLAKE_ENABLE_OTEL=ON builds. Mutually exclusive with
// nvme_cache_metrics_otel_stub.cpp -- see src/storage/CMakeLists.txt.
//
// Talks to opentelemetry-cpp's global Provider directly
// (opentelemetry::metrics::Provider::GetMeterProvider()), same reasoning as
// src/memory/gpu_memory_metrics_otel.cpp's own comment: kernellake_storage
// has no CUDA/observability-module dependency and must stay that way (built
// in every preset), so it can't route through kernellake::observability's
// own API. The Provider is process-wide and already holds whatever
// MeterProvider observability::init() installed by the time
// register_cache_otel_instruments() is called (kernellake-server's
// constructor always runs after main()'s own observability::init() call).
#include "kernellake/storage/nvme_cache_otel.hpp"

#include <opentelemetry/metrics/observer_result.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/nostd/variant.h>

#include <optional>

#include "kernellake/storage/nvme_object_cache.hpp"
#include "kernellake/storage/object_store_registry.hpp"

namespace kernellake {

namespace {

namespace metrics_api = opentelemetry::metrics;

using ObserverResultI64 = opentelemetry::nostd::shared_ptr<metrics_api::ObserverResultT<int64_t>>;

// `state` is always the `const ObjectStoreRegistry*` passed to AddCallback()
// in create_instruments() below -- unlike gpu_memory_metrics_otel.cpp's
// callbacks (which read a process-wide static registry and ignore this
// parameter), these need it: NvmeObjectCache is a plain instance member,
// not a global. Observes nothing at all if the cache is disabled -- a
// deliberately empty observation, not a spurious all-zero series, for a
// registry this function was never asked to skip (register_nvme_cache_otel_
// instruments() already returns early for that case; this null check is
// just defense against a config that changed cache.enabled after
// registration, which nothing in this codebase does today but costs
// nothing to handle correctly here).
void observe_hits(metrics_api::ObserverResult result, void* state) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  const auto* registry = static_cast<const ObjectStoreRegistry*>(state);
  if (const std::optional<NvmeCacheMetricsSnapshot> snapshot = registry->cache_metrics()) {
    observer->Observe(static_cast<std::int64_t>(snapshot->hits));
  }
}

void observe_misses(metrics_api::ObserverResult result, void* state) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  const auto* registry = static_cast<const ObjectStoreRegistry*>(state);
  if (const std::optional<NvmeCacheMetricsSnapshot> snapshot = registry->cache_metrics()) {
    observer->Observe(static_cast<std::int64_t>(snapshot->misses));
  }
}

void observe_evictions(metrics_api::ObserverResult result, void* state) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  const auto* registry = static_cast<const ObjectStoreRegistry*>(state);
  if (const std::optional<NvmeCacheMetricsSnapshot> snapshot = registry->cache_metrics()) {
    observer->Observe(static_cast<std::int64_t>(snapshot->evictions));
  }
}

void observe_current_bytes(metrics_api::ObserverResult result, void* state) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  const auto* registry = static_cast<const ObjectStoreRegistry*>(state);
  if (const std::optional<NvmeCacheMetricsSnapshot> snapshot = registry->cache_metrics()) {
    observer->Observe(static_cast<std::int64_t>(snapshot->current_bytes));
  }
}

void observe_current_entries(metrics_api::ObserverResult result, void* state) {
  const auto& observer = opentelemetry::nostd::get<ObserverResultI64>(result);
  const auto* registry = static_cast<const ObjectStoreRegistry*>(state);
  if (const std::optional<NvmeCacheMetricsSnapshot> snapshot = registry->cache_metrics()) {
    observer->Observe(static_cast<std::int64_t>(snapshot->current_entries));
  }
}

// Kept alive for the process lifetime, same reasoning as
// gpu_memory_metrics_otel.cpp's own g_instruments: opentelemetry-cpp's
// Meter::CreateXxxObservable*() contract expects the returned
// shared_ptr<ObservableInstrument> to outlive AddCallback() registration.
// A plain function-local static (not a member of the ObjectStoreRegistry
// this call is for) is fine here specifically because this function is
// documented to run at most once per process -- kernellake-server's
// constructor is the only real call site.
struct Instruments {
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> hits;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> misses;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> evictions;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> current_bytes;
  opentelemetry::nostd::shared_ptr<metrics_api::ObservableInstrument> current_entries;
};

Instruments g_instruments;  // NOLINT(cert-err58-cpp) -- default-constructed, no throwing init.

}  // namespace

void register_nvme_cache_otel_instruments(const ObjectStoreRegistry& registry) {
  if (!registry.cache_metrics().has_value()) {
    return;  // storage.cache.enabled is false -- nothing to export.
  }

  const opentelemetry::nostd::shared_ptr<metrics_api::MeterProvider> provider =
      metrics_api::Provider::GetMeterProvider();
  const opentelemetry::nostd::shared_ptr<metrics_api::Meter> meter = provider->GetMeter("kernellake.storage");

  // `&registry` is passed as AddCallback()'s user-data, threaded straight
  // through to the ObserverResult callbacks above -- see this function's
  // own header comment for the lifetime requirement this relies on.
  // const_cast is safe here: every callback above only ever calls
  // cache_metrics() const on it.
  void* const registry_ptr = const_cast<ObjectStoreRegistry*>(&registry);

  // ObservableCounters (async, read-current-cumulative-value), not
  // synchronous Counters -- same reasoning as gpu_memory_metrics_otel.cpp:
  // a synchronous Counter::Add() call would have to happen from inside
  // NvmeObjectCache::get_or_populate() itself, which is exactly what this
  // design avoids (that method only ever touches plain atomics).
  g_instruments.hits = meter->CreateInt64ObservableCounter(
      "kernellake.storage.cache.hits",
      "Cumulative NVMe cache hits (get_or_populate() calls that found an existing entry, plus "
      "successful cached_info() lookups)",
      "{hit}");
  g_instruments.hits->AddCallback(&observe_hits, registry_ptr);

  g_instruments.misses = meter->CreateInt64ObservableCounter(
      "kernellake.storage.cache.misses", "Cumulative NVMe cache misses (a new entry had to be populated)",
      "{miss}");
  g_instruments.misses->AddCallback(&observe_misses, registry_ptr);

  g_instruments.evictions = meter->CreateInt64ObservableCounter(
      "kernellake.storage.cache.evictions",
      "Cumulative NVMe cache entries evicted to stay under max_size_bytes", "{eviction}");
  g_instruments.evictions->AddCallback(&observe_evictions, registry_ptr);

  g_instruments.current_bytes =
      meter->CreateInt64ObservableGauge("kernellake.storage.cache.current_bytes",
                                        "Current total bytes cached on the local NVMe cache directory", "By");
  g_instruments.current_bytes->AddCallback(&observe_current_bytes, registry_ptr);

  g_instruments.current_entries = meter->CreateInt64ObservableGauge(
      "kernellake.storage.cache.current_entries",
      "Current number of objects cached on the local NVMe cache directory", "{entry}");
  g_instruments.current_entries->AddCallback(&observe_current_entries, registry_ptr);
}

}  // namespace kernellake
