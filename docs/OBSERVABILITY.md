# Observability

KernelLake's telemetry is OpenTelemetry-based, built behind the
`KERNELLAKE_ENABLE_OTEL` CMake option (default `OFF`; the `otel-dev` preset
turns it on). When built without it, every call in this document's API is a
real, zero-cost no-op -- `query_tracing_stub.cpp` provides the same public
surface as the real implementation, so no call site needs an `#ifdef`.

This document lists every metric/span/log signal KernelLake actually emits
today, the config schema that controls export, and how to point it at a
real collector. For the module's internal design (file layout, why it's
split the way it is, the real bugs found implementing it), see
`docs/ARCHITECTURE.md`'s "OpenTelemetry observability" section -- this
document is the operator-facing reference, that one is the implementation
writeup.

**Scope, as shipped**: one span per whole query, one histogram metric, and
every existing `spdlog` log line bridged into OTel's Logs signal. There are
no per-operator spans (no "this is the Parquet scan step, this is the hash
join step" breakdown) -- see "Not yet done" below.

## 1. Enabling it

Build with the CMake option on:

```bash
cmake --preset otel-dev   # -DKERNELLAKE_ENABLE_OTEL=ON, CPU-only
# or, for a GPU build:
cmake --preset gpu-dev -DKERNELLAKE_ENABLE_OTEL=ON
```

`docker/Dockerfile`'s `dev-cpu`/`dev-gpu` stages (and therefore
`runtime-cpu`/`runtime-gpu`) already build with `KERNELLAKE_ENABLE_OTEL=ON`,
so the published images support this out of the box -- you only need the
config below to actually turn it on.

Then set `observability.enabled: true` in `config/kernellake.yaml` (or
whatever `--config` path you pass) and point `otlp_endpoint` at a real
collector. Nothing is exported with `enabled: false` (the default) --
`kernellake query`/`kernellake-server` run exactly as if `KERNELLAKE_ENABLE_OTEL`
were off, no connection attempts, no `spdlog` output about it.

## 2. What gets emitted

### 2.1 Traces: one span per query

Two call sites, one span each, both named for what they wrap:

| Where | Span name |
| --- | --- |
| `kernellake query` (CLI) | `kernellake.query` |
| `kernellake-server`'s `GetFlightInfoStatement` (Arrow Flight SQL) | `kernellake.flight_sql.get_flight_info_statement` |

Every span carries `kernellake.backend` (`"cpu"` or `"gpu"`) and
`kernellake.sql` (the query text, truncated to 4096 bytes so one very large
generated query can't balloon export payload size). On success, the span
also gets one attribute per `QueryResult` field that was actually measured
for that query -- **not** a fixed attribute set; a field that
`QueryResult` left unset (e.g. `peak_gpu_memory_bytes` on the CPU backend,
which doesn't touch the GPU) is simply absent from the span, never a
zero or null placeholder:

| Span attribute | Source `QueryResult` field |
| --- | --- |
| `kernellake.rows_returned` | `rows_returned` |
| `kernellake.rows_scanned` | `rows_scanned` |
| `kernellake.files_considered` | `files_considered` |
| `kernellake.files_scanned` | `files_scanned` |
| `kernellake.row_groups_considered` | `row_groups_considered` |
| `kernellake.row_groups_scanned` | `row_groups_scanned` |
| `kernellake.compressed_bytes_read` | `compressed_bytes_read` |
| `kernellake.parquet_decoding_seconds` | `parquet_decoding_seconds` |
| `kernellake.gpu_execution_seconds` | `gpu_execution_seconds` |
| `kernellake.cpu_execution_seconds` | `cpu_execution_seconds` |
| `kernellake.host_to_device_seconds` | `host_to_device_seconds` |
| `kernellake.device_to_host_seconds` | `device_to_host_seconds` |
| `kernellake.peak_gpu_memory_bytes` | `peak_gpu_memory_bytes` |
| `kernellake.elapsed_wall_seconds` | `elapsed_wall_seconds` |

Two `QueryResult` fields are **not** exported as span attributes today
(nothing wrong with them, they're just not wired up):
`estimated_uncompressed_bytes` and `metadata_inspection_seconds`.

On failure (`QuerySpan::finish_error`), the span gets `kernellake.backend`/
`kernellake.sql` plus an `exception` event (`exception.message` = the
caught `std::exception::what()`) and status `ERROR` with that same message
as the status description -- no `QueryResult` fields, since there isn't one
to read from. A span that's neither `finish()`ed nor `finish_error()`d
(shouldn't happen in practice, but the destructor is a safety net) ends
unfinished with status `Unset`.

### 2.2 Metrics: one histogram

`kernellake.query.duration_seconds` -- a `Double` histogram, one
observation per successfully-finished query (`elapsed_wall_seconds`),
tagged with a `kernellake.backend` attribute (`"cpu"`/`"gpu"`). Nothing is
recorded for a failed query (`finish_error()` doesn't touch the histogram,
matching the "no `elapsed_wall_seconds` to record" reasoning above).

Anything you want beyond query duration and GPU memory (below) currently
has to come from the span attributes (via a trace-to-metrics pipeline in
your collector, e.g. Grafana Tempo's span metrics processor) rather than a
native KernelLake metric -- no per-query-type counters, no gauge for
in-flight queries.

### 2.2.1 Metrics: GPU memory (process/device level)

GPU-build-only (`KERNELLAKE_WITH_CUDA=ON`; see
`src/memory/gpu_memory_metrics.cpp`/`gpu_memory_metrics_otel.cpp`).
Instrumented at the RMM memory-resource layer
(`TrackingMemoryResource`, the outermost layer of `RmmEnvironment`'s
resource stack -- see that file's own comment) rather than in any
individual operator, so every GPU allocation contributes automatically:

| Metric (OTel name) | Prometheus name | Instrument | Unit | Meaning |
|---|---|---|---|---|
| `kernellake.gpu.memory.allocated` | `kernellake_gpu_memory_allocated_bytes` | ObservableGauge | `By` | Current live GPU bytes allocated |
| `kernellake.gpu.memory.peak` | `kernellake_gpu_memory_peak_bytes` | ObservableGauge | `By` | Highest `allocated` value observed since process startup |
| `kernellake.gpu.memory.allocations` | `kernellake_gpu_memory_allocations_total` | ObservableCounter | `{allocation}` | Cumulative successful allocation count |
| `kernellake.gpu.memory.deallocations` | `kernellake_gpu_memory_deallocations_total` | ObservableCounter | `{allocation}` | Cumulative deallocation count |
| `kernellake.gpu.memory.allocated_total` | `kernellake_gpu_memory_allocated_bytes_total` | ObservableCounter | `By` | Cumulative bytes ever successfully allocated (never decreases, even as `allocated` drops back to 0) |
| `kernellake.gpu.memory.allocation_failures` | `kernellake_gpu_memory_allocation_failures_total` | ObservableCounter | `{allocation}` | Cumulative failed allocation attempts -- includes both genuine CUDA/RMM OOM *and* `engine.query_memory_limit_bytes` rejections (`TrackingMemoryResource` sits outside the limiter, so it sees both) |

Prometheus name verified for real (`benchmarks/local/`'s stack, real GPU
hardware, real query traffic -- not read from the OTel spec alone): the
Collector's Prometheus exporter renders the raw dotted OTel name with
underscores, inserts a suffix derived from the unit (`By` -> `_bytes`),
then -- for a monotonic counter only -- strips any trailing `_total`
already in the name and re-appends exactly one `_total` at the very end.
That's why `allocated_total` (which already ends in `_total`) still comes
out as `..._allocated_bytes_total`, not `..._allocated_total_bytes` or a
doubled `..._total_total` -- the unit suffix always lands *before* the
counter suffix, regardless of where `_total` sat in the original name.
`gpu.device.id` becomes the Prometheus label `gpu_device_id` (dots
sanitize to underscores in attribute names too).

Every series carries a `gpu.device.id` attribute (int) -- never a query
ID, request ID, trace ID, or SQL text: those are unbounded-cardinality
identifiers that belong on spans (see the per-query note below), not on a
metric dimension a collector/backend has to keep one active series per
value of forever.

**Why ObservableCounter, not a synchronous `Counter`:** a synchronous
counter would mean calling into the OTel SDK from inside
`TrackingMemoryResource::allocate()` -- i.e. the GPU allocation hot path,
on every single query. Instead, allocation only ever touches a
process-wide, per-device `std::atomic` struct
(`GpuMemoryMetricsRegistry`); the OTel SDK reads a snapshot of it from its
own periodic-export callback, on its own thread, at
`observability.metrics.export_interval_ms` cadence -- no network activity
and no SDK call ever happens on the allocation path itself.

**Per-query GPU stats are not a metric dimension.** `QueryResult::
peak_gpu_memory_bytes` (already a span attribute -- see 2.1 above) remains
the way to get one query's own GPU memory story; these process-level
metrics answer a different question ("how is the GPU doing right now/over
time") than a per-query breakdown does. Wiring `GpuMemoryMetricsRegistry`
into a per-query delta (mirroring how `RmmEnvironment::track_query()`
already isolates `statistics_resource_adaptor` counters per query) is a
natural extension if a genuine need for it shows up, not implemented here.

**When telemetry is disabled, or this isn't an OTel build:** allocation
behaves identically either way -- registration
(`register_gpu_memory_otel_instruments()`) is independent of GPU
allocation itself (see that function's own comment), and the
`KERNELLAKE_ENABLE_OTEL=OFF` stub is a plain no-op, mirroring
`query_tracing_stub.cpp`'s own pattern.

### 2.3 Logs: every existing `spdlog` call, for free

`OtelSpdlogSink` (`src/observability/internal.hpp`) is pushed onto
`spdlog::default_logger()`'s sink list when observability is enabled,
alongside whatever console sink `init_logging()` already configured --
console output is unaffected, log lines are just *also* exported as OTel
Logs signal records. This means **every** `spdlog::info/warn/error/...`
call anywhere in the codebase is exported, not a curated subset -- no
existing call site needed to change to make this work. Severity maps
directly (`spdlog::level::info` -> `Severity::kInfo`, `warn` -> `kWarn`,
`err` -> `kError`, `critical` -> `kFatal`, etc.).

## 3. Config reference (`observability.*`)

```yaml
observability:
  enabled: false               # must be true to export anything at all
  otlp_protocol: grpc          # grpc | http
  otlp_endpoint: localhost:4317
  service_name: kernellake
  use_tls: false                # gRPC server-CA verification only
  tls_ca_cert_path: ""           # empty = default trust store
  tls_client_cert_path: ""       # HTTP-only mTLS (see below)
  tls_client_key_path: ""
  tracing:
    processor: batch            # simple | batch
    batch:
      max_queue_size: 2048
      max_export_batch_size: 512   # must be <= max_queue_size
      schedule_delay_ms: 5000
    sampler: default             # default | always | never
  metrics:
    export_interval_ms: 60000
    export_timeout_ms: 30000  # must be < export_interval_ms -- see MetricExportConfig's own comment
                              # (config.hpp) for what happens if you get this wrong (not an error --
                              # silently falls back to the SDK's own 60000/30000 defaults instead)
  logs:
    processor: batch            # simple | batch
    batch:
      max_queue_size: 2048
      max_export_batch_size: 512
      schedule_delay_ms: 5000
```

Field-by-field notes worth knowing before you tune these:

- **`otlp_protocol`**: `"grpc"` (default) uses a single `host:port` target
  (gRPC's own convention -- no scheme, no path; one port multiplexes all
  three signals as separate gRPC services). `"http"` uses a *base* URL
  (`http://host:4318` or `https://host:4318`); KernelLake appends the OTLP
  spec's own per-signal path itself (`/v1/traces`, `/v1/metrics`,
  `/v1/logs`) -- **do not include it in `otlp_endpoint`**, or you'll get a
  double-suffixed URL and a 404.
- **TLS**: for gRPC, `use_tls`/`tls_ca_cert_path` control server-CA
  verification; for HTTP, TLS is instead selected by `otlp_endpoint`'s own
  `http://` vs. `https://` scheme (`use_tls` is ignored for HTTP).
  `tls_client_cert_path`/`tls_client_key_path` (client-certificate mTLS)
  only take effect for HTTP -- this apt package's gRPC exporter has its
  mTLS fields compiled out (gated behind
  `ENABLE_OTLP_GRPC_SSL_MTLS_PREVIEW`, which it doesn't define), so those
  two fields are silently ignored under `otlp_protocol: grpc`.
- **`tracing.sampler`**: `"default"` is `ParentBased(AlwaysOn)` (the OTel
  SDK's own recommended default -- sample unless an incoming parent
  context says not to), `"always"` samples every span unconditionally,
  `"never"` creates only non-recording spans (`QuerySpan::finish`/
  `finish_error` still run and still do their normal work, they just have
  nothing to actually export). There's no ratio-based/probabilistic
  sampling knob yet -- these three cover the common cases.
  `*.processor: simple` exports synchronously on every span/log `End()`
  (useful for debugging, higher overhead); `"batch"` (default) buffers and
  exports on the `batch.schedule_delay_ms` interval, capped by
  `batch.max_queue_size`/`max_export_batch_size`. Metrics have no
  simple/batch choice -- always a `PeriodicExportingMetricReader`, tuned
  by `metrics.export_interval_ms`/`export_timeout_ms` instead.
- Named `observability.tracing`/`.metrics`/`.logs`, not `.logging`, to
  avoid confusion with the separate top-level `logging:` section (that one
  is spdlog's own console level/pattern -- unrelated to OTel export; you
  can have console logging on with OTel export off, or vice versa).

## 4. Trying it against a real collector

The fastest way to see real spans end to end is Jaeger's all-in-one image
(this is exactly how the OTel integration itself was verified -- see
`docs/ARCHITECTURE.md`):

```bash
docker run --rm -d --name jaeger \
  -p 4317:4317 -p 4318:4318 -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

Point `config/kernellake.yaml` at it:

```yaml
observability:
  enabled: true
  otlp_protocol: grpc      # or http, against port 4318 instead
  otlp_endpoint: localhost:4317
  service_name: kernellake
```

Run a query, then open `http://localhost:16686` and look for the
`kernellake` service:

```bash
./build/otel-dev/src/cli/kernellake query \
  --sql "SELECT region, SUM(amount) FROM read_parquet('/path/*.parquet') GROUP BY region"
```

**A real caveat, not hypothetical**: Jaeger's all-in-one image only
implements the OTLP TraceService. Pointing the `metrics`/`logs` signals at
it produces a clean, expected failure (`unknown service
opentelemetry.proto.collector.{metrics,logs}.v1.*Service` over gRPC, HTTP
404 over HTTP) -- that's a fact about Jaeger, not a bug in KernelLake's
integration. For metrics/logs, use a full collector (the OpenTelemetry
Collector itself, or a backend like Grafana's stack) instead.

## 5. Verifying it without a collector

`tests/unit/query_tracing_test.cpp` exercises the real implementation
against in-memory exporters (`InMemorySpanExporter`/
`InMemoryMetricExporter`/a small custom `LogRecordExporter`, swapped in via
a test-only `init_for_testing()` seam) -- no network involved, only built
under `otel-dev`/other `KERNELLAKE_ENABLE_OTEL=ON` presets:

- `QueryTracingTest.FinishRecordsSpanAndHistogram` -- a successful query
  produces a span with the expected attributes and a histogram
  observation.
- `QueryTracingTest.FinishErrorSetsErrorStatus` -- a failed query produces
  an `ERROR`-status span with the exception message, no histogram
  observation.
- `QueryTracingTest.SpdlogCallsAreBridgedToOtelLogs` -- a plain
  `spdlog::info()` call is actually received as an OTel log record via the
  bridge sink.

```bash
cmake --preset otel-dev && cmake --build --preset otel-dev
ctest --preset otel-dev --output-on-failure -R QueryTracingTest
```

## 6. Kubernetes / Helm

`charts/kernellake/values.yaml`'s `observability.*` maps directly onto
`ObservabilitySection`'s top-level fields (`enabled`, `otlpProtocol`,
`otlpEndpoint`, `serviceName`, `useTls`) -- see
`charts/kernellake/README.md` for the exact key names. Per-signal
processor/batch/sampler tuning (`tracing.batch.*`, `logs.batch.*`,
`metrics.export_interval_ms`, etc.) is **not** yet exposed through Helm
values; those stay at KernelLake's own compiled-in defaults for a Helm
deployment today. Only takes effect if the deployed image was built with
`KERNELLAKE_ENABLE_OTEL=ON` (both `runtime-cpu` and `runtime-gpu` are, per
`docker/Dockerfile`).

## 7. Not yet done

- **Per-operator spans.** Today's tracing is whole-query only -- there's
  no span for "this is the Parquet scan," "this is the hash join," etc.
  Explicitly deferred, not attempted, per `docs/ROADMAP.md`.
- **No metric beyond query duration and GPU memory (2.2.1).** No
  per-query-type counters, no gauge for in-flight queries, no
  queue-depth metric.
- **No per-query GPU memory metric** (as opposed to the process/device-
  level one in 2.2.1) -- `QueryResult::peak_gpu_memory_bytes` on the span
  is the only per-query GPU memory signal today.
- **No pool-capacity/used/free metrics.** KernelLake's default allocator
  (`memory.use_async_allocator: true`, `cuda_async_memory_resource`) has
  no fixed "capacity" to report -- it's a CUDA stream-ordered pool that
  grows on demand (see `docs/GPU_OPTIMIZATIONS.md`'s own note on this
  allocator's caching behavior). The `pool_memory_resource` alternative
  (`use_async_allocator: false`) does have a real fixed
  `pool_initial_bytes`/`pool_max_bytes` ceiling that could be exposed
  faithfully, but isn't yet -- not invented for the async default, where
  it wouldn't mean anything real.
- **No ratio-based trace sampling.** Only `default`/`always`/`never`.
- **CI coverage** for `KERNELLAKE_ENABLE_OTEL=ON` exists (`otel-build-test`
  in `.github/workflows/ci.yml`, mirroring `server-build-test`'s
  `container: ubuntu:26.04` structure) but only builds/tests/lints the
  code -- it does not spin up a collector, so the Jaeger-based manual
  verification in section 4 above has no CI equivalent today.
