#pragma once

#include <cstdint>
#include <string>

namespace kernellake {

struct EngineSection {
  int device_id = 0;
  std::uint64_t batch_rows = 1'000'000;
  std::uint64_t result_batch_rows = 65'536;
  std::uint64_t query_memory_limit_bytes = 8ULL * 1024 * 1024 * 1024;
  // "gpu" (default) or "cpu" -- see docs/ARCHITECTURE.md's CPU backend
  // section. "gpu" on a CPU-only (dev preset) build throws the existing
  // clear ExecutionError explaining why, exactly as it always has;
  // requesting "cpu" works in *either* build, since the Acero-based CPU
  // backend needs no CUDA at all. `kernellake query --backend cpu|gpu`
  // overrides this per invocation without editing the config file.
  std::string backend = "gpu";
};

struct MemorySection {
  std::uint64_t pool_initial_bytes = 1ULL * 1024 * 1024 * 1024;
  std::uint64_t pool_max_bytes = 8ULL * 1024 * 1024 * 1024;
  bool use_async_allocator = true;
};

struct StorageSection {
  std::string local_root = "/";
  bool enable_s3 = false;
};

struct LoggingSection {
  std::string level = "info";
  bool json = false;
};

struct ProfilingSection {
  bool nvtx = true;
  bool operator_metrics = true;
};

struct BenchmarkSection {
  int default_iterations = 5;
  int warmup_iterations = 1;
  std::string output_format = "json";
  bool verify_results = true;
  std::string baseline = "duckdb";
};

// Only consumed by kernellake-server (KERNELLAKE_BUILD_SERVER); present
// unconditionally here like every other section so EngineConfig has one
// shape regardless of build options.
struct ServerSection {
  std::string host = "0.0.0.0";
  std::uint16_t port = 31337;
};

// Batch export tuning shared by the trace and log OTLP/gRPC processors
// (metrics have no Simple/Batch processor choice in the OTel SDK -- they
// always read on a timer, tuned by MetricExportConfig below instead).
// Field names/defaults mirror opentelemetry-cpp's own
// BatchSpanProcessorOptions/BatchLogRecordProcessorOptions exactly.
struct BatchExportConfig {
  std::uint32_t max_queue_size = 2048;
  // Must be <= max_queue_size (validate_config() checks this).
  std::uint32_t max_export_batch_size = 512;
  std::uint32_t schedule_delay_ms = 5000;
};

// "simple" (export synchronously on every span End() -- useful for
// debugging, higher overhead) or "batch" (default; buffers and exports on
// an interval via `batch` below).
struct TraceExportConfig {
  std::string processor = "batch";
  BatchExportConfig batch;
  // "default" (the OTel SDK's own default -- ParentBased(AlwaysOn), i.e.
  // sample unless an incoming parent context says not to), "always"
  // (AlwaysOnSampler -- sample every span unconditionally), or "never"
  // (AlwaysOffSampler -- create only non-recording spans; QuerySpan::finish/
  // finish_error still run, they just have nothing to export). No ratio-
  // based sampling knob yet -- these three cover the common cases.
  std::string sampler = "default";
};

// Same processor/batch shape as TraceExportConfig, for the log signal.
struct LogExportConfig {
  std::string processor = "batch";
  BatchExportConfig batch;
};

// Metrics always use a PeriodicExportingMetricReader (no simple/batch
// choice) -- tuned by these two fields instead.
struct MetricExportConfig {
  std::uint32_t export_interval_ms = 60000;
  std::uint32_t export_timeout_ms = 30000;
};

// Only takes effect when built with KERNELLAKE_ENABLE_OTEL (default OFF);
// present unconditionally here like every other section so EngineConfig has
// one shape regardless of build options. `enabled` defaults to false so
// users without a running OTLP collector don't get connection-refused
// noise (see docs/ARCHITECTURE.md's Ubuntu 26.04 baseline section, whose
// own otel-cpp verification saw exactly that -- expected, not a build
// failure -- when nothing was listening). `tracing`/`metrics`/`logs` are
// named distinctly from the existing top-level LoggingSection (spdlog's own
// console level/pattern, unrelated to OTel export) to avoid confusion --
// "observability.logs", not "observability.logging".
struct ObservabilitySection {
  bool enabled = false;
  // "grpc" (default) or "http".
  std::string otlp_protocol = "grpc";
  // For otlp_protocol == "grpc": a host:port target string (e.g.
  // "localhost:4317"), gRPC's own convention -- no scheme prefix, no path
  // (one port multiplexes all three signals as separate gRPC services).
  // For otlp_protocol == "http": the *base* URL, e.g. "http://localhost:4318"
  // or "https://collector.example.com:4318" -- the scheme controls TLS for
  // HTTP (use_tls below does not apply to HTTP; see its own comment).
  // kernellake appends the OTLP spec's own per-signal path itself
  // ("/v1/traces", "/v1/metrics", "/v1/logs") -- do not include it here.
  std::string otlp_endpoint = "localhost:4317";
  std::string service_name = "kernellake";
  // Server-CA verification. gRPC only: sets grpc::SslCredentials, via
  // OtlpGrpcClientOptions::use_ssl_credentials -- ignored for
  // otlp_protocol == "http", where TLS is instead selected by
  // otlp_endpoint's own "http://" vs "https://" scheme.
  bool use_tls = false;
  // Empty string means "use the default trust store" (gRPC) -- only
  // meaningful when use_tls is true (grpc) or the endpoint uses "https://"
  // (http).
  std::string tls_ca_cert_path;
  // HTTP-only client-certificate mTLS. Silently ignored for
  // otlp_protocol == "grpc": opentelemetry-cpp-dev's gRPC mTLS fields are
  // compiled out behind ENABLE_OTLP_GRPC_SSL_MTLS_PREVIEW, which this apt
  // package does not define, but the HTTP exporter's mTLS fields
  // (ssl_client_cert_path/ssl_client_key_path) are unconditionally present
  // -- confirmed by inspecting the installed otlp_http_*_options.h headers,
  // not assumed from the gRPC case.
  std::string tls_client_cert_path;
  std::string tls_client_key_path;
  TraceExportConfig tracing;
  MetricExportConfig metrics;
  LogExportConfig logs;
};

struct EngineConfig {
  EngineSection engine;
  MemorySection memory;
  StorageSection storage;
  LoggingSection logging;
  ProfilingSection profiling;
  BenchmarkSection benchmark;
  ServerSection server;
  ObservabilitySection observability;
};

// Returns the built-in defaults, matching config/kernellake.yaml.
[[nodiscard]] EngineConfig default_config();

// Parses YAML text into an EngineConfig. Missing keys fall back to defaults.
// Throws ConfigurationError on malformed YAML or wrong value types.
[[nodiscard]] EngineConfig parse_config(const std::string& yaml_text);

// Loads and parses a YAML config file. Throws ConfigurationError if the file
// is missing, unreadable, or fails to parse.
[[nodiscard]] EngineConfig load_config_file(const std::string& path);

// Validates value ranges and known enumerations (log level, benchmark output
// format, benchmark baseline) and cross-field constraints (e.g. pool_max
// must be >= pool_initial). Does not validate GPU device availability here;
// CUDA device-count checks happen where a CUDA context is actually created,
// since this validation must also run in CPU-only builds.
// Throws ConfigurationError with an actionable message on the first
// violation found.
void validate_config(const EngineConfig& config);

}  // namespace kernellake
