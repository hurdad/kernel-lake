#pragma once

#include <arrow/filesystem/azurefs.h>
#include <arrow/filesystem/gcsfs.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/filesystem/s3fs.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace kernellake {

struct EngineSection {
  int device_id = 0;
  std::uint64_t batch_rows = 1'000'000;
  std::uint64_t result_batch_rows = 65'536;
  // 0 (the default) means "auto-detect": resolve_query_memory_limit_bytes()
  // (kernellake/memory/rmm_environment.hpp) sizes it as a fraction of the
  // GPU's currently *free* VRAM (see that function's own comment for why
  // free, not total, and why 75%), queried fresh via cudaMemGetInfo()
  // rather than baked in at config-load time -- a fixed absolute default
  // here would either waste most of a large card's VRAM or, as happened
  // for real this session (a 3-way join that needed the inherited 8 GiB
  // default manually raised to 12 GiB on a 16 GiB card just to complete),
  // be too tight on anything bigger than the card this default happened
  // to be tuned against. An explicit non-zero value here always overrides
  // auto-detection.
  std::uint64_t query_memory_limit_bytes = 0;
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

// Each of these four sections embeds Arrow's own filesystem options struct
// directly (arrow::fs::{S3,Gcs,Azure,Hdfs}Options) rather than hand-copying
// its field list into a parallel kernellake type -- every plain-data field
// (region, endpoint_override, scheme, timeouts, proxy_options, TLS paths,
// bucket-creation toggles, background_writes, HDFS's connection_config,
// etc.) is set directly on `options` and can never drift out of sync with
// Arrow's own struct, since it *is* Arrow's own struct. A backend is never
// "enabled" by a flag here -- ObjectStoreRegistry (src/storage) constructs
// it lazily, the first time a query actually references a matching URI
// scheme, and these sections only supply the settings used at that point.
//
// The one thing that can't be embedded this way is *credential* material:
// Arrow deliberately keeps credential state behind private fields, settable
// only through each Options type's own factory/Configure*() methods
// (S3Options::Anonymous()/Defaults()/FromAccessKey()/FromAssumeRole()/
// FromAssumeRoleWithWebIdentity(); GcsOptions::Anonymous()/FromAccessToken()/
// FromServiceAccountCredentials(); AzureOptions::ConfigureAnonymousCredential()/
// ConfigureAccountKeyCredential()/ConfigureSASCredential()/
// ConfigureClientSecretCredential()/etc.) -- there is no `options.access_key`
// field to just set. Each section below therefore keeps a small
// `credentials_kind` selector plus the raw material for whichever kind it
// names; the backend implementation (src/storage) calls the matching
// factory/Configure*() method at construction time. HDFS has no such
// split -- its connection_config (host/port/user/kerb_ticket/extra_conf) is
// plain public data on Arrow's own struct, same as everything else.

// arrow::fs::S3Options has no access-key/secret-key fields at all (by
// design -- see credentials_kind below), so "explicit" mode reads
// AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN from the
// environment at ObjectStore-construction time instead of from this config.
struct S3Section {
  arrow::fs::S3Options options;
  // "anonymous" | "default" | "explicit" | "role" | "web_identity" (mirrors
  // S3CredentialsKind). "role" uses options.role_arn/session_name/
  // external_id/load_frequency (all plain fields on S3Options already);
  // "web_identity" uses the AWS SDK's own AssumeRoleWithWebIdentity env
  // vars (AWS_WEB_IDENTITY_TOKEN_FILE, etc.), nothing further to configure.
  std::string credentials_kind = "default";
};

struct GcsSection {
  arrow::fs::GcsOptions options;
  // "anonymous" | "access_token" | "service_account_json" | "default".
  std::string credentials_kind = "default";
  std::string access_token;             // for "access_token"
  std::string access_token_expiration;  // ISO-8601; pairs with access_token
  std::string target_service_account;   // optional impersonation target
  std::string json_credentials;         // inline JSON content, not a path; for "service_account_json"
};

struct AzureSection {
  arrow::fs::AzureOptions options;
  // "default" | "anonymous" | "storage_shared_key" | "sas_token" |
  // "client_secret" | "managed_identity" | "cli" | "workload_identity" |
  // "environment" (mirrors AzureCredentialKind).
  std::string credentials_kind = "default";
  std::string storage_shared_key;  // for "storage_shared_key"
  std::string sas_token;           // for "sas_token"
  std::string tenant_id;           // for "client_secret"
  std::string client_id;           // for "client_secret" / "managed_identity" / "workload_identity"
  std::string client_secret;       // for "client_secret"
};

// HdfsObjectStore (src/storage/hdfs_object_store.cpp) compiles and links
// cleanly with no Hadoop installed at all -- arrow::fs::HadoopFileSystem
// dlopen()s libhdfs.so lazily at runtime, not a build-time link dependency
// -- but unlike S3/GCS/Azure, this project has no way to *run* it against
// a real cluster: there's no lightweight single-container emulator the
// way MinIO/fake-gcs-server/Azurite give the other three backends, and
// this project's own development sandbox has no JDK/Hadoop installation
// either. See docs/ARCHITECTURE.md's "Cloud object storage" section.
struct HdfsSection {
  // arrow::fs::HdfsOptions's default constructor leaves
  // options.connection_config.port genuinely uninitialized -- unlike every
  // other field on HdfsOptions/HdfsConnectionConfig, arrow::io::
  // HdfsConnectionConfig::port has no default member initializer of its
  // own (confirmed directly against /usr/include/arrow/io/hdfs.h), so a
  // default-constructed HdfsOptions carries indeterminate value there.
  // Fixed here so a default HdfsSection is never a source of read-
  // uninitialized-memory (e.g. as parse_config()'s fallback default when
  // storage.hdfs.connection_config.port isn't present in the YAML).
  arrow::fs::HdfsOptions options{[] {
    arrow::fs::HdfsOptions opts;
    opts.connection_config.port = 0;
    return opts;
  }()};
};

struct StorageSection {
  std::string local_root = "/";
  S3Section s3;
  GcsSection gcs;
  AzureSection azure;
  HdfsSection hdfs;
};

// One named Iceberg REST catalog (see src/iceberg/rest_catalog_client.cpp).
// `read_iceberg('catalog.namespace.table')`'s leading `catalog` component
// looks this up by name -- a real deployment commonly has more than one
// (e.g. "prod"/"staging"), so this is a map, not a single section, unlike
// S3/GCS/Azure/HDFS above (KernelLake talks to exactly one of each of
// those per query, chosen by URI scheme, never by a name the query spells
// out itself).
struct IcebergCatalogSection {
  std::string catalog_uri;  // e.g. "http://localhost:8181"
  std::string warehouse;    // optional; some REST catalog servers require it
  std::string prefix;       // REST catalog API's "{prefix}" path segment; "" (root) by default
  // "none" | "bearer_token" | "oauth2_client_credentials" (mirrors the
  // credentials_kind convention already used by S3Section/GcsSection/etc.
  // above). "bearer_token" covers a pre-obtained static token (the common
  // case for Polaris/Nessie deployments that front their own auth);
  // "oauth2_client_credentials" performs the REST Catalog spec's own
  // POST /v1/oauth/tokens client_credentials flow and refreshes the token
  // as it nears expiry.
  std::string credentials_kind = "none";
  std::string bearer_token;         // for "bearer_token"
  std::string oauth2_client_id;     // for "oauth2_client_credentials"
  std::string oauth2_client_secret;  // for "oauth2_client_credentials"
  std::string oauth2_scope;         // optional; for "oauth2_client_credentials"
};

struct IcebergSection {
  std::unordered_map<std::string, IcebergCatalogSection> catalogs;
};

// Delta Lake read support (src/delta/) talks to a separate, standalone
// delta-txn-service (a Rust gRPC service -- deliberately kept out of this
// project's own build, see cmake/ThirdPartyDeltaTxnProto.cmake) as a
// client, for both table inspection (GetTable) and active-file listing
// (ListActiveFiles). Unlike IcebergSection, this is a single section, not
// a name-keyed map: delta-txn-service's own README describes it as "a
// centralized Delta commit coordinator" -- one deployment per environment,
// not one per catalog the way Iceberg REST catalogs commonly are -- and a
// Delta table is addressed directly by its storage URI
// (`read_delta('s3://bucket/table')`), with no catalog/namespace concept
// of its own to key a map by. `grpc_endpoint` empty means "not
// configured": validate_config() below only requires it be checked at the
// point read_delta(...) is actually used, matching how storage.s3/etc.
// aren't required just because they're present in the schema.
struct DeltaSection {
  std::string grpc_endpoint;  // e.g. "delta-txn.internal:50051"; empty = not configured
  bool use_tls = false;
  std::string tls_ca_cert_path;  // optional; empty uses the system default trust store
  std::string api_key;           // sent as the "x-api-key" gRPC metadata header; empty = no auth
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
  // KernelLakeFlightSqlServer executes a statement eagerly in
  // GetFlightInfoStatement and buffers the whole result in results_ until
  // DoGetStatement fetches it (see that class's own doc comment for why).
  // A client that calls GetFlightInfoStatement repeatedly without ever
  // fetching (or that just disconnects after getting the ticket) would
  // otherwise grow results_ without bound -- this caps how many buffered,
  // not-yet-fetched results the server holds at once; GetFlightInfoStatement
  // rejects new statements past this cap with ExecutionError rather than
  // buffering unboundedly.
  std::uint32_t max_pending_results = 1024;
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
  IcebergSection iceberg;
  DeltaSection delta;
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
