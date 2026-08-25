#pragma once

#include <arrow/filesystem/azurefs.h>
#include <arrow/filesystem/gcsfs.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/filesystem/s3fs.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kernellake {

struct EngineSection {
  // Row-count caps enforced by BatchSizeLimitOperator
  // (kernellake/execution_gpu/batch_size_limit_operator.hpp), forwarded via
  // build_operator_tree() (operator_builder.hpp/.cpp): batch_rows caps
  // each ParquetScanNode's own output batch, result_batch_rows caps the
  // whole query's final output. Both real row-count limits (validated
  // > 0), not "0 means unlimited" sentinels. Splitting an oversized batch
  // costs a real device-to-device copy per extra chunk -- see that
  // operator's own doc comment -- so raising either value trades a
  // smaller number of larger host-side Arrow conversions/network
  // messages against more GPU memory held by one batch at a time;
  // lowering it trades the reverse, plus more splitting-copy overhead if
  // batches naturally produced upstream (governed by
  // query_memory_limit_bytes-derived byte budgets, not row counts) often
  // exceed the cap.
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
  // Forwarded to every HashAggregateOperator as its own max_distinct_keys
  // (see that class's own doc comment) -- a fixed safety cap on result
  // cardinality, not a cardinality estimate. 0 (the default) means "use
  // HashAggregateOperator::kDefaultMaxDistinctKeys". This has already
  // needed raising twice as real TPC-H scale factor grew (SF100's real Q3
  // measured ~10.8M distinct groups, SF1000's ~38.9M), each time requiring
  // a code change + rebuild; making it config-driven means the next scale
  // bump is a YAML edit + restart instead.
  std::uint64_t max_distinct_keys = 0;
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
  // Threads backing the AWS Common Runtime's S3 I/O event loop for the
  // whole process (arrow::fs::S3GlobalOptions::num_event_loop_threads,
  // plumbed through s3_object_store.cpp's ensure_s3_initialized() calling
  // arrow::fs::InitializeS3() directly instead of the bare
  // EnsureS3Initialized(), which always uses Arrow's own default of 1).
  // This is a genuinely process-global, initialize-once AWS SDK setting
  // (not per-S3Section/per-filesystem), but there's only one S3Section per
  // running kernellake-server process, so it lives here rather than
  // inventing a separate top-level config section for one field.
  // Arrow's own doc calls 1 "recommended... when the # of connections is
  // expected to be, at most, in the hundreds" -- ObjectStoreDatasource's
  // device_read_async() can burst well past that for a single Parquet scan
  // pass at SF1000 scale (pass_read_limit_bytes ~5GB, many concurrent
  // column-chunk reads), all funneled through this one thread regardless
  // of how many host_read() threads are in flight. Confirmed for real
  // (2026-08-17, real SF1000 Q6, g6.4xlarge, 3 cold reps per config): 1 ->
  // 4 gave a genuine, reproducible ~13% wall-time improvement (mean
  // 201.3s -> 174.7s), well outside both configs' run-to-run stddev.
  // 4 -> 8 (this instance's vCPU count) made no further difference to the
  // mean (176.4s, statistically indistinguishable from 4's 174.7s) -- the
  // benefit plateaus once the event loop has "enough" threads, since the
  // actual work is I/O-wait-bound, not CPU-bound, so matching vCPU count
  // buys nothing further. 4 chosen as the new default over 8 for the same
  // result at lower resource cost (fewer OS threads) -- not because of a
  // variance difference between the two (8's stddev happened to be
  // tighter in this one 3-sample run, 3.2s vs 4's 12.6s, but with only 3
  // reps per config that alone isn't a robust signal either way).
  int s3_event_loop_threads = 4;
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

// Optional local-NVMe read-through cache sitting in front of the *remote*
// (non-"file") ObjectStore backends above -- see docs/ARCHITECTURE.md's
// "NVMe cache tier" section for the full design. Disabled by default:
// existing deployments behave identically until an operator opts in with a
// real local directory. Never applies to plain local paths, which are
// already local -- caching them would just duplicate the same bytes on the
// same disk.
struct CacheSection {
  bool enabled = false;
  // Local filesystem directory the cache lives in; must be non-empty if
  // enabled is true (validated in validate_config()). A plain directory of
  // files, not an in-memory structure, so it survives kernellake-server
  // restarts by construction.
  std::string directory;
  // Total cache size budget in bytes. Once a newly-populated entry would
  // push the directory over this budget, least-recently-used entries (by
  // file mtime, bumped on every cache hit) are evicted until it's back
  // under budget. 0 means unbounded (no eviction) -- matches this project's
  // existing "0 == no limit" convention (see EngineSection::query_memory_limit_bytes).
  std::uint64_t max_size_bytes = 100ULL * 1024 * 1024 * 1024;  // 100 GiB
};

struct StorageSection {
  std::string local_root = "/";
  S3Section s3;
  GcsSection gcs;
  AzureSection azure;
  HdfsSection hdfs;
  CacheSection cache;
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
  std::string bearer_token;          // for "bearer_token"
  std::string oauth2_client_id;      // for "oauth2_client_credentials"
  std::string oauth2_client_secret;  // for "oauth2_client_credentials"
  std::string oauth2_scope;          // optional; for "oauth2_client_credentials"
};

struct IcebergSection {
  std::unordered_map<std::string, IcebergCatalogSection> catalogs;
};

// One named Unity Catalog instance (src/unitycatalog/unity_catalog_client.cpp)
// -- a deployment may talk to more than one workspace/instance, so this is a
// map, keyed the same way IcebergSection::catalogs is.
// `read_unity_catalog('instance.catalog.schema.table')`'s leading component
// looks this up; `catalog.schema.table` is then resolved against Unity
// Catalog's own REST API. Unlike IcebergCatalogSection, the OAuth2 token
// endpoint isn't derivable from `uc_url` (Databricks uses
// "https://<workspace>/oidc/v1/token", a different host/path shape than the
// UC REST API itself; other UC deployments may differ again) -- it's always
// an explicit field, required whenever credentials_kind is
// "oauth2_client_credentials".
struct UnityCatalogInstanceSection {
  std::string uc_url;                 // e.g. "https://<workspace>/api/2.1/unity-catalog"
  std::string oauth2_token_endpoint;  // required for "oauth2_client_credentials"; e.g. ".../oidc/v1/token"
  // "none" | "bearer_token" | "oauth2_client_credentials" (mirrors
  // IcebergCatalogSection::credentials_kind's own convention).
  std::string credentials_kind = "none";
  std::string bearer_token;          // for "bearer_token"
  std::string oauth2_client_id;      // for "oauth2_client_credentials"
  std::string oauth2_client_secret;  // for "oauth2_client_credentials"
  std::string oauth2_scope;          // optional; for "oauth2_client_credentials"
};

struct UnityCatalogSection {
  std::unordered_map<std::string, UnityCatalogInstanceSection> instances;
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

// CLI-only (`kernellake benchmark tpch`) -- see CliConfig below.
// output_format/verify_results/baseline were removed 2026-08-24: all three
// were read from YAML and validated (output_format in {"json","csv"},
// baseline in {"duckdb","none"}) but never actually consumed by
// benchmark_tpch_command.cpp, which unconditionally emits JSON and never
// runs an in-process DuckDB comparison -- that comparison already lives in
// the separate tools/validate_tpch.py, which this command's own output
// tells the caller to run. Dead config from day one of this section,
// found in a docs/config audit; removed rather than wired up, since no
// caller ever needed the feature they implied.
struct BenchmarkSection {
  int default_iterations = 5;
  int warmup_iterations = 1;
};

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

  // Inbound TLS for the Flight SQL listener itself, as opposed to
  // delta.use_tls/observability.use_tls above, which are outbound TLS for
  // this process's own gRPC clients. tls_cert_path/tls_key_path are PEM
  // files read at startup and passed to
  // arrow::flight::FlightServerOptions::tls_certificates; both must be set
  // when use_tls is true (validate_config() below checks this).
  bool use_tls = false;
  std::string tls_cert_path;
  std::string tls_key_path;
  // mTLS: when set, clients must present a certificate signed by the CA in
  // tls_client_ca_cert_path (-> FlightServerOptions::verify_client /
  // root_certificates). Only meaningful when use_tls is also true.
  bool require_client_cert = false;
  std::string tls_client_ca_cert_path;

  // Static bearer-token auth, checked by a ServerMiddleware
  // (src/server/auth_middleware.cpp) against every call's "authorization:
  // Bearer <token>" header. Same shape as delta.api_key above (a single
  // shared secret, not per-client credentials) -- a first pass, not a
  // long-term identity system. Independent of use_tls: sending a bearer
  // token over a plaintext connection defeats its purpose, but that's a
  // deployment-configuration mistake for the operator to avoid, not
  // something validate_config() cross-checks here.
  bool auth_enabled = false;
  std::string auth_token;
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
// choice) -- tuned by these two fields instead. export_timeout_ms must be
// strictly less than export_interval_ms -- confirmed for real:
// opentelemetry-cpp's PeriodicExportingMetricReader does not reject an
// equal-or-greater timeout with an error, it silently falls back to its
// own built-in defaults instead (logged as a startup warning: "Invalid
// configuration: export_timeout_millis_ should be less than
// export_interval_millis_, using default values") -- not validated here,
// since this project's own config.cpp has no reach into
// opentelemetry-cpp's internals to check it ahead of time.
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

// Every setting shared by both kernellake binaries (the CLI and
// kernellake-server): storage/catalog/credential config, memory pool
// tuning, logging, profiling, and observability. Neither binary-specific
// concern lives here -- see CliConfig/ServerConfig below, added 2026-08-24
// specifically so a field meaningful to only one binary can't sit in a
// shared struct where the other binary silently ignores it (the bug
// EngineSection::device_id used to be: post-Tier-1
// (docs/MULTI_GPU_SCALING.md), it was real for the CLI's one-shot queries
// but silently discarded by kernellake-server, which had already moved to
// a per-device model). RmmEnvironment/QueryEngine now take "which device"
// as an explicit constructor parameter instead of reading it from this
// struct -- see their own headers.
struct EngineConfig {
  EngineSection engine;
  MemorySection memory;
  StorageSection storage;
  IcebergSection iceberg;
  UnityCatalogSection unity_catalog;
  DeltaSection delta;
  LoggingSection logging;
  ProfilingSection profiling;
  ObservabilitySection observability;
};

// kernellake CLI's own top-level config: the shared EngineConfig plus the
// two things only a one-shot, single-process CLI invocation needs --
// which single GPU an ad-hoc query runs on (device_id; RmmEnvironment/
// QueryEngine take this as an explicit constructor argument, defaulting to
// 0 if unset -- see their own headers), and `kernellake benchmark tpch`'s
// own tuning (benchmark). kernellake-server never constructs one of these.
struct CliConfig {
  EngineConfig engine_config;
  int device_id = 0;
  BenchmarkSection benchmark;
};

// kernellake-server's own top-level config: the shared EngineConfig plus
// the Flight SQL listener's own settings (server) and how many queries
// GpuExecutionCoordinator lets run concurrently against *each* GPU
// (max_concurrent_gpu_queries -- moved here from EngineSection 2026-08-24,
// since GpuExecutionCoordinator is the only consumer; see its own header
// for why this cap is per-device, not process-wide, since Tier 1). The
// CLI never constructs one of these.
struct ServerConfig {
  EngineConfig engine_config;
  ServerSection server;
  // See EngineSection::max_concurrent_gpu_queries's old comment (now
  // moved here) for the full reasoning behind bounding this at all rather
  // than leaving it unbounded: (1) pass_read_limit_bytes/
  // build_side_budget_bytes (query_engine_execute_gpu.cpp) are each sized
  // as a fraction of one device's entire memory ceiling, so N concurrent
  // queries on the same device can collectively demand up to N times
  // that; (2) the opt #6 parallel-decode prototype
  // (docs/GPU_OPTIMIZATIONS.md) found concurrent decode streams on one
  // GPU degrade past ~N=2. 2 is a conservative starting point pending a
  // real scaling_test.py re-run to tune per-deployment.
  int max_concurrent_gpu_queries = 2;
  // Which CUDA device ordinals GpuExecutionCoordinator builds one
  // RmmEnvironment per and round-robins queries across (Tier 1, see
  // docs/MULTI_GPU_SCALING.md). Empty (the default) means "every device
  // cudaGetDeviceCount() reports" -- Tier 1's original all-devices
  // behavior. A non-empty list pins the server to exactly these ordinals
  // instead, in the given order (which also becomes the round-robin
  // order) -- for a shared box where other workloads need some GPUs left
  // alone, or where an operator wants fewer than every visible device
  // devoted to kernellake-server. GpuExecutionCoordinator validates each
  // entry against the real cudaGetDeviceCount() at construction time
  // (out-of-range or duplicate entries throw ConfigurationError) --
  // range-checking here in validate_server_config() isn't possible
  // without a CUDA context, matching how device availability is never
  // checked at plain config-validation time elsewhere in this file.
  //
  // Ordinals, not UUIDs: every CUDA API this project calls
  // (cudaSetDevice(), cudaMemGetInfo(), etc.) is ordinal-only regardless,
  // so a UUID layer would still need to resolve back to an ordinal before
  // any of it could be used -- extra machinery this project's actual
  // target deployments (single-tenant bare-metal/VM boxes, not GPU-sliced
  // multi-tenant Kubernetes, where a device plugin already remaps
  // whichever physical GPUs a pod gets down to ordinals 0..N-1 before this
  // config is ever read) don't need. The one real gotcha with plain
  // ordinals: CUDA's own default device enumeration order
  // (`CUDA_DEVICE_ORDER=FASTEST_FIRST`) is not guaranteed to match
  // `nvidia-smi`'s (PCI bus ID order) -- an operator cross-referencing
  // `nvidia-smi -L` output to decide which ordinals to reserve here should
  // set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in kernellake-server's environment
  // first, or the numbers can silently disagree.
  std::vector<int> gpu_device_ids;
};

// Returns the built-in defaults for the shared sections, matching the
// sections common to config/kernellake-cli.yaml and
// config/kernellake-server.yaml.
[[nodiscard]] EngineConfig default_config();

// Returns default_config() plus the CLI's own section defaults.
[[nodiscard]] CliConfig default_cli_config();

// Returns default_config() plus the server's own section defaults.
[[nodiscard]] ServerConfig default_server_config();

// Parses YAML text into an EngineConfig -- only the shared sections; a
// server: or benchmark: key present in `yaml_text` is silently ignored
// here (parse_cli_config()/parse_server_config() below read those
// themselves). Missing keys fall back to defaults. Throws
// ConfigurationError on malformed YAML or wrong value types.
[[nodiscard]] EngineConfig parse_config(const std::string& yaml_text);

// parse_config() plus the CLI's own top-level keys (engine.device_id,
// benchmark:). Both binaries can read the exact same YAML file -- this
// simply ignores any server:/max_concurrent_gpu_queries keys present.
[[nodiscard]] CliConfig parse_cli_config(const std::string& yaml_text);

// parse_config() plus the server's own top-level keys (server:,
// engine.max_concurrent_gpu_queries). Ignores any benchmark:/
// engine.device_id keys present.
[[nodiscard]] ServerConfig parse_server_config(const std::string& yaml_text);

// Loads and parses a YAML config file as a CliConfig. Throws
// ConfigurationError if the file is missing, unreadable, or fails to
// parse.
[[nodiscard]] CliConfig load_cli_config_file(const std::string& path);

// Loads and parses a YAML config file as a ServerConfig. Throws
// ConfigurationError if the file is missing, unreadable, or fails to
// parse.
[[nodiscard]] ServerConfig load_server_config_file(const std::string& path);

// Validates the shared sections' value ranges/known enumerations (log
// level, etc.) and cross-field constraints (e.g. pool_max must be >=
// pool_initial). Does not validate GPU device availability here; CUDA
// device-count checks happen where a CUDA context is actually created,
// since this validation must also run in CPU-only builds. Throws
// ConfigurationError with an actionable message on the first violation
// found. Called by both validate_cli_config()/validate_server_config()
// below -- not usually called directly.
void validate_config(const EngineConfig& config);

// validate_config(config.engine_config) plus the CLI-only fields
// (device_id >= 0).
void validate_cli_config(const CliConfig& config);

// validate_config(config.engine_config) plus the server-only fields
// (server.port, TLS/auth cross-field checks, max_concurrent_gpu_queries).
void validate_server_config(const ServerConfig& config);

}  // namespace kernellake
