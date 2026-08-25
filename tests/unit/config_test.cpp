#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

// --- EngineConfig: sections shared by both binaries (see config.hpp's own
// comment on the CLI/server split, added 2026-08-24) ------------------------

TEST(Config, DefaultsMatchSpec) {
  const EngineConfig config = default_config();
  EXPECT_EQ(config.engine.batch_rows, 1'000'000u);
  EXPECT_EQ(config.memory.pool_max_bytes, 8ULL * 1024 * 1024 * 1024);
  EXPECT_NO_THROW((void)(validate_config(config)));
}

TEST(Config, ParsesOverrides) {
  const std::string yaml = R"(
engine:
  batch_rows: 500000
logging:
  level: debug
  json: true
)";
  const EngineConfig config = parse_config(yaml);
  EXPECT_EQ(config.engine.batch_rows, 500000u);
  EXPECT_EQ(config.logging.level, "debug");
  EXPECT_TRUE(config.logging.json);
  // Fields not present in the override fall back to defaults.
  EXPECT_EQ(config.engine.result_batch_rows, default_config().engine.result_batch_rows);
}

// device_id/benchmark: are CLI-only keys (CliConfig, below) -- parse_config()
// doesn't read them at all, so a malformed device_id here is simply never
// looked at, not an error.
TEST(Config, ParseConfigIgnoresCliOnlyKeys) {
  const EngineConfig config = parse_config("engine:\n  device_id: not-a-number\n");
  EXPECT_EQ(config.engine.batch_rows, default_config().engine.batch_rows);
}

TEST(Config, RejectsZeroBatchRows) {
  EngineConfig config = default_config();
  config.engine.batch_rows = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, ParsesS3EventLoopThreads) {
  const std::string yaml = R"(
storage:
  s3:
    event_loop_threads: 4
)";
  const EngineConfig config = parse_config(yaml);
  EXPECT_EQ(config.storage.s3.s3_event_loop_threads, 4);
}

TEST(Config, RejectsZeroS3EventLoopThreads) {
  EngineConfig config = default_config();
  config.storage.s3.s3_event_loop_threads = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsContradictoryMemoryPool) {
  EngineConfig config = default_config();
  config.memory.pool_initial_bytes = 100;
  config.memory.pool_max_bytes = 10;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsUnknownLogLevel) {
  EngineConfig config = default_config();
  config.logging.level = "verbose";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsMalformedYaml) {
  EXPECT_THROW((void)(parse_config("engine: [this is not a map")), ConfigurationError);
}

TEST(Config, ParsesIcebergCatalogs) {
  const std::string yaml = R"(
iceberg:
  catalogs:
    prod:
      catalog_uri: http://localhost:8181
      warehouse: s3://prod-warehouse
      prefix: prod
      credentials_kind: bearer_token
      bearer_token: secret-token
    staging:
      catalog_uri: http://localhost:8182
      credentials_kind: oauth2_client_credentials
      oauth2_client_id: client-id
      oauth2_client_secret: client-secret
      oauth2_scope: catalog
)";
  const EngineConfig config = parse_config(yaml);
  ASSERT_EQ(config.iceberg.catalogs.size(), 2u);

  const IcebergCatalogSection& prod = config.iceberg.catalogs.at("prod");
  EXPECT_EQ(prod.catalog_uri, "http://localhost:8181");
  EXPECT_EQ(prod.warehouse, "s3://prod-warehouse");
  EXPECT_EQ(prod.prefix, "prod");
  EXPECT_EQ(prod.credentials_kind, "bearer_token");
  EXPECT_EQ(prod.bearer_token, "secret-token");

  const IcebergCatalogSection& staging = config.iceberg.catalogs.at("staging");
  EXPECT_EQ(staging.credentials_kind, "oauth2_client_credentials");
  EXPECT_EQ(staging.oauth2_client_id, "client-id");
  EXPECT_EQ(staging.oauth2_client_secret, "client-secret");
  EXPECT_EQ(staging.oauth2_scope, "catalog");

  EXPECT_NO_THROW((void)(validate_config(config)));
}

TEST(Config, RejectsIcebergCatalogMissingUri) {
  EngineConfig config = default_config();
  config.iceberg.catalogs["broken"] = IcebergCatalogSection{};
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsIcebergCatalogUnknownCredentialsKind) {
  EngineConfig config = default_config();
  IcebergCatalogSection catalog;
  catalog.catalog_uri = "http://localhost:8181";
  catalog.credentials_kind = "kerberos";
  config.iceberg.catalogs["broken"] = catalog;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsIcebergCatalogBearerTokenMissingToken) {
  EngineConfig config = default_config();
  IcebergCatalogSection catalog;
  catalog.catalog_uri = "http://localhost:8181";
  catalog.credentials_kind = "bearer_token";
  config.iceberg.catalogs["broken"] = catalog;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsIcebergCatalogOauth2MissingClientCredentials) {
  EngineConfig config = default_config();
  IcebergCatalogSection catalog;
  catalog.catalog_uri = "http://localhost:8181";
  catalog.credentials_kind = "oauth2_client_credentials";
  catalog.oauth2_client_id = "client-id";
  // oauth2_client_secret intentionally left empty.
  config.iceberg.catalogs["broken"] = catalog;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, ParsesUnityCatalogInstances) {
  const std::string yaml = R"(
unity_catalog:
  instances:
    prod:
      uc_url: https://workspace.example.com/api/2.1/unity-catalog
      credentials_kind: bearer_token
      bearer_token: secret-token
    staging:
      uc_url: https://staging.example.com/api/2.1/unity-catalog
      oauth2_token_endpoint: https://staging.example.com/oidc/v1/token
      credentials_kind: oauth2_client_credentials
      oauth2_client_id: client-id
      oauth2_client_secret: client-secret
      oauth2_scope: all-apis
)";
  const EngineConfig config = parse_config(yaml);
  ASSERT_EQ(config.unity_catalog.instances.size(), 2u);

  const UnityCatalogInstanceSection& prod = config.unity_catalog.instances.at("prod");
  EXPECT_EQ(prod.uc_url, "https://workspace.example.com/api/2.1/unity-catalog");
  EXPECT_EQ(prod.credentials_kind, "bearer_token");
  EXPECT_EQ(prod.bearer_token, "secret-token");

  const UnityCatalogInstanceSection& staging = config.unity_catalog.instances.at("staging");
  EXPECT_EQ(staging.oauth2_token_endpoint, "https://staging.example.com/oidc/v1/token");
  EXPECT_EQ(staging.credentials_kind, "oauth2_client_credentials");
  EXPECT_EQ(staging.oauth2_client_id, "client-id");
  EXPECT_EQ(staging.oauth2_client_secret, "client-secret");
  EXPECT_EQ(staging.oauth2_scope, "all-apis");

  EXPECT_NO_THROW((void)(validate_config(config)));
}

TEST(Config, RejectsUnityCatalogInstanceMissingUrl) {
  EngineConfig config = default_config();
  config.unity_catalog.instances["broken"] = UnityCatalogInstanceSection{};
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsUnityCatalogInstanceUnknownCredentialsKind) {
  EngineConfig config = default_config();
  UnityCatalogInstanceSection instance;
  instance.uc_url = "https://workspace.example.com/api/2.1/unity-catalog";
  instance.credentials_kind = "kerberos";
  config.unity_catalog.instances["broken"] = instance;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsUnityCatalogInstanceBearerTokenMissingToken) {
  EngineConfig config = default_config();
  UnityCatalogInstanceSection instance;
  instance.uc_url = "https://workspace.example.com/api/2.1/unity-catalog";
  instance.credentials_kind = "bearer_token";
  config.unity_catalog.instances["broken"] = instance;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsUnityCatalogInstanceOauth2MissingClientCredentials) {
  EngineConfig config = default_config();
  UnityCatalogInstanceSection instance;
  instance.uc_url = "https://workspace.example.com/api/2.1/unity-catalog";
  instance.oauth2_token_endpoint = "https://workspace.example.com/oidc/v1/token";
  instance.credentials_kind = "oauth2_client_credentials";
  instance.oauth2_client_id = "client-id";
  // oauth2_client_secret intentionally left empty.
  config.unity_catalog.instances["broken"] = instance;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsUnityCatalogInstanceOauth2MissingTokenEndpoint) {
  EngineConfig config = default_config();
  UnityCatalogInstanceSection instance;
  instance.uc_url = "https://workspace.example.com/api/2.1/unity-catalog";
  instance.credentials_kind = "oauth2_client_credentials";
  instance.oauth2_client_id = "client-id";
  instance.oauth2_client_secret = "client-secret";
  // oauth2_token_endpoint intentionally left empty.
  config.unity_catalog.instances["broken"] = instance;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, ParsesDeltaSection) {
  const std::string yaml = R"(
delta:
  grpc_endpoint: delta-txn.internal:50051
  use_tls: true
  tls_ca_cert_path: /etc/ssl/ca.pem
  api_key: secret-key
)";
  const EngineConfig config = parse_config(yaml);
  EXPECT_EQ(config.delta.grpc_endpoint, "delta-txn.internal:50051");
  EXPECT_TRUE(config.delta.use_tls);
  EXPECT_EQ(config.delta.tls_ca_cert_path, "/etc/ssl/ca.pem");
  EXPECT_EQ(config.delta.api_key, "secret-key");
}

TEST(Config, DeltaSectionDefaultsToUnconfigured) {
  const EngineConfig config = default_config();
  EXPECT_TRUE(config.delta.grpc_endpoint.empty());
  EXPECT_FALSE(config.delta.use_tls);
  EXPECT_TRUE(config.delta.api_key.empty());
}

// read_string_map() (storage.hdfs.connection_config.extra_conf) had no
// coverage at all, success or failure.
TEST(Config, ParsesHdfsExtraConf) {
  const std::string yaml = R"(
storage:
  hdfs:
    connection_config:
      host: namenode
      extra_conf:
        dfs.client.use.datanode.hostname: "true"
)";
  const EngineConfig config = parse_config(yaml);
  EXPECT_EQ(config.storage.hdfs.options.connection_config.host, "namenode");
  ASSERT_EQ(config.storage.hdfs.options.connection_config.extra_conf.size(), 1u);
  EXPECT_EQ(config.storage.hdfs.options.connection_config.extra_conf.at("dfs.client.use.datanode.hostname"),
            "true");
}

TEST(Config, RejectsMalformedHdfsExtraConfEntry) {
  // A nested sequence as a map value can't convert via .as<std::string>().
  const std::string yaml = R"(
storage:
  hdfs:
    connection_config:
      extra_conf:
        bad_key: [not, a, scalar]
)";
  EXPECT_THROW((void)(parse_config(yaml)), ConfigurationError);
}

// storage.gcs.options.retry_limit_seconds/project_id are std::optional<>,
// read via their own explicit-presence-check branch in
// parse_engine_config_from_root() rather than read_or<T>() -- neither the
// success nor failure path of either had coverage.
TEST(Config, ParsesGcsRetryLimitSecondsAndProjectId) {
  const std::string yaml = R"(
storage:
  gcs:
    retry_limit_seconds: 30.5
    project_id: my-project
)";
  const EngineConfig config = parse_config(yaml);
  ASSERT_TRUE(config.storage.gcs.options.retry_limit_seconds.has_value());
  EXPECT_DOUBLE_EQ(*config.storage.gcs.options.retry_limit_seconds, 30.5);
  ASSERT_TRUE(config.storage.gcs.options.project_id.has_value());
  EXPECT_EQ(*config.storage.gcs.options.project_id, "my-project");
}

TEST(Config, RejectsMalformedGcsRetryLimitSeconds) {
  const std::string yaml = "storage:\n  gcs:\n    retry_limit_seconds: [not, a, number]\n";
  EXPECT_THROW((void)(parse_config(yaml)), ConfigurationError);
}

TEST(Config, RejectsMalformedGcsProjectId) {
  const std::string yaml = "storage:\n  gcs:\n    project_id: [not, a, scalar]\n";
  EXPECT_THROW((void)(parse_config(yaml)), ConfigurationError);
}

// A few representative read_or<T>() malformed-value cases across distinct
// instantiated types (int, bool) -- read_or<T> is instantiated separately
// per T, and most of its many instantiations (one per distinct config field
// type) previously had neither their success nor failure path exercised by
// any parse_config() test.
TEST(Config, RejectsMalformedMemoryUseAsyncAllocator) {
  EXPECT_THROW((void)(parse_config("memory:\n  use_async_allocator: not-a-bool\n")), ConfigurationError);
}

TEST(Config, RejectsZeroResultBatchRows) {
  EngineConfig config = default_config();
  config.engine.result_batch_rows = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsUnsupportedBackend) {
  EngineConfig config = default_config();
  config.engine.backend = "tpu";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsZeroMemoryPoolInitialBytes) {
  EngineConfig config = default_config();
  config.memory.pool_initial_bytes = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsS3UnknownCredentialsKind) {
  EngineConfig config = default_config();
  config.storage.s3.credentials_kind = "kerberos";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsS3RoleCredentialsMissingRoleArn) {
  EngineConfig config = default_config();
  config.storage.s3.credentials_kind = "role";
  // options.role_arn left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsS3InvalidScheme) {
  EngineConfig config = default_config();
  config.storage.s3.options.scheme = "ftp";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsGcsUnknownCredentialsKind) {
  EngineConfig config = default_config();
  config.storage.gcs.credentials_kind = "kerberos";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsGcsAccessTokenCredentialsMissingToken) {
  EngineConfig config = default_config();
  config.storage.gcs.credentials_kind = "access_token";
  // access_token left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsGcsServiceAccountJsonCredentialsMissingJson) {
  EngineConfig config = default_config();
  config.storage.gcs.credentials_kind = "service_account_json";
  // json_credentials left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsGcsInvalidScheme) {
  EngineConfig config = default_config();
  config.storage.gcs.options.scheme = "ftp";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsAzureUnknownCredentialsKind) {
  EngineConfig config = default_config();
  config.storage.azure.credentials_kind = "kerberos";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsAzureStorageSharedKeyCredentialsMissingKey) {
  EngineConfig config = default_config();
  config.storage.azure.credentials_kind = "storage_shared_key";
  // storage_shared_key left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsAzureSasTokenCredentialsMissingToken) {
  EngineConfig config = default_config();
  config.storage.azure.credentials_kind = "sas_token";
  // sas_token left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsAzureClientSecretCredentialsMissingFields) {
  EngineConfig config = default_config();
  config.storage.azure.credentials_kind = "client_secret";
  // tenant_id/client_id/client_secret all left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsAzureInvalidBlobStorageScheme) {
  EngineConfig config = default_config();
  config.storage.azure.options.blob_storage_scheme = "ftp";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsAzureInvalidDfsStorageScheme) {
  EngineConfig config = default_config();
  config.storage.azure.options.dfs_storage_scheme = "ftp";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsCacheEnabledWithoutDirectory) {
  EngineConfig config = default_config();
  config.storage.cache.enabled = true;
  // directory left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsObservabilityInvalidOtlpProtocol) {
  EngineConfig config = default_config();
  config.observability.otlp_protocol = "carrier-pigeon";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsObservabilityEnabledWithoutEndpoint) {
  EngineConfig config = default_config();
  config.observability.enabled = true;
  config.observability.otlp_endpoint.clear();  // non-empty by default.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsObservabilityEmptyServiceName) {
  EngineConfig config = default_config();
  config.observability.service_name.clear();
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsObservabilityTracingInvalidSampler) {
  EngineConfig config = default_config();
  config.observability.tracing.sampler = "coinflip";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsObservabilityMetricsZeroExportIntervalMs) {
  EngineConfig config = default_config();
  config.observability.metrics.export_interval_ms = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsObservabilityMetricsZeroExportTimeoutMs) {
  EngineConfig config = default_config();
  config.observability.metrics.export_timeout_ms = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

// validate_batch_export_config() (shared by observability.tracing.batch and
// observability.logs.batch) had none of its five error branches covered --
// exercised once via tracing.batch, representative of both callers since
// they share the exact same validation function.
TEST(Config, RejectsTracingBatchInvalidProcessor) {
  EngineConfig config = default_config();
  config.observability.tracing.processor = "carrier-pigeon";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsTracingBatchZeroMaxQueueSize) {
  EngineConfig config = default_config();
  config.observability.tracing.batch.max_queue_size = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsTracingBatchZeroMaxExportBatchSize) {
  EngineConfig config = default_config();
  config.observability.tracing.batch.max_export_batch_size = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsTracingBatchExportSizeExceedingQueueSize) {
  EngineConfig config = default_config();
  config.observability.tracing.batch.max_queue_size = 10;
  config.observability.tracing.batch.max_export_batch_size = 20;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsTracingBatchZeroScheduleDelayMs) {
  EngineConfig config = default_config();
  config.observability.tracing.batch.schedule_delay_ms = 0;
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

// logs.batch shares validate_batch_export_config() with tracing.batch
// above -- one test confirms the *logs* call site itself is reached (not
// just tracing's), rather than re-covering every branch a second time.
TEST(Config, RejectsLogsBatchInvalidProcessor) {
  EngineConfig config = default_config();
  config.observability.logs.processor = "carrier-pigeon";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

// --- CliConfig: the CLI's own top-level config (device_id, benchmark:) -----

TEST(CliConfig, DefaultsMatchSpec) {
  const CliConfig config = default_cli_config();
  EXPECT_EQ(config.device_id, 0);
  EXPECT_EQ(config.benchmark.default_iterations, 5);
  EXPECT_EQ(config.benchmark.warmup_iterations, 1);
  EXPECT_NO_THROW((void)(validate_cli_config(config)));
}

TEST(CliConfig, ParsesOwnTopLevelKeys) {
  const std::string yaml = R"(
engine:
  device_id: 2
  batch_rows: 500000
benchmark:
  default_iterations: 10
  warmup_iterations: 2
)";
  const CliConfig config = parse_cli_config(yaml);
  EXPECT_EQ(config.device_id, 2);
  EXPECT_EQ(config.engine_config.engine.batch_rows, 500000u);
  EXPECT_EQ(config.benchmark.default_iterations, 10);
  EXPECT_EQ(config.benchmark.warmup_iterations, 2);
}

// A server: key present in the same YAML a CLI config is parsed from is
// simply not read here -- both binaries can share one config file (see
// EngineConfig's own comment).
TEST(CliConfig, IgnoresServerOnlyKeys) {
  const std::string yaml = "server:\n  port: 9999\n";
  const CliConfig config = parse_cli_config(yaml);
  EXPECT_EQ(config.device_id, default_cli_config().device_id);
}

TEST(CliConfig, RejectsMalformedDeviceId) {
  EXPECT_THROW((void)(parse_cli_config("engine:\n  device_id: not-a-number\n")), ConfigurationError);
}

TEST(CliConfig, RejectsNegativeDeviceId) {
  CliConfig config = default_cli_config();
  config.device_id = -1;
  EXPECT_THROW((void)(validate_cli_config(config)), ConfigurationError);
}

TEST(CliConfig, RejectsBenchmarkNonPositiveDefaultIterations) {
  CliConfig config = default_cli_config();
  config.benchmark.default_iterations = 0;
  EXPECT_THROW((void)(validate_cli_config(config)), ConfigurationError);
}

TEST(CliConfig, RejectsBenchmarkNegativeWarmupIterations) {
  CliConfig config = default_cli_config();
  config.benchmark.warmup_iterations = -1;
  EXPECT_THROW((void)(validate_cli_config(config)), ConfigurationError);
}

TEST(CliConfig, LoadConfigFileRejectsMissingPath) {
  EXPECT_THROW((void)(load_cli_config_file("/nonexistent/kernellake.yaml")), ConfigurationError);
}

// load_cli_config_file()'s success path (reading a real file off disk and
// delegating to parse_cli_config()) was never exercised -- only the
// missing-path rejection above was.
TEST(CliConfig, LoadConfigFileReadsRealFile) {
  // Suffixed with the test name (not a bare literal) to avoid colliding
  // with another concurrently-running test binary/preset writing to the
  // same shared temp directory -- see every other new fixture in this
  // area (explain_command_test.cpp, generate_data_command_test.cpp,
  // benchmark_tpch_command_test.cpp) for the same convention.
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("kernellake_config_test_" +
       std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + ".yaml");
  {
    std::ofstream out(path);
    out << "engine:\n  device_id: 3\n";
  }
  const CliConfig config = load_cli_config_file(path.string());
  std::filesystem::remove(path);
  EXPECT_EQ(config.device_id, 3);
}

// --- ServerConfig: kernellake-server's own top-level config (server:,
// engine.max_concurrent_gpu_queries, engine.gpu_device_ids) ------------------

TEST(ServerConfig, DefaultsMatchSpec) {
  const ServerConfig config = default_server_config();
  EXPECT_EQ(config.max_concurrent_gpu_queries, 2);
  EXPECT_TRUE(config.gpu_device_ids.empty());
  EXPECT_NO_THROW((void)(validate_server_config(config)));
}

// A benchmark:/engine.device_id key present in the same YAML a server
// config is parsed from is simply not read here.
TEST(ServerConfig, IgnoresCliOnlyKeys) {
  const std::string yaml = "engine:\n  device_id: 7\nbenchmark:\n  default_iterations: 99\n";
  const ServerConfig config = parse_server_config(yaml);
  EXPECT_EQ(config.max_concurrent_gpu_queries, default_server_config().max_concurrent_gpu_queries);
}

TEST(ServerConfig, ParsesMaxConcurrentGpuQueries) {
  const ServerConfig config = parse_server_config("engine:\n  max_concurrent_gpu_queries: 4\n");
  EXPECT_EQ(config.max_concurrent_gpu_queries, 4);
}

TEST(ServerConfig, RejectsNonPositiveMaxConcurrentGpuQueries) {
  ServerConfig config = default_server_config();
  config.max_concurrent_gpu_queries = 0;
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, ParsesGpuDeviceIds) {
  const ServerConfig config = parse_server_config("engine:\n  gpu_device_ids: [0, 2, 3]\n");
  EXPECT_EQ(config.gpu_device_ids, (std::vector<int>{0, 2, 3}));
}

TEST(ServerConfig, RejectsNegativeGpuDeviceId) {
  ServerConfig config = default_server_config();
  config.gpu_device_ids = {0, -1};
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, RejectsDuplicateGpuDeviceIds) {
  ServerConfig config = default_server_config();
  config.gpu_device_ids = {0, 1, 0};
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, ParsesServerTlsSection) {
  const std::string yaml = R"(
server:
  host: 127.0.0.1
  port: 31338
  use_tls: true
  tls_cert_path: /etc/ssl/server.pem
  tls_key_path: /etc/ssl/server.key
  require_client_cert: true
  tls_client_ca_cert_path: /etc/ssl/client_ca.pem
)";
  const ServerConfig config = parse_server_config(yaml);
  EXPECT_TRUE(config.server.use_tls);
  EXPECT_EQ(config.server.tls_cert_path, "/etc/ssl/server.pem");
  EXPECT_EQ(config.server.tls_key_path, "/etc/ssl/server.key");
  EXPECT_TRUE(config.server.require_client_cert);
  EXPECT_EQ(config.server.tls_client_ca_cert_path, "/etc/ssl/client_ca.pem");
  EXPECT_NO_THROW((void)(validate_server_config(config)));
}

TEST(ServerConfig, ServerTlsDefaultsToDisabled) {
  const ServerConfig config = default_server_config();
  EXPECT_FALSE(config.server.use_tls);
  EXPECT_TRUE(config.server.tls_cert_path.empty());
  EXPECT_TRUE(config.server.tls_key_path.empty());
  EXPECT_FALSE(config.server.require_client_cert);
}

TEST(ServerConfig, RejectsServerTlsMissingCertOrKey) {
  ServerConfig config = default_server_config();
  config.server.use_tls = true;
  config.server.tls_cert_path = "/etc/ssl/server.pem";
  // tls_key_path left empty.
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);

  config = default_server_config();
  config.server.use_tls = true;
  config.server.tls_key_path = "/etc/ssl/server.key";
  // tls_cert_path left empty.
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, RejectsServerRequireClientCertWithoutTls) {
  ServerConfig config = default_server_config();
  config.server.require_client_cert = true;
  config.server.tls_client_ca_cert_path = "/etc/ssl/client_ca.pem";
  // use_tls left false.
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, RejectsServerRequireClientCertMissingCaPath) {
  ServerConfig config = default_server_config();
  config.server.use_tls = true;
  config.server.tls_cert_path = "/etc/ssl/server.pem";
  config.server.tls_key_path = "/etc/ssl/server.key";
  config.server.require_client_cert = true;
  // tls_client_ca_cert_path left empty.
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, ParsesServerAuthSection) {
  const std::string yaml = R"(
server:
  auth_enabled: true
  auth_token: super-secret-token
)";
  const ServerConfig config = parse_server_config(yaml);
  EXPECT_TRUE(config.server.auth_enabled);
  EXPECT_EQ(config.server.auth_token, "super-secret-token");
  EXPECT_NO_THROW((void)(validate_server_config(config)));
}

TEST(ServerConfig, ServerAuthDefaultsToDisabled) {
  const ServerConfig config = default_server_config();
  EXPECT_FALSE(config.server.auth_enabled);
  EXPECT_TRUE(config.server.auth_token.empty());
}

TEST(ServerConfig, RejectsServerAuthEnabledWithoutToken) {
  ServerConfig config = default_server_config();
  config.server.auth_enabled = true;
  // auth_token left empty.
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, LoadConfigFileRejectsMissingPath) {
  EXPECT_THROW((void)(load_server_config_file("/nonexistent/kernellake.yaml")), ConfigurationError);
}

TEST(ServerConfig, RejectsMalformedServerPort) {
  EXPECT_THROW((void)(parse_server_config("server:\n  port: not-a-port\n")), ConfigurationError);
}

TEST(ServerConfig, RejectsServerZeroPort) {
  ServerConfig config = default_server_config();
  config.server.port = 0;
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

TEST(ServerConfig, RejectsServerZeroMaxPendingResults) {
  ServerConfig config = default_server_config();
  config.server.max_pending_results = 0;
  EXPECT_THROW((void)(validate_server_config(config)), ConfigurationError);
}

}  // namespace
}  // namespace kernellake
