#include "kernellake/common/config.hpp"

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

template <typename T>
T read_or(const YAML::Node& node, const char* key, T fallback) {
  if (!node || !node[key]) {
    return fallback;
  }
  try {
    return node[key].as<T>();
  } catch (const YAML::Exception& e) {
    throw ConfigurationError(fmt::format("invalid value for '{}': {}", key, e.what()));
  }
}

// yaml-cpp throws ("invalid node") if operator[] is called on a node that is
// itself undefined (e.g. `observability["tracing"]` when the YAML has no
// `observability:` key at all) -- unlike read_or above, which guards this
// via its own `!node ||` short-circuit for a single level. Nested section
// lookups (observability.tracing, .tracing.batch, etc.) need this guard
// explicitly since they index two levels deep.
YAML::Node child(const YAML::Node& node, const char* key) {
  return node ? node[key] : YAML::Node();
}

// Reads a YAML mapping into a string->string map -- used by
// storage.hdfs.connection_config.extra_conf (arrow::io::HdfsConnectionConfig's
// own field type, std::unordered_map<std::string, std::string>).
std::unordered_map<std::string, std::string> read_string_map(const YAML::Node& node) {
  std::unordered_map<std::string, std::string> result;
  if (!node) {
    return result;
  }
  for (const auto& entry : node) {
    try {
      result.emplace(entry.first.as<std::string>(), entry.second.as<std::string>());
    } catch (const YAML::Exception& e) {
      throw ConfigurationError(fmt::format("invalid string map entry: {}", e.what()));
    }
  }
  return result;
}

// Shared by observability.tracing.batch and observability.logs.batch, which
// have an identical shape.
BatchExportConfig read_batch_export_config(const YAML::Node& node, BatchExportConfig fallback) {
  const YAML::Node batch = child(node, "batch");
  BatchExportConfig result = fallback;
  result.max_queue_size = read_or(batch, "max_queue_size", result.max_queue_size);
  result.max_export_batch_size = read_or(batch, "max_export_batch_size", result.max_export_batch_size);
  result.schedule_delay_ms = read_or(batch, "schedule_delay_ms", result.schedule_delay_ms);
  return result;
}

// Shared by observability.tracing and observability.logs, which have an
// identical processor/batch validation shape.
void validate_batch_export_config(const std::string& prefix, const std::string& processor,
                                  const BatchExportConfig& batch) {
  if (processor != "simple" && processor != "batch") {
    throw ConfigurationError(
        fmt::format("{}.processor '{}' is unsupported (expected 'simple' or 'batch')", prefix, processor));
  }
  if (batch.max_queue_size == 0) {
    throw ConfigurationError(fmt::format("{}.batch.max_queue_size must be > 0", prefix));
  }
  if (batch.max_export_batch_size == 0) {
    throw ConfigurationError(fmt::format("{}.batch.max_export_batch_size must be > 0", prefix));
  }
  if (batch.max_export_batch_size > batch.max_queue_size) {
    throw ConfigurationError(
        fmt::format("{}.batch.max_export_batch_size ({}) must be <= {}.batch.max_queue_size ({})", prefix,
                    batch.max_export_batch_size, prefix, batch.max_queue_size));
  }
  if (batch.schedule_delay_ms == 0) {
    throw ConfigurationError(fmt::format("{}.batch.schedule_delay_ms must be > 0", prefix));
  }
}

}  // namespace

EngineConfig default_config() {
  return EngineConfig{};
}

EngineConfig parse_config(const std::string& yaml_text) {
  YAML::Node root;
  try {
    root = YAML::Load(yaml_text);
  } catch (const YAML::Exception& e) {
    throw ConfigurationError(fmt::format("failed to parse YAML configuration: {}", e.what()));
  }

  EngineConfig config;

  const YAML::Node engine = root["engine"];
  config.engine.device_id = read_or(engine, "device_id", config.engine.device_id);
  config.engine.batch_rows = read_or(engine, "batch_rows", config.engine.batch_rows);
  config.engine.result_batch_rows = read_or(engine, "result_batch_rows", config.engine.result_batch_rows);
  config.engine.query_memory_limit_bytes =
      read_or(engine, "query_memory_limit_bytes", config.engine.query_memory_limit_bytes);
  config.engine.backend = read_or(engine, "backend", config.engine.backend);

  const YAML::Node memory = root["memory"];
  config.memory.pool_initial_bytes = read_or(memory, "pool_initial_bytes", config.memory.pool_initial_bytes);
  config.memory.pool_max_bytes = read_or(memory, "pool_max_bytes", config.memory.pool_max_bytes);
  config.memory.use_async_allocator =
      read_or(memory, "use_async_allocator", config.memory.use_async_allocator);

  const YAML::Node storage = root["storage"];
  config.storage.local_root = read_or(storage, "local_root", config.storage.local_root);

  const YAML::Node s3 = child(storage, "s3");
  config.storage.s3.credentials_kind = read_or(s3, "credentials_kind", config.storage.s3.credentials_kind);
  arrow::fs::S3Options& s3_opts = config.storage.s3.options;
  s3_opts.smart_defaults = read_or(s3, "smart_defaults", s3_opts.smart_defaults);
  s3_opts.region = read_or(s3, "region", s3_opts.region);
  s3_opts.connect_timeout = read_or(s3, "connect_timeout", s3_opts.connect_timeout);
  s3_opts.request_timeout = read_or(s3, "request_timeout", s3_opts.request_timeout);
  s3_opts.endpoint_override = read_or(s3, "endpoint_override", s3_opts.endpoint_override);
  s3_opts.scheme = read_or(s3, "scheme", s3_opts.scheme);
  s3_opts.role_arn = read_or(s3, "role_arn", s3_opts.role_arn);
  s3_opts.session_name = read_or(s3, "session_name", s3_opts.session_name);
  s3_opts.external_id = read_or(s3, "external_id", s3_opts.external_id);
  s3_opts.load_frequency = read_or(s3, "load_frequency", s3_opts.load_frequency);
  const YAML::Node s3_proxy = child(s3, "proxy_options");
  s3_opts.proxy_options.scheme = read_or(s3_proxy, "scheme", s3_opts.proxy_options.scheme);
  s3_opts.proxy_options.host = read_or(s3_proxy, "host", s3_opts.proxy_options.host);
  s3_opts.proxy_options.port = read_or(s3_proxy, "port", s3_opts.proxy_options.port);
  s3_opts.proxy_options.username = read_or(s3_proxy, "username", s3_opts.proxy_options.username);
  s3_opts.proxy_options.password = read_or(s3_proxy, "password", s3_opts.proxy_options.password);
  s3_opts.force_virtual_addressing =
      read_or(s3, "force_virtual_addressing", s3_opts.force_virtual_addressing);
  s3_opts.background_writes = read_or(s3, "background_writes", s3_opts.background_writes);
  s3_opts.allow_bucket_creation = read_or(s3, "allow_bucket_creation", s3_opts.allow_bucket_creation);
  s3_opts.allow_bucket_deletion = read_or(s3, "allow_bucket_deletion", s3_opts.allow_bucket_deletion);
  s3_opts.check_directory_existence_before_creation = read_or(
      s3, "check_directory_existence_before_creation", s3_opts.check_directory_existence_before_creation);
  s3_opts.allow_delayed_open = read_or(s3, "allow_delayed_open", s3_opts.allow_delayed_open);
  s3_opts.sse_customer_key = read_or(s3, "sse_customer_key", s3_opts.sse_customer_key);
  s3_opts.tls_ca_file_path = read_or(s3, "tls_ca_file_path", s3_opts.tls_ca_file_path);
  s3_opts.tls_ca_dir_path = read_or(s3, "tls_ca_dir_path", s3_opts.tls_ca_dir_path);
  s3_opts.tls_verify_certificates = read_or(s3, "tls_verify_certificates", s3_opts.tls_verify_certificates);

  const YAML::Node gcs = child(storage, "gcs");
  config.storage.gcs.credentials_kind = read_or(gcs, "credentials_kind", config.storage.gcs.credentials_kind);
  config.storage.gcs.access_token = read_or(gcs, "access_token", config.storage.gcs.access_token);
  config.storage.gcs.access_token_expiration =
      read_or(gcs, "access_token_expiration", config.storage.gcs.access_token_expiration);
  config.storage.gcs.target_service_account =
      read_or(gcs, "target_service_account", config.storage.gcs.target_service_account);
  config.storage.gcs.json_credentials = read_or(gcs, "json_credentials", config.storage.gcs.json_credentials);
  arrow::fs::GcsOptions& gcs_opts = config.storage.gcs.options;
  gcs_opts.endpoint_override = read_or(gcs, "endpoint_override", gcs_opts.endpoint_override);
  gcs_opts.scheme = read_or(gcs, "scheme", gcs_opts.scheme);
  gcs_opts.default_bucket_location =
      read_or(gcs, "default_bucket_location", gcs_opts.default_bucket_location);
  // retry_limit_seconds/project_id are std::optional<> on GcsOptions --
  // read_or's `.as<T>()` has no std::optional<T> specialization, so these
  // need an explicit presence check instead.
  if (gcs && gcs["retry_limit_seconds"]) {
    try {
      gcs_opts.retry_limit_seconds = gcs["retry_limit_seconds"].as<double>();
    } catch (const YAML::Exception& e) {
      throw ConfigurationError(fmt::format("invalid value for 'retry_limit_seconds': {}", e.what()));
    }
  }
  if (gcs && gcs["project_id"]) {
    try {
      gcs_opts.project_id = gcs["project_id"].as<std::string>();
    } catch (const YAML::Exception& e) {
      throw ConfigurationError(fmt::format("invalid value for 'project_id': {}", e.what()));
    }
  }

  const YAML::Node azure = child(storage, "azure");
  config.storage.azure.credentials_kind =
      read_or(azure, "credentials_kind", config.storage.azure.credentials_kind);
  config.storage.azure.storage_shared_key =
      read_or(azure, "storage_shared_key", config.storage.azure.storage_shared_key);
  config.storage.azure.sas_token = read_or(azure, "sas_token", config.storage.azure.sas_token);
  config.storage.azure.tenant_id = read_or(azure, "tenant_id", config.storage.azure.tenant_id);
  config.storage.azure.client_id = read_or(azure, "client_id", config.storage.azure.client_id);
  config.storage.azure.client_secret = read_or(azure, "client_secret", config.storage.azure.client_secret);
  arrow::fs::AzureOptions& azure_opts = config.storage.azure.options;
  azure_opts.account_name = read_or(azure, "account_name", azure_opts.account_name);
  azure_opts.blob_storage_authority =
      read_or(azure, "blob_storage_authority", azure_opts.blob_storage_authority);
  azure_opts.dfs_storage_authority =
      read_or(azure, "dfs_storage_authority", azure_opts.dfs_storage_authority);
  azure_opts.blob_storage_scheme = read_or(azure, "blob_storage_scheme", azure_opts.blob_storage_scheme);
  azure_opts.dfs_storage_scheme = read_or(azure, "dfs_storage_scheme", azure_opts.dfs_storage_scheme);
  azure_opts.background_writes = read_or(azure, "background_writes", azure_opts.background_writes);

  const YAML::Node hdfs = child(storage, "hdfs");
  const YAML::Node hdfs_conn = child(hdfs, "connection_config");
  arrow::fs::HdfsOptions& hdfs_opts = config.storage.hdfs.options;
  hdfs_opts.connection_config.host = read_or(hdfs_conn, "host", hdfs_opts.connection_config.host);
  hdfs_opts.connection_config.port = read_or(hdfs_conn, "port", hdfs_opts.connection_config.port);
  hdfs_opts.connection_config.user = read_or(hdfs_conn, "user", hdfs_opts.connection_config.user);
  hdfs_opts.connection_config.kerb_ticket =
      read_or(hdfs_conn, "kerb_ticket", hdfs_opts.connection_config.kerb_ticket);
  hdfs_opts.connection_config.extra_conf = read_string_map(child(hdfs_conn, "extra_conf"));
  hdfs_opts.buffer_size = read_or(hdfs, "buffer_size", hdfs_opts.buffer_size);
  hdfs_opts.replication = read_or(hdfs, "replication", hdfs_opts.replication);
  hdfs_opts.default_block_size = read_or(hdfs, "default_block_size", hdfs_opts.default_block_size);

  const YAML::Node cache = child(storage, "cache");
  config.storage.cache.enabled = read_or(cache, "enabled", config.storage.cache.enabled);
  config.storage.cache.directory = read_or(cache, "directory", config.storage.cache.directory);
  config.storage.cache.max_size_bytes = read_or(cache, "max_size_bytes", config.storage.cache.max_size_bytes);

  // iceberg.catalogs is a map keyed by catalog name (read_iceberg('name.ns.table')'s
  // leading component looks it up), unlike storage.{s3,gcs,azure,hdfs}'s
  // single-section-per-scheme shape above -- each entry gets its own
  // sub-object read the same way those do.
  const YAML::Node iceberg_catalogs = root["iceberg"]["catalogs"];
  if (iceberg_catalogs) {
    for (const auto& entry : iceberg_catalogs) {
      std::string name;
      try {
        name = entry.first.as<std::string>();
      } catch (const YAML::Exception& e) {
        throw ConfigurationError(fmt::format("invalid iceberg.catalogs entry name: {}", e.what()));
      }
      const YAML::Node catalog_node = entry.second;
      IcebergCatalogSection catalog;
      catalog.catalog_uri = read_or(catalog_node, "catalog_uri", catalog.catalog_uri);
      catalog.warehouse = read_or(catalog_node, "warehouse", catalog.warehouse);
      catalog.prefix = read_or(catalog_node, "prefix", catalog.prefix);
      catalog.credentials_kind = read_or(catalog_node, "credentials_kind", catalog.credentials_kind);
      catalog.bearer_token = read_or(catalog_node, "bearer_token", catalog.bearer_token);
      catalog.oauth2_client_id = read_or(catalog_node, "oauth2_client_id", catalog.oauth2_client_id);
      catalog.oauth2_client_secret =
          read_or(catalog_node, "oauth2_client_secret", catalog.oauth2_client_secret);
      catalog.oauth2_scope = read_or(catalog_node, "oauth2_scope", catalog.oauth2_scope);
      config.iceberg.catalogs.emplace(std::move(name), std::move(catalog));
    }
  }

  const YAML::Node delta = root["delta"];
  config.delta.grpc_endpoint = read_or(delta, "grpc_endpoint", config.delta.grpc_endpoint);
  config.delta.use_tls = read_or(delta, "use_tls", config.delta.use_tls);
  config.delta.tls_ca_cert_path = read_or(delta, "tls_ca_cert_path", config.delta.tls_ca_cert_path);
  config.delta.api_key = read_or(delta, "api_key", config.delta.api_key);

  const YAML::Node logging = root["logging"];
  config.logging.level = read_or(logging, "level", config.logging.level);
  config.logging.json = read_or(logging, "json", config.logging.json);

  const YAML::Node profiling = root["profiling"];
  config.profiling.nvtx = read_or(profiling, "nvtx", config.profiling.nvtx);
  config.profiling.operator_metrics =
      read_or(profiling, "operator_metrics", config.profiling.operator_metrics);

  const YAML::Node benchmark = root["benchmark"];
  config.benchmark.default_iterations =
      read_or(benchmark, "default_iterations", config.benchmark.default_iterations);
  config.benchmark.warmup_iterations =
      read_or(benchmark, "warmup_iterations", config.benchmark.warmup_iterations);
  config.benchmark.output_format = read_or(benchmark, "output_format", config.benchmark.output_format);
  config.benchmark.verify_results = read_or(benchmark, "verify_results", config.benchmark.verify_results);
  config.benchmark.baseline = read_or(benchmark, "baseline", config.benchmark.baseline);

  const YAML::Node server = root["server"];
  config.server.host = read_or(server, "host", config.server.host);
  config.server.port = read_or(server, "port", config.server.port);
  config.server.max_pending_results =
      read_or(server, "max_pending_results", config.server.max_pending_results);
  config.server.use_tls = read_or(server, "use_tls", config.server.use_tls);
  config.server.tls_cert_path = read_or(server, "tls_cert_path", config.server.tls_cert_path);
  config.server.tls_key_path = read_or(server, "tls_key_path", config.server.tls_key_path);
  config.server.require_client_cert =
      read_or(server, "require_client_cert", config.server.require_client_cert);
  config.server.tls_client_ca_cert_path =
      read_or(server, "tls_client_ca_cert_path", config.server.tls_client_ca_cert_path);
  config.server.auth_enabled = read_or(server, "auth_enabled", config.server.auth_enabled);
  config.server.auth_token = read_or(server, "auth_token", config.server.auth_token);

  const YAML::Node observability = root["observability"];
  config.observability.enabled = read_or(observability, "enabled", config.observability.enabled);
  config.observability.otlp_protocol =
      read_or(observability, "otlp_protocol", config.observability.otlp_protocol);
  config.observability.otlp_endpoint =
      read_or(observability, "otlp_endpoint", config.observability.otlp_endpoint);
  config.observability.service_name =
      read_or(observability, "service_name", config.observability.service_name);
  config.observability.use_tls = read_or(observability, "use_tls", config.observability.use_tls);
  config.observability.tls_ca_cert_path =
      read_or(observability, "tls_ca_cert_path", config.observability.tls_ca_cert_path);
  config.observability.tls_client_cert_path =
      read_or(observability, "tls_client_cert_path", config.observability.tls_client_cert_path);
  config.observability.tls_client_key_path =
      read_or(observability, "tls_client_key_path", config.observability.tls_client_key_path);

  const YAML::Node tracing = child(observability, "tracing");
  config.observability.tracing.processor =
      read_or(tracing, "processor", config.observability.tracing.processor);
  config.observability.tracing.batch = read_batch_export_config(tracing, config.observability.tracing.batch);
  config.observability.tracing.sampler = read_or(tracing, "sampler", config.observability.tracing.sampler);

  const YAML::Node metrics = child(observability, "metrics");
  config.observability.metrics.export_interval_ms =
      read_or(metrics, "export_interval_ms", config.observability.metrics.export_interval_ms);
  config.observability.metrics.export_timeout_ms =
      read_or(metrics, "export_timeout_ms", config.observability.metrics.export_timeout_ms);

  const YAML::Node logs = child(observability, "logs");
  config.observability.logs.processor = read_or(logs, "processor", config.observability.logs.processor);
  config.observability.logs.batch = read_batch_export_config(logs, config.observability.logs.batch);

  return config;
}

EngineConfig load_config_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw ConfigurationError(
        fmt::format("cannot open configuration file '{}': check that the path exists and is readable", path));
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parse_config(buffer.str());
}

void validate_config(const EngineConfig& config) {
  if (config.engine.device_id < 0) {
    throw ConfigurationError(fmt::format("engine.device_id must be >= 0, got {}", config.engine.device_id));
  }
  if (config.engine.batch_rows == 0) {
    throw ConfigurationError("engine.batch_rows must be > 0");
  }
  if (config.engine.result_batch_rows == 0) {
    throw ConfigurationError("engine.result_batch_rows must be > 0");
  }
  // 0 is a real, meaningful value here ("auto-detect from GPU VRAM" -- see
  // EngineSection::query_memory_limit_bytes's own comment), not an error;
  // no validation needed either way, since any std::uint64_t is a valid
  // byte count.
  if (config.engine.backend != "gpu" && config.engine.backend != "cpu") {
    throw ConfigurationError(
        fmt::format("engine.backend '{}' is unsupported (expected 'gpu' or 'cpu')", config.engine.backend));
  }

  if (config.memory.pool_initial_bytes == 0) {
    throw ConfigurationError("memory.pool_initial_bytes must be > 0");
  }
  if (config.memory.pool_max_bytes < config.memory.pool_initial_bytes) {
    throw ConfigurationError(
        fmt::format("memory.pool_max_bytes ({}) must be >= memory.pool_initial_bytes ({})",
                    config.memory.pool_max_bytes, config.memory.pool_initial_bytes));
  }

  static constexpr std::array<const char*, 5> kS3CredentialsKinds = {"anonymous", "default", "explicit",
                                                                     "role", "web_identity"};
  if (std::find(kS3CredentialsKinds.begin(), kS3CredentialsKinds.end(), config.storage.s3.credentials_kind) ==
      kS3CredentialsKinds.end()) {
    throw ConfigurationError(
        fmt::format("storage.s3.credentials_kind '{}' is unsupported (expected 'anonymous', 'default', "
                    "'explicit', 'role', or "
                    "'web_identity')",
                    config.storage.s3.credentials_kind));
  }
  if (config.storage.s3.credentials_kind == "role" && config.storage.s3.options.role_arn.empty()) {
    throw ConfigurationError(
        "storage.s3.options.role_arn must not be empty when "
        "storage.s3.credentials_kind is 'role'");
  }
  if (config.storage.s3.options.scheme != "https" && config.storage.s3.options.scheme != "http") {
    throw ConfigurationError(fmt::format("storage.s3.scheme '{}' is unsupported (expected 'https' or 'http')",
                                         config.storage.s3.options.scheme));
  }

  static constexpr std::array<const char*, 4> kGcsCredentialsKinds = {"anonymous", "default", "access_token",
                                                                      "service_account_json"};
  if (std::find(kGcsCredentialsKinds.begin(), kGcsCredentialsKinds.end(),
                config.storage.gcs.credentials_kind) == kGcsCredentialsKinds.end()) {
    throw ConfigurationError(
        fmt::format("storage.gcs.credentials_kind '{}' is unsupported (expected 'anonymous', 'default', "
                    "'access_token', or "
                    "'service_account_json')",
                    config.storage.gcs.credentials_kind));
  }
  if (config.storage.gcs.credentials_kind == "access_token" && config.storage.gcs.access_token.empty()) {
    throw ConfigurationError(
        "storage.gcs.access_token must not be empty when storage.gcs.credentials_kind is 'access_token'");
  }
  if (config.storage.gcs.credentials_kind == "service_account_json" &&
      config.storage.gcs.json_credentials.empty()) {
    throw ConfigurationError(
        "storage.gcs.json_credentials must not be empty when "
        "storage.gcs.credentials_kind is 'service_account_json'");
  }
  if (config.storage.gcs.options.scheme != "https" && config.storage.gcs.options.scheme != "http") {
    throw ConfigurationError(
        fmt::format("storage.gcs.scheme '{}' is unsupported (expected 'https' or 'http')",
                    config.storage.gcs.options.scheme));
  }

  static constexpr std::array<const char*, 9> kAzureCredentialsKinds = {
      "default",          "anonymous", "storage_shared_key", "sas_token",  "client_secret",
      "managed_identity", "cli",       "workload_identity",  "environment"};
  if (std::find(kAzureCredentialsKinds.begin(), kAzureCredentialsKinds.end(),
                config.storage.azure.credentials_kind) == kAzureCredentialsKinds.end()) {
    throw ConfigurationError(fmt::format(
        "storage.azure.credentials_kind '{}' is unsupported (expected 'default', 'anonymous', "
        "'storage_shared_key', 'sas_token', 'client_secret', 'managed_identity', 'cli', 'workload_identity', "
        "or 'environment')",
        config.storage.azure.credentials_kind));
  }
  if (config.storage.azure.credentials_kind == "storage_shared_key" &&
      config.storage.azure.storage_shared_key.empty()) {
    throw ConfigurationError(
        "storage.azure.storage_shared_key must not be empty when "
        "storage.azure.credentials_kind is 'storage_shared_key'");
  }
  if (config.storage.azure.credentials_kind == "sas_token" && config.storage.azure.sas_token.empty()) {
    throw ConfigurationError(
        "storage.azure.sas_token must not be empty when storage.azure.credentials_kind is 'sas_token'");
  }
  if (config.storage.azure.credentials_kind == "client_secret" &&
      (config.storage.azure.tenant_id.empty() || config.storage.azure.client_id.empty() ||
       config.storage.azure.client_secret.empty())) {
    throw ConfigurationError(
        "storage.azure.tenant_id, client_id, and client_secret must all be set when "
        "storage.azure.credentials_kind is 'client_secret'");
  }
  if (config.storage.azure.options.blob_storage_scheme != "https" &&
      config.storage.azure.options.blob_storage_scheme != "http") {
    throw ConfigurationError(
        fmt::format("storage.azure.blob_storage_scheme '{}' is unsupported (expected 'https' or 'http')",
                    config.storage.azure.options.blob_storage_scheme));
  }
  if (config.storage.azure.options.dfs_storage_scheme != "https" &&
      config.storage.azure.options.dfs_storage_scheme != "http") {
    throw ConfigurationError(
        fmt::format("storage.azure.dfs_storage_scheme '{}' is unsupported (expected 'https' or 'http')",
                    config.storage.azure.options.dfs_storage_scheme));
  }

  if (config.storage.cache.enabled && config.storage.cache.directory.empty()) {
    throw ConfigurationError("storage.cache.directory must not be empty when storage.cache.enabled is true");
  }

  static constexpr std::array<const char*, 3> kIcebergCredentialsKinds = {"none", "bearer_token",
                                                                          "oauth2_client_credentials"};
  for (const auto& [name, catalog] : config.iceberg.catalogs) {
    if (catalog.catalog_uri.empty()) {
      throw ConfigurationError(fmt::format("iceberg.catalogs.{}.catalog_uri must not be empty", name));
    }
    if (std::find(kIcebergCredentialsKinds.begin(), kIcebergCredentialsKinds.end(),
                  catalog.credentials_kind) == kIcebergCredentialsKinds.end()) {
      throw ConfigurationError(fmt::format(
          "iceberg.catalogs.{}.credentials_kind '{}' is unsupported (expected 'none', 'bearer_token', or "
          "'oauth2_client_credentials')",
          name, catalog.credentials_kind));
    }
    if (catalog.credentials_kind == "bearer_token" && catalog.bearer_token.empty()) {
      throw ConfigurationError(fmt::format(
          "iceberg.catalogs.{}.bearer_token must not be empty when credentials_kind is 'bearer_token'",
          name));
    }
    if (catalog.credentials_kind == "oauth2_client_credentials" &&
        (catalog.oauth2_client_id.empty() || catalog.oauth2_client_secret.empty())) {
      throw ConfigurationError(
          fmt::format("iceberg.catalogs.{}.oauth2_client_id and oauth2_client_secret must both be set when "
                      "credentials_kind is 'oauth2_client_credentials'",
                      name));
    }
  }

  static constexpr std::array<const char*, 7> kLogLevels = {"trace",   "debug", "info",    "warn",
                                                            "warning", "error", "critical"};
  bool level_ok = false;
  for (const char* level : kLogLevels) {
    if (config.logging.level == level) {
      level_ok = true;
      break;
    }
  }
  if (!level_ok) {
    throw ConfigurationError(
        fmt::format("logging.level '{}' is not a recognized level "
                    "(expected one of trace/debug/info/warn/error/critical)",
                    config.logging.level));
  }

  if (config.benchmark.output_format != "json" && config.benchmark.output_format != "csv") {
    throw ConfigurationError(
        fmt::format("benchmark.output_format '{}' is unsupported (expected 'json' or 'csv')",
                    config.benchmark.output_format));
  }
  if (config.benchmark.baseline != "duckdb" && config.benchmark.baseline != "none") {
    throw ConfigurationError(fmt::format(
        "benchmark.baseline '{}' is unsupported (expected 'duckdb' or 'none')", config.benchmark.baseline));
  }
  if (config.benchmark.default_iterations <= 0) {
    throw ConfigurationError("benchmark.default_iterations must be > 0");
  }
  if (config.benchmark.warmup_iterations < 0) {
    throw ConfigurationError("benchmark.warmup_iterations must be >= 0");
  }

  if (config.server.port == 0) {
    throw ConfigurationError("server.port must be > 0");
  }
  if (config.server.max_pending_results == 0) {
    throw ConfigurationError("server.max_pending_results must be > 0");
  }
  if (config.server.use_tls && (config.server.tls_cert_path.empty() || config.server.tls_key_path.empty())) {
    throw ConfigurationError(
        "server.tls_cert_path and server.tls_key_path must both be set when server.use_tls is true");
  }
  if (config.server.require_client_cert && !config.server.use_tls) {
    throw ConfigurationError("server.require_client_cert requires server.use_tls to also be true");
  }
  if (config.server.require_client_cert && config.server.tls_client_ca_cert_path.empty()) {
    throw ConfigurationError(
        "server.tls_client_ca_cert_path must be set when server.require_client_cert is true");
  }
  if (config.server.auth_enabled && config.server.auth_token.empty()) {
    throw ConfigurationError("server.auth_token must not be empty when server.auth_enabled is true");
  }

  if (config.observability.otlp_protocol != "grpc" && config.observability.otlp_protocol != "http") {
    throw ConfigurationError(
        fmt::format("observability.otlp_protocol '{}' is unsupported (expected 'grpc' or 'http')",
                    config.observability.otlp_protocol));
  }
  if (config.observability.enabled && config.observability.otlp_endpoint.empty()) {
    throw ConfigurationError(
        "observability.otlp_endpoint must not be empty when observability.enabled is true");
  }
  if (config.observability.service_name.empty()) {
    throw ConfigurationError("observability.service_name must not be empty");
  }
  validate_batch_export_config("observability.tracing", config.observability.tracing.processor,
                               config.observability.tracing.batch);
  if (config.observability.tracing.sampler != "default" && config.observability.tracing.sampler != "always" &&
      config.observability.tracing.sampler != "never") {
    throw ConfigurationError(fmt::format(
        "observability.tracing.sampler '{}' is unsupported (expected 'default', 'always', or 'never')",
        config.observability.tracing.sampler));
  }
  validate_batch_export_config("observability.logs", config.observability.logs.processor,
                               config.observability.logs.batch);
  if (config.observability.metrics.export_interval_ms == 0) {
    throw ConfigurationError("observability.metrics.export_interval_ms must be > 0");
  }
  if (config.observability.metrics.export_timeout_ms == 0) {
    throw ConfigurationError("observability.metrics.export_timeout_ms must be > 0");
  }
}

}  // namespace kernellake
