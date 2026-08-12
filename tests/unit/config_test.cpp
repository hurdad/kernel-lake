#include <gtest/gtest.h>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

TEST(Config, DefaultsMatchSpec) {
  const EngineConfig config = default_config();
  EXPECT_EQ(config.engine.device_id, 0);
  EXPECT_EQ(config.engine.batch_rows, 1'000'000u);
  EXPECT_EQ(config.memory.pool_max_bytes, 8ULL * 1024 * 1024 * 1024);
  EXPECT_EQ(config.benchmark.baseline, "duckdb");
  EXPECT_NO_THROW((void)(validate_config(config)));
}

TEST(Config, ParsesOverrides) {
  const std::string yaml = R"(
engine:
  device_id: 2
  batch_rows: 500000
logging:
  level: debug
  json: true
benchmark:
  output_format: csv
)";
  const EngineConfig config = parse_config(yaml);
  EXPECT_EQ(config.engine.device_id, 2);
  EXPECT_EQ(config.engine.batch_rows, 500000u);
  EXPECT_EQ(config.logging.level, "debug");
  EXPECT_TRUE(config.logging.json);
  EXPECT_EQ(config.benchmark.output_format, "csv");
  // Fields not present in the override fall back to defaults.
  EXPECT_EQ(config.engine.result_batch_rows, default_config().engine.result_batch_rows);
}

TEST(Config, RejectsZeroBatchRows) {
  EngineConfig config = default_config();
  config.engine.batch_rows = 0;
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

TEST(Config, RejectsUnsupportedBenchmarkMode) {
  EngineConfig config = default_config();
  config.benchmark.output_format = "xml";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);

  config = default_config();
  config.benchmark.baseline = "clickhouse-but-not-really";
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

TEST(Config, ParsesServerTlsSection) {
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
  const EngineConfig config = parse_config(yaml);
  EXPECT_TRUE(config.server.use_tls);
  EXPECT_EQ(config.server.tls_cert_path, "/etc/ssl/server.pem");
  EXPECT_EQ(config.server.tls_key_path, "/etc/ssl/server.key");
  EXPECT_TRUE(config.server.require_client_cert);
  EXPECT_EQ(config.server.tls_client_ca_cert_path, "/etc/ssl/client_ca.pem");
  EXPECT_NO_THROW((void)(validate_config(config)));
}

TEST(Config, ServerTlsDefaultsToDisabled) {
  const EngineConfig config = default_config();
  EXPECT_FALSE(config.server.use_tls);
  EXPECT_TRUE(config.server.tls_cert_path.empty());
  EXPECT_TRUE(config.server.tls_key_path.empty());
  EXPECT_FALSE(config.server.require_client_cert);
}

TEST(Config, RejectsServerTlsMissingCertOrKey) {
  EngineConfig config = default_config();
  config.server.use_tls = true;
  config.server.tls_cert_path = "/etc/ssl/server.pem";
  // tls_key_path left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);

  config = default_config();
  config.server.use_tls = true;
  config.server.tls_key_path = "/etc/ssl/server.key";
  // tls_cert_path left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsServerRequireClientCertWithoutTls) {
  EngineConfig config = default_config();
  config.server.require_client_cert = true;
  config.server.tls_client_ca_cert_path = "/etc/ssl/client_ca.pem";
  // use_tls left false.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, RejectsServerRequireClientCertMissingCaPath) {
  EngineConfig config = default_config();
  config.server.use_tls = true;
  config.server.tls_cert_path = "/etc/ssl/server.pem";
  config.server.tls_key_path = "/etc/ssl/server.key";
  config.server.require_client_cert = true;
  // tls_client_ca_cert_path left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, ParsesServerAuthSection) {
  const std::string yaml = R"(
server:
  auth_enabled: true
  auth_token: super-secret-token
)";
  const EngineConfig config = parse_config(yaml);
  EXPECT_TRUE(config.server.auth_enabled);
  EXPECT_EQ(config.server.auth_token, "super-secret-token");
  EXPECT_NO_THROW((void)(validate_config(config)));
}

TEST(Config, ServerAuthDefaultsToDisabled) {
  const EngineConfig config = default_config();
  EXPECT_FALSE(config.server.auth_enabled);
  EXPECT_TRUE(config.server.auth_token.empty());
}

TEST(Config, RejectsServerAuthEnabledWithoutToken) {
  EngineConfig config = default_config();
  config.server.auth_enabled = true;
  // auth_token left empty.
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
}

TEST(Config, LoadConfigFileRejectsMissingPath) {
  EXPECT_THROW((void)(load_config_file("/nonexistent/kernellake.yaml")), ConfigurationError);
}

}  // namespace
}  // namespace kernellake
