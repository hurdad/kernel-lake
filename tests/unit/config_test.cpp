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

TEST(Config, LoadConfigFileRejectsMissingPath) {
  EXPECT_THROW((void)(load_config_file("/nonexistent/kernellake.yaml")), ConfigurationError);
}

}  // namespace
}  // namespace kernellake
