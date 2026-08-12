#include "kernellake/iceberg/rest_catalog_client.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "loopback_http_server.hpp"

namespace kernellake::iceberg {
namespace {

using kernellake::LoopbackHttpServer;
using kernellake::http_ok_json;
using kernellake::http_status;

constexpr const char* kLoadTableResultJson = R"({
  "metadata": {
    "format-version": 2,
    "location": "s3://warehouse/db/orders",
    "current-snapshot-id": 42,
    "snapshots": [
      {"snapshot-id": 41, "manifest-list": "s3://warehouse/db/orders/metadata/snap-41.avro"},
      {"snapshot-id": 42, "manifest-list": "s3://warehouse/db/orders/metadata/snap-42.avro"}
    ]
  }
})";

IcebergCatalogSection catalog_config(const std::string& base_url) {
  IcebergCatalogSection config;
  config.catalog_uri = base_url;
  return config;
}

TEST(IcebergTableMetadata, CurrentManifestListResolvesFromSnapshots) {
  IcebergTableMetadata metadata;
  metadata.current_snapshot_id = 42;
  metadata.snapshots = {{41, "snap-41.avro"}, {42, "snap-42.avro"}};
  EXPECT_EQ(metadata.current_manifest_list(), "snap-42.avro");
}

TEST(IcebergTableMetadata, CurrentManifestListIsNulloptWithoutACurrentSnapshot) {
  IcebergTableMetadata metadata;
  EXPECT_FALSE(metadata.current_manifest_list().has_value());
}

TEST(IcebergTableMetadata, CurrentManifestListIsNulloptWhenSnapshotIdIsUnknown) {
  IcebergTableMetadata metadata;
  metadata.current_snapshot_id = 99;
  metadata.snapshots = {{41, "snap-41.avro"}};
  EXPECT_FALSE(metadata.current_manifest_list().has_value());
}

TEST(IcebergRestCatalogClient, LoadTableMetadataParsesLocationAndSnapshots) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(kLoadTableResultJson)}, &requests);

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  const IcebergTableMetadata metadata = client.load_table_metadata({"db"}, "orders");
  server.join();

  EXPECT_EQ(metadata.location, "s3://warehouse/db/orders");
  EXPECT_EQ(metadata.format_version, 2);
  ASSERT_TRUE(metadata.current_snapshot_id.has_value());
  EXPECT_EQ(*metadata.current_snapshot_id, 42);
  ASSERT_EQ(metadata.snapshots.size(), 2u);
  EXPECT_EQ(metadata.current_manifest_list(), "s3://warehouse/db/orders/metadata/snap-42.avro");

  EXPECT_NE(requests[0].find("GET /v1/namespaces/db/tables/orders"), std::string::npos);
  EXPECT_EQ(requests[0].find("Authorization:"), std::string::npos);
}

// A named delimiter (not the default unnamed R"(...)") is required here:
// the payload's "decimal(10,2)" value itself contains the literal
// characters ")\"", which would otherwise terminate an unnamed raw string
// early, silently truncating the fixture and misparsing everything after
// it as code.
constexpr const char* kLoadTableResultWithV2SchemasJson = R"json({
  "metadata": {
    "format-version": 2,
    "location": "s3://warehouse/db/orders",
    "current-schema-id": 1,
    "schemas": [
      {"schema-id": 0, "fields": [{"id": 1, "name": "old_col", "required": true, "type": "int"}]},
      {"schema-id": 1, "fields": [
        {"id": 1, "name": "id", "required": true, "type": "long"},
        {"id": 2, "name": "amount", "required": false, "type": "decimal(10,2)"}
      ]}
    ]
  }
})json";

TEST(IcebergRestCatalogClient, LoadTableMetadataParsesCurrentSchemaFromV2SchemasArray) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(kLoadTableResultWithV2SchemasJson)});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  const IcebergTableMetadata metadata = client.load_table_metadata({"db"}, "orders");
  server.join();

  ASSERT_EQ(metadata.schema_fields.size(), 2u);
  EXPECT_EQ(metadata.schema_fields[0].name, "id");
  EXPECT_EQ(metadata.schema_fields[0].type, "long");
  EXPECT_TRUE(metadata.schema_fields[0].required);
  EXPECT_EQ(metadata.schema_fields[1].name, "amount");
  EXPECT_EQ(metadata.schema_fields[1].type, "decimal(10,2)");
  EXPECT_FALSE(metadata.schema_fields[1].required);
}

constexpr const char* kLoadTableResultWithPartitionSpecsJson = R"json({
  "metadata": {
    "format-version": 2,
    "location": "s3://warehouse/db/orders",
    "current-schema-id": 0,
    "schemas": [{"schema-id": 0, "fields": [
      {"id": 1, "name": "id", "required": true, "type": "long"},
      {"id": 2, "name": "ts", "required": false, "type": "timestamp"}
    ]}],
    "partition-specs": [
      {"spec-id": 0, "fields": []},
      {"spec-id": 1, "fields": [
        {"source-id": 2, "field-id": 1000, "name": "ts_day", "transform": "day"}
      ]}
    ]
  }
})json";

TEST(IcebergRestCatalogClient, LoadTableMetadataParsesPartitionSpecs) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(kLoadTableResultWithPartitionSpecsJson)});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  const IcebergTableMetadata metadata = client.load_table_metadata({"db"}, "orders");
  server.join();

  ASSERT_EQ(metadata.partition_specs.size(), 2u);
  const IcebergPartitionSpec* spec0 = metadata.find_partition_spec(0);
  ASSERT_NE(spec0, nullptr);
  EXPECT_TRUE(spec0->fields.empty());

  const IcebergPartitionSpec* spec1 = metadata.find_partition_spec(1);
  ASSERT_NE(spec1, nullptr);
  ASSERT_EQ(spec1->fields.size(), 1u);
  EXPECT_EQ(spec1->fields[0].source_id, 2);
  EXPECT_EQ(spec1->fields[0].field_id, 1000);
  EXPECT_EQ(spec1->fields[0].name, "ts_day");
  EXPECT_EQ(spec1->fields[0].transform, "day");

  EXPECT_EQ(metadata.find_partition_spec(99), nullptr);
}

TEST(IcebergRestCatalogClient, LoadTableMetadataLeavesPartitionSpecsEmptyWhenAbsent) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(kLoadTableResultJson)});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  const IcebergTableMetadata metadata = client.load_table_metadata({"db"}, "orders");
  server.join();

  EXPECT_TRUE(metadata.partition_specs.empty());
  EXPECT_EQ(metadata.find_partition_spec(0), nullptr);
}

TEST(IcebergRestCatalogClient, LoadTableMetadataFallsBackToV1SchemaField) {
  constexpr const char* kV1Json = R"({
    "metadata": {
      "format-version": 1,
      "location": "s3://warehouse/db/orders",
      "schema": {"fields": [{"id": 1, "name": "id", "required": true, "type": "long"}]}
    }
  })";
  LoopbackHttpServer server;
  server.respond({http_ok_json(kV1Json)});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  const IcebergTableMetadata metadata = client.load_table_metadata({"db"}, "orders");
  server.join();

  ASSERT_EQ(metadata.schema_fields.size(), 1u);
  EXPECT_EQ(metadata.schema_fields[0].name, "id");
}

TEST(IcebergRestCatalogClient, LoadTableMetadataThrowsWhenCurrentSchemaIdIsUnmatched) {
  constexpr const char* kBadJson = R"({
    "metadata": {
      "format-version": 2,
      "location": "s3://warehouse/db/orders",
      "current-schema-id": 99,
      "schemas": [{"schema-id": 0, "fields": []}]
    }
  })";
  LoopbackHttpServer server;
  server.respond({http_ok_json(kBadJson)});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  EXPECT_THROW((void)(client.load_table_metadata({"db"}, "orders")), StorageError);
  server.join();
}

TEST(IcebergRestCatalogClient, LoadTableMetadataEncodesMultiLevelNamespace) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(kLoadTableResultJson)}, &requests);

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  (void)client.load_table_metadata({"db", "schema"}, "orders");
  server.join();

  // "db" + U+001F + "schema" percent-encoded as a single path segment.
  EXPECT_NE(requests[0].find("GET /v1/namespaces/db%1Fschema/tables/orders"), std::string::npos);
}

TEST(IcebergRestCatalogClient, LoadTableMetadataIncludesPrefixWhenConfigured) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(kLoadTableResultJson)}, &requests);

  IcebergCatalogSection config = catalog_config(server.base_url());
  config.prefix = "prod";
  IcebergRestCatalogClient client(config);
  (void)client.load_table_metadata({"db"}, "orders");
  server.join();

  EXPECT_NE(requests[0].find("GET /v1/prod/namespaces/db/tables/orders"), std::string::npos);
}

TEST(IcebergRestCatalogClient, LoadTableMetadataSendsBearerToken) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(kLoadTableResultJson)}, &requests);

  IcebergCatalogSection config = catalog_config(server.base_url());
  config.credentials_kind = "bearer_token";
  config.bearer_token = "static-secret-token";
  IcebergRestCatalogClient client(config);
  (void)client.load_table_metadata({"db"}, "orders");
  server.join();

  EXPECT_NE(requests[0].find("Authorization: Bearer static-secret-token"), std::string::npos);
}

TEST(IcebergRestCatalogClient, LoadTableMetadataPerformsOauth2ClientCredentialsFlow) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(2);
  const std::string token_response = R"({"access_token": "minted-token", "expires_in": 3600})";
  server.respond({http_ok_json(token_response), http_ok_json(kLoadTableResultJson)}, &requests);

  IcebergCatalogSection config = catalog_config(server.base_url());
  config.credentials_kind = "oauth2_client_credentials";
  config.oauth2_client_id = "my-client";
  config.oauth2_client_secret = "my-secret";
  IcebergRestCatalogClient client(config);
  const IcebergTableMetadata metadata = client.load_table_metadata({"db"}, "orders");
  server.join();

  EXPECT_EQ(metadata.location, "s3://warehouse/db/orders");
  EXPECT_NE(requests[0].find("POST /v1/oauth/tokens"), std::string::npos);
  EXPECT_NE(requests[0].find("grant_type=client_credentials"), std::string::npos);
  EXPECT_NE(requests[0].find("client_id=my-client"), std::string::npos);
  EXPECT_NE(requests[0].find("client_secret=my-secret"), std::string::npos);
  EXPECT_NE(requests[1].find("Authorization: Bearer minted-token"), std::string::npos);
}

TEST(IcebergRestCatalogClient, ThrowsOnNon2xxStatus) {
  LoopbackHttpServer server;
  server.respond({http_status(404, "Not Found", R"({"error": {"message": "no such table"}})")});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  EXPECT_THROW((void)(client.load_table_metadata({"db"}, "missing")), StorageError);
  server.join();
}

TEST(IcebergRestCatalogClient, ThrowsOnMalformedJsonResponse) {
  LoopbackHttpServer server;
  server.respond({http_ok_json("this is not json")});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  EXPECT_THROW((void)(client.load_table_metadata({"db"}, "orders")), StorageError);
  server.join();
}

TEST(IcebergRestCatalogClient, ThrowsWhenMetadataFieldIsMissing) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"not-metadata": true})")});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  EXPECT_THROW((void)(client.load_table_metadata({"db"}, "orders")), StorageError);
  server.join();
}

TEST(IcebergRestCatalogClient, ThrowsWhenLocationFieldIsMissing) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"metadata": {"format-version": 2}})")});

  IcebergRestCatalogClient client(catalog_config(server.base_url()));
  EXPECT_THROW((void)(client.load_table_metadata({"db"}, "orders")), StorageError);
  server.join();
}

TEST(IcebergRestCatalogClient, ThrowsOnConnectionFailure) {
  // Nothing listens on this port: curl_easy_perform() should fail with
  // CURLE_COULDNT_CONNECT before any HTTP status even exists.
  IcebergRestCatalogClient client(catalog_config("http://127.0.0.1:1"));
  EXPECT_THROW((void)(client.load_table_metadata({"db"}, "orders")), StorageError);
}

}  // namespace
}  // namespace kernellake::iceberg
