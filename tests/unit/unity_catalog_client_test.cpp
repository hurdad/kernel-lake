#include "kernellake/unitycatalog/unity_catalog_client.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "loopback_http_server.hpp"

namespace kernellake::unitycatalog {
namespace {

using kernellake::LoopbackHttpServer;
using kernellake::http_ok_json;
using kernellake::http_status;

constexpr const char* kTableInfoJson = R"({
  "table_id": "table-uuid-1",
  "table_type": "EXTERNAL",
  "data_source_format": "DELTA",
  "storage_location": "s3://warehouse/db/orders",
  "columns": [
    {"name": "order_id", "type_name": "LONG", "type_json": "\"LONG\"", "nullable": false, "position": 0},
    {"name": "amount", "type_name": "DOUBLE", "type_json": "\"DOUBLE\"", "nullable": true, "position": 1}
  ]
})";

constexpr const char* kTemporaryCredentialsJson = R"({
  "aws_temp_credentials": {
    "access_key_id": "AKIA-VENDED",
    "secret_access_key": "vended-secret",
    "session_token": "vended-session-token"
  },
  "expiration_time": 1999999999999,
  "url": "s3://warehouse/db/orders"
})";

UnityCatalogInstanceSection instance_config(const std::string& base_url) {
  UnityCatalogInstanceSection config;
  config.uc_url = base_url;
  return config;
}

TEST(UnityCatalogClient, GetTableParsesTableInfoAndColumns) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(kTableInfoJson)}, &requests);

  UnityCatalogClient client(instance_config(server.base_url()));
  const UnityCatalogTableInfo info = client.get_table("main", "db", "orders");
  server.join();

  EXPECT_EQ(info.table_id, "table-uuid-1");
  EXPECT_EQ(info.table_type, "EXTERNAL");
  EXPECT_EQ(info.data_source_format, "DELTA");
  EXPECT_EQ(info.storage_location, "s3://warehouse/db/orders");
  ASSERT_EQ(info.columns.size(), 2u);
  EXPECT_EQ(info.columns[0].name, "order_id");
  EXPECT_EQ(info.columns[0].type_name, "LONG");
  EXPECT_FALSE(info.columns[0].nullable);
  EXPECT_EQ(info.columns[1].name, "amount");
  EXPECT_TRUE(info.columns[1].nullable);

  EXPECT_NE(requests[0].find("GET /tables/main.db.orders"), std::string::npos);
}

TEST(UnityCatalogClient, GetTableSendsBearerToken) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(kTableInfoJson)}, &requests);

  UnityCatalogInstanceSection config = instance_config(server.base_url());
  config.credentials_kind = "bearer_token";
  config.bearer_token = "static-secret-token";
  UnityCatalogClient client(config);
  (void)client.get_table("main", "db", "orders");
  server.join();

  EXPECT_NE(requests[0].find("Authorization: Bearer static-secret-token"), std::string::npos);
}

TEST(UnityCatalogClient, GetTablePerformsOauth2ClientCredentialsFlowAgainstConfiguredTokenEndpoint) {
  LoopbackHttpServer token_server;
  LoopbackHttpServer table_server;
  std::vector<std::string> token_requests(1);
  std::vector<std::string> table_requests(1);
  const std::string token_response = R"({"access_token": "minted-token", "expires_in": 3600})";
  token_server.respond({http_ok_json(token_response)}, &token_requests);
  table_server.respond({http_ok_json(kTableInfoJson)}, &table_requests);

  UnityCatalogInstanceSection config = instance_config(table_server.base_url());
  config.oauth2_token_endpoint = token_server.base_url() + "/oidc/v1/token";
  config.credentials_kind = "oauth2_client_credentials";
  config.oauth2_client_id = "my-client";
  config.oauth2_client_secret = "my-secret";
  UnityCatalogClient client(config);
  const UnityCatalogTableInfo info = client.get_table("main", "db", "orders");
  token_server.join();
  table_server.join();

  EXPECT_EQ(info.data_source_format, "DELTA");
  EXPECT_NE(token_requests[0].find("POST /oidc/v1/token"), std::string::npos);
  EXPECT_NE(token_requests[0].find("grant_type=client_credentials"), std::string::npos);
  EXPECT_NE(token_requests[0].find("client_id=my-client"), std::string::npos);
  EXPECT_NE(token_requests[0].find("client_secret=my-secret"), std::string::npos);
  EXPECT_NE(table_requests[0].find("Authorization: Bearer minted-token"), std::string::npos);
}

TEST(UnityCatalogClient, GetTemporaryTableCredentialsParsesAwsCredentials) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(kTemporaryCredentialsJson)}, &requests);

  UnityCatalogInstanceSection config = instance_config(server.base_url());
  config.credentials_kind = "bearer_token";
  config.bearer_token = "static-secret-token";
  UnityCatalogClient client(config);
  const UnityCatalogTemporaryCredentials credentials =
      client.get_temporary_table_credentials("table-uuid-1", "READ");
  server.join();

  EXPECT_EQ(credentials.access_key_id, "AKIA-VENDED");
  EXPECT_EQ(credentials.secret_access_key, "vended-secret");
  EXPECT_EQ(credentials.session_token, "vended-session-token");
  EXPECT_NE(requests[0].find("POST /temporary-table-credentials"), std::string::npos);
  EXPECT_NE(requests[0].find("Authorization: Bearer static-secret-token"), std::string::npos);
  EXPECT_NE(requests[0].find(R"("table_id":"table-uuid-1")"), std::string::npos);
  EXPECT_NE(requests[0].find(R"("operation":"READ")"), std::string::npos);
}

TEST(UnityCatalogClient, GetTemporaryTableCredentialsThrowsWhenAwsCredentialsMissing) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"gcp_oauth_token": {"oauth_token": "x"}})")});

  UnityCatalogClient client(instance_config(server.base_url()));
  EXPECT_THROW((void)(client.get_temporary_table_credentials("table-uuid-1", "READ")), StorageError);
  server.join();
}

TEST(UnityCatalogClient, ThrowsOnNon2xxStatus) {
  LoopbackHttpServer server;
  server.respond({http_status(404, "Not Found", R"({"error_code": "TABLE_NOT_FOUND"})")});

  UnityCatalogClient client(instance_config(server.base_url()));
  EXPECT_THROW((void)(client.get_table("main", "db", "missing")), StorageError);
  server.join();
}

TEST(UnityCatalogClient, ThrowsOnMalformedJsonResponse) {
  LoopbackHttpServer server;
  server.respond({http_ok_json("this is not json")});

  UnityCatalogClient client(instance_config(server.base_url()));
  EXPECT_THROW((void)(client.get_table("main", "db", "orders")), StorageError);
  server.join();
}

TEST(UnityCatalogClient, ThrowsWhenStorageLocationFieldIsMissing) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"table_id": "t1", "table_type": "EXTERNAL"})")});

  UnityCatalogClient client(instance_config(server.base_url()));
  EXPECT_THROW((void)(client.get_table("main", "db", "orders")), StorageError);
  server.join();
}

TEST(UnityCatalogClient, ThrowsOnConnectionFailure) {
  // Nothing listens on this port: curl_easy_perform() should fail with
  // CURLE_COULDNT_CONNECT before any HTTP status even exists.
  UnityCatalogClient client(instance_config("http://127.0.0.1:1"));
  EXPECT_THROW((void)(client.get_table("main", "db", "orders")), StorageError);
}

}  // namespace
}  // namespace kernellake::unitycatalog
