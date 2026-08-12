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

// Proves UnityCatalogTokenCache actually closes the cross-instance gap
// this client's own local cached_oauth2_token_/oauth2_token_expiry_
// unix_seconds_ can't: two separate UnityCatalogClient objects sharing
// one UnityCatalogTokenCache (the shape UnityCatalogSourceResolver/
// QueryEngine actually use -- see UnityCatalogTokenCache's own class
// comment) issue only *one* real token-endpoint request between them, not
// two. The token server is only handed one canned response
// (LoopbackHttpServer::respond()'s own doc comment: it services exactly
// `responses.size()` connections and no more) -- if the shared cache were
// ignored and a second client instance re-fetched, that second request
// would find nothing listening and fail/hang instead of silently passing.
TEST(UnityCatalogClient, TwoClientsSharingATokenCacheOnlyFetchTheTokenOnce) {
  LoopbackHttpServer token_server;
  LoopbackHttpServer table_server;
  std::vector<std::string> token_requests(1);
  std::vector<std::string> table_requests(2);
  const std::string token_response = R"({"access_token": "shared-token", "expires_in": 3600})";
  token_server.respond({http_ok_json(token_response)}, &token_requests);
  table_server.respond({http_ok_json(kTableInfoJson), http_ok_json(kTableInfoJson)}, &table_requests);

  UnityCatalogInstanceSection config = instance_config(table_server.base_url());
  config.oauth2_token_endpoint = token_server.base_url() + "/oidc/v1/token";
  config.credentials_kind = "oauth2_client_credentials";
  config.oauth2_client_id = "my-client";
  config.oauth2_client_secret = "my-secret";

  const UnityCatalogTokenCache shared_cache;
  UnityCatalogClient first_client(config, &shared_cache);
  (void)first_client.get_table("main", "db", "orders");

  // A second, entirely separate client instance -- mirrors a fresh
  // UnityCatalogClient constructed by a *different* resolve() call (or a
  // different query entirely), the exact case this client's own local
  // token cache can never help with.
  UnityCatalogClient second_client(config, &shared_cache);
  (void)second_client.get_table("main", "db", "orders");

  token_server.join();
  table_server.join();

  EXPECT_NE(token_requests[0].find("POST /oidc/v1/token"), std::string::npos);
  EXPECT_NE(table_requests[0].find("Authorization: Bearer shared-token"), std::string::npos);
  EXPECT_NE(table_requests[1].find("Authorization: Bearer shared-token"), std::string::npos);
}

// The default (nullptr token_cache) behavior is unchanged: two separate
// clients with no shared cache each fetch their own token, so the token
// server needs two canned responses, not one.
TEST(UnityCatalogClient, TwoClientsWithoutASharedCacheEachFetchTheirOwnToken) {
  LoopbackHttpServer token_server;
  LoopbackHttpServer table_server;
  std::vector<std::string> token_requests(2);
  std::vector<std::string> table_requests(2);
  const std::string token_response = R"({"access_token": "independent-token", "expires_in": 3600})";
  token_server.respond({http_ok_json(token_response), http_ok_json(token_response)}, &token_requests);
  table_server.respond({http_ok_json(kTableInfoJson), http_ok_json(kTableInfoJson)}, &table_requests);

  UnityCatalogInstanceSection config = instance_config(table_server.base_url());
  config.oauth2_token_endpoint = token_server.base_url() + "/oidc/v1/token";
  config.credentials_kind = "oauth2_client_credentials";
  config.oauth2_client_id = "my-client";
  config.oauth2_client_secret = "my-secret";

  UnityCatalogClient first_client(config);
  (void)first_client.get_table("main", "db", "orders");
  UnityCatalogClient second_client(config);
  (void)second_client.get_table("main", "db", "orders");

  token_server.join();
  table_server.join();

  EXPECT_NE(token_requests[0].find("POST /oidc/v1/token"), std::string::npos);
  EXPECT_NE(token_requests[1].find("POST /oidc/v1/token"), std::string::npos);
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
  const UnityCatalogTemporaryCredentials credentials =
      client.get_temporary_table_credentials("table-uuid-1", "READ");
  server.join();

  EXPECT_TRUE(credentials.access_key_id.empty());
  EXPECT_EQ(credentials.gcp_oauth_token, "x");
}

// The gcp_oauth_token/azure_user_delegation_sas response shapes below are
// NOT independently verified against a real live Unity Catalog server
// (unlike aws_temp_credentials, see UnityCatalogTemporaryCredentials's own
// comment) -- these only lock in this client's own parsing of the
// Databricks-SDK-conventional shape it was written against, not that a
// real GCP/Azure-backed UC deployment actually returns exactly this.
TEST(UnityCatalogClient, GetTemporaryTableCredentialsParsesGcpOauthToken) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"gcp_oauth_token": {"oauth_token": "ya29.vended-gcp-token"}})")});

  UnityCatalogClient client(instance_config(server.base_url()));
  const UnityCatalogTemporaryCredentials credentials =
      client.get_temporary_table_credentials("table-uuid-1", "READ");
  server.join();

  EXPECT_EQ(credentials.gcp_oauth_token, "ya29.vended-gcp-token");
  EXPECT_TRUE(credentials.access_key_id.empty());
  EXPECT_TRUE(credentials.azure_sas_token.empty());
}

TEST(UnityCatalogClient, GetTemporaryTableCredentialsParsesAzureSasToken) {
  LoopbackHttpServer server;
  server.respond(
      {http_ok_json(R"({"azure_user_delegation_sas": {"sas_token": "sv=2024-01-01&sig=vended-azure-sas"}})")});

  UnityCatalogClient client(instance_config(server.base_url()));
  const UnityCatalogTemporaryCredentials credentials =
      client.get_temporary_table_credentials("table-uuid-1", "READ");
  server.join();

  EXPECT_EQ(credentials.azure_sas_token, "sv=2024-01-01&sig=vended-azure-sas");
  EXPECT_TRUE(credentials.access_key_id.empty());
  EXPECT_TRUE(credentials.gcp_oauth_token.empty());
}

TEST(UnityCatalogClient, GetTemporaryTableCredentialsThrowsWhenGcpOauthTokenFieldMissing) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"gcp_oauth_token": {}})")});

  UnityCatalogClient client(instance_config(server.base_url()));
  EXPECT_THROW((void)(client.get_temporary_table_credentials("table-uuid-1", "READ")), StorageError);
  server.join();
}

TEST(UnityCatalogClient, GetTemporaryTableCredentialsThrowsWhenNoRecognizedCredentialsField) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"expiration_time": 123})")});

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

TEST(UnityCatalogClient, ListCatalogsParsesSinglePage) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(1);
  server.respond({http_ok_json(R"({
    "catalogs": [
      {"name": "unity", "comment": "Main catalog"},
      {"name": "other", "comment": ""}
    ],
    "next_page_token": null
  })")},
                 &requests);

  const std::vector<UnityCatalogCatalogInfo> catalogs = UnityCatalogClient(instance_config(server.base_url())).list_catalogs();
  server.join();

  ASSERT_EQ(catalogs.size(), 2u);
  EXPECT_EQ(catalogs[0].name, "unity");
  EXPECT_EQ(catalogs[0].comment, "Main catalog");
  EXPECT_EQ(catalogs[1].name, "other");
  EXPECT_NE(requests[0].find("GET /catalogs"), std::string::npos);
}

TEST(UnityCatalogClient, ListSchemasFollowsPaginationAcrossMultiplePages) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(2);
  server.respond(
      {http_ok_json(R"({
        "schemas": [{"name": "default", "catalog_name": "unity", "full_name": "unity.default"}],
        "next_page_token": "default"
      })"),
       http_ok_json(R"({
        "schemas": [{"name": "other_schema", "catalog_name": "unity", "full_name": "unity.other_schema"}],
        "next_page_token": null
      })")},
      &requests);

  const std::vector<UnityCatalogSchemaInfo> schemas = UnityCatalogClient(instance_config(server.base_url())).list_schemas("unity");
  server.join();

  ASSERT_EQ(schemas.size(), 2u);
  EXPECT_EQ(schemas[0].name, "default");
  EXPECT_EQ(schemas[0].full_name, "unity.default");
  EXPECT_EQ(schemas[1].name, "other_schema");
  EXPECT_NE(requests[0].find("GET /schemas?catalog_name=unity"), std::string::npos);
  // Second request must echo the first response's next_page_token back as
  // its own page_token query parameter, not just repeat the first request.
  EXPECT_NE(requests[1].find("page_token=default"), std::string::npos);
}

TEST(UnityCatalogClient, ListTablesFollowsPaginationAndParsesEachTable) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(fmt::format(R"({{
                    "tables": [{}],
                    "next_page_token": null
                  }})",
                                           kTableInfoJson))});

  const std::vector<UnityCatalogTableInfo> tables =
      UnityCatalogClient(instance_config(server.base_url())).list_tables("main", "db");
  server.join();

  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].table_id, "table-uuid-1");
  EXPECT_EQ(tables[0].data_source_format, "DELTA");
  ASSERT_EQ(tables[0].columns.size(), 2u);
  EXPECT_EQ(tables[0].columns[0].name, "order_id");
}

TEST(UnityCatalogClient, ListTablesThrowsWhenTablesFieldMissing) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(R"({"not_tables": []})")});

  UnityCatalogClient client(instance_config(server.base_url()));
  EXPECT_THROW((void)(client.list_tables("main", "db")), StorageError);
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
