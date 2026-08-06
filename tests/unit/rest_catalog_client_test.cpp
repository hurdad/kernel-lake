#include "kernellake/iceberg/rest_catalog_client.hpp"

#include <arpa/inet.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <thread>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake::iceberg {
namespace {

// A minimal single-connection-at-a-time loopback HTTP stub, purpose-built
// for this test file: not a general HTTP server. RestCatalogClient's own
// requests are small enough (headers + a short JSON/form body) to always
// land in one TCP segment on loopback, so a single recv() into a generous
// buffer is enough to capture the whole request -- a real HTTP server would
// need to loop on recv() and parse Content-Length, this doesn't.
class LoopbackHttpServer {
 public:
  LoopbackHttpServer() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    ::listen(listen_fd_, /*backlog=*/4);

    timeval timeout{5, 0};
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  }

  ~LoopbackHttpServer() {
    if (thread_.joinable()) {
      thread_.join();
    }
    ::close(listen_fd_);
  }

  [[nodiscard]] std::string base_url() const { return fmt::format("http://127.0.0.1:{}", port_); }

  // Accepts `response_count` connections in order, one per element of
  // `responses`, capturing each raw request into the matching slot of
  // `captured_requests` (if non-null). Runs on a background thread since
  // the calling test's main thread is blocked inside the client call that
  // connects to this server.
  void respond(std::vector<std::string> responses, std::vector<std::string>* captured_requests = nullptr) {
    thread_ = std::thread([this, responses = std::move(responses), captured_requests] {
      for (size_t i = 0; i < responses.size(); ++i) {
        const int conn_fd = ::accept(listen_fd_, nullptr, nullptr);
        if (conn_fd < 0) {
          return;
        }
        timeval timeout{5, 0};
        ::setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        std::string request(65536, '\0');
        const ssize_t received = ::recv(conn_fd, request.data(), request.size(), 0);
        request.resize(received > 0 ? static_cast<size_t>(received) : 0);
        if (captured_requests != nullptr) {
          (*captured_requests)[i] = request;
        }

        const std::string& response = responses[i];
        ::send(conn_fd, response.data(), response.size(), 0);
        ::close(conn_fd);
      }
    });
  }

  void join() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread thread_;
};

std::string http_ok_json(const std::string& json_body) {
  return fmt::format(
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: "
      "close\r\n\r\n{}",
      json_body.size(), json_body);
}

std::string http_status(int status_code, const std::string& reason, const std::string& body = "") {
  return fmt::format("HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", status_code,
                     reason, body.size(), body);
}

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
