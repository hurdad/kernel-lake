#include "kernellake/unitycatalog/unity_catalog_source_resolver.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"
#include "loopback_http_server.hpp"

namespace kernellake::unitycatalog {
namespace {

namespace fs = std::filesystem;

using kernellake::http_ok_json;
using kernellake::LoopbackHttpServer;

UnityCatalogSourceResolver make_resolver(UnityCatalogSection instances = {}, DeltaSection delta = {},
                                         S3Section s3 = {}, GcsSection gcs = {}, AzureSection azure = {}) {
  return UnityCatalogSourceResolver(std::move(instances), std::move(delta), std::move(s3), std::move(gcs),
                                    std::move(azure));
}

TEST(UnityCatalogSourceResolver, CanResolveOnlyClaimsUnityCatalogSchemeSources) {
  UnityCatalogSourceResolver resolver = make_resolver();
  EXPECT_TRUE(resolver.can_resolve({"unitycatalog://prod.main.db.orders"}));
  EXPECT_FALSE(resolver.can_resolve({"/data/sales.parquet"}));
  EXPECT_FALSE(resolver.can_resolve({"s3://bucket/table/"}));
  EXPECT_FALSE(resolver.can_resolve({"iceberg://prod.db.orders"}));
  EXPECT_FALSE(resolver.can_resolve({}));
  EXPECT_FALSE(
      resolver.can_resolve({"unitycatalog://prod.main.db.orders", "unitycatalog://prod.main.db.other"}));
}

TEST(UnityCatalogSourceResolver, ThrowsOnUnknownInstance) {
  UnityCatalogSourceResolver resolver = make_resolver();
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {})),
               ConfigurationError);
}

TEST(UnityCatalogSourceResolver, ThrowsOnTooFewQualifiedNameParts) {
  UnityCatalogInstanceSection instance;
  instance.uc_url = "http://localhost:8080/api/2.1/unity-catalog";
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;
  UnityCatalogSourceResolver resolver = make_resolver(instances);
  LocalObjectStore store;
  // "prod.db.orders" has only 3 dot-separated parts (instance + catalog +
  // table, no schema level) -- Unity Catalog's own naming is always
  // exactly catalog.schema.table, so this is rejected rather than guessed
  // at.
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod.db.orders"}, {})), StorageError);
}

TEST(UnityCatalogSourceResolver, ThrowsOnTooManyQualifiedNameParts) {
  UnityCatalogInstanceSection instance;
  instance.uc_url = "http://localhost:8080/api/2.1/unity-catalog";
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;
  UnityCatalogSourceResolver resolver = make_resolver(instances);
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod.main.db.schema.orders"}, {})),
               StorageError);
}

TEST(UnityCatalogSourceResolver, ThrowsOnEmptyQualifiedNamePart) {
  UnityCatalogInstanceSection instance;
  instance.uc_url = "http://localhost:8080/api/2.1/unity-catalog";
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;
  UnityCatalogSourceResolver resolver = make_resolver(instances);
  LocalObjectStore store;
  EXPECT_THROW((void)(resolver.resolve(store, {"unitycatalog://prod..db.orders"}, {})), StorageError);
}

UnityCatalogSection single_instance_config(const std::string& uc_url) {
  UnityCatalogInstanceSection instance;
  instance.uc_url = uc_url;
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;
  return instances;
}

constexpr const char* kGcsTableInfoJson = R"({
  "table_id": "table-uuid-gcs",
  "table_type": "EXTERNAL",
  "data_source_format": "PARQUET",
  "storage_location": "gs://warehouse/db/orders"
})";

constexpr const char* kAzureTableInfoJson = R"({
  "table_id": "table-uuid-azure",
  "table_type": "EXTERNAL",
  "data_source_format": "PARQUET",
  "storage_location": "abfss://container@account.dfs.core.windows.net/db/orders"
})";

// UnityCatalogClient::get_temporary_table_credentials() itself only throws
// when a cloud's credentials sub-object is present but missing its own
// required nested field entirely (see unity_catalog_client_test.cpp's
// GetTemporaryTableCredentialsThrowsWhenGcpOauthTokenFieldMissing) -- this
// is a genuinely different case: the field is present but empty, which the
// client happily parses and returns. The resolver's own check
// (unity_catalog_source_resolver.cpp:150-154) is what rejects it.
TEST(UnityCatalogSourceResolver, ThrowsWhenGcsVendedCredentialsCarryNoGcpOauthToken) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(2);
  server.respond(
      {http_ok_json(kGcsTableInfoJson), http_ok_json(R"({"gcp_oauth_token": {"oauth_token": ""}})")},
      &requests);

  UnityCatalogSourceResolver resolver = make_resolver(single_instance_config(server.base_url()));
  LocalObjectStore store;
  try {
    (void)resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {});
    FAIL() << "expected resolve() to throw";
  } catch (const StorageError& e) {
    EXPECT_NE(std::string(e.what()).find("gcp_oauth_token"), std::string::npos) << e.what();
  }
  server.join();
}

// Mirrors ThrowsWhenGcsVendedCredentialsCarryNoGcpOauthToken exactly, for
// the azure_sas_token sibling check (unity_catalog_source_resolver.cpp
// :157-163).
TEST(UnityCatalogSourceResolver, ThrowsWhenAzureVendedCredentialsCarryNoSasToken) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(2);
  server.respond({http_ok_json(kAzureTableInfoJson),
                  http_ok_json(R"({"azure_user_delegation_sas": {"sas_token": ""}})")},
                 &requests);

  UnityCatalogSourceResolver resolver = make_resolver(single_instance_config(server.base_url()));
  LocalObjectStore store;
  try {
    (void)resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {});
    FAIL() << "expected resolve() to throw";
  } catch (const StorageError& e) {
    EXPECT_NE(std::string(e.what()).find("azure_sas_token"), std::string::npos) << e.what();
  }
  server.join();
}

constexpr const char* kS3TableInfoJsonForCredentialCheck = R"({
  "table_id": "table-uuid-s3",
  "table_type": "EXTERNAL",
  "data_source_format": "PARQUET",
  "storage_location": "s3://warehouse/db/orders"
})";

// Regression test: unlike the GCS/Azure branches right next to it in
// unity_catalog_source_resolver.cpp, the S3 branch used to build an
// S3ObjectStore straight from credentials.access_key_id/secret_access_key
// with no empty check at all -- a present-but-empty aws_temp_credentials
// response (the client itself only rejects a field missing entirely, see
// unity_catalog_client_test.cpp) would silently produce a non-functional
// S3ObjectStore instead of failing clearly here.
TEST(UnityCatalogSourceResolver, ThrowsWhenS3VendedCredentialsCarryNoAccessKeyId) {
  LoopbackHttpServer server;
  std::vector<std::string> requests(2);
  server.respond(
      {http_ok_json(kS3TableInfoJsonForCredentialCheck),
       http_ok_json(
           R"({"aws_temp_credentials": {"access_key_id": "", "secret_access_key": "", "session_token": ""}})")},
      &requests);

  UnityCatalogSourceResolver resolver = make_resolver(single_instance_config(server.base_url()));
  LocalObjectStore store;
  try {
    (void)resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {});
    FAIL() << "expected resolve() to throw";
  } catch (const StorageError& e) {
    EXPECT_NE(std::string(e.what()).find("access_key_id"), std::string::npos) << e.what();
  }
  server.join();
}

// There's no real GCS endpoint in this test environment, so the actual
// file listing resolve() attempts once it's past credential-fetching must
// fail -- what matters here is *how* it fails: endpoint_override/
// retry_limit_seconds point GcsObjectStore at a closed local port instead
// of the real googleapis.com (a real network call would either hang behind
// GCS's own default up-to-15-minute retry policy or need real internet
// access), so the failure is fast and deterministic. Asserting the
// resulting StorageError is "gcs: failed to list ..." (generic_fs_list()'s
// own backend_label, distinct from "s3:"/"azure:") is direct evidence
// GcsObjectStore -- not some other store type -- was actually constructed
// and used; asserting it does NOT mention gcp_oauth_token confirms
// dispatch got past the credential check in ThrowsWhenGcsVendedCredentials
// CarryNoGcpOauthToken above and reached real (attempted) I/O.
TEST(UnityCatalogSourceResolver, DispatchesGcsStorageLocationToGcsObjectStore) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(kGcsTableInfoJson),
                  http_ok_json(R"({"gcp_oauth_token": {"oauth_token": "ya29.vended-token"}})")});

  GcsSection gcs;
  gcs.options.endpoint_override = "127.0.0.1:1";  // nothing listens here
  gcs.options.scheme = "http";
  gcs.options.retry_limit_seconds = 1.0;  // bound retries -- GCS's own default is up to 15 minutes

  UnityCatalogSourceResolver resolver =
      make_resolver(single_instance_config(server.base_url()), {}, {}, gcs, {});
  LocalObjectStore store;
  try {
    (void)resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {});
    FAIL() << "expected resolve() to throw once it tries to actually list gs://... with no reachable "
              "GCS endpoint";
  } catch (const StorageError& e) {
    const std::string message = e.what();
    EXPECT_EQ(message.find("gcp_oauth_token"), std::string::npos) << message;
    EXPECT_NE(message.find("gcs:"), std::string::npos) << message;
  }
  server.join();
}

// Mirrors DispatchesGcsStorageLocationToGcsObjectStore exactly, for the
// Azure dispatch branch.
TEST(UnityCatalogSourceResolver, DispatchesAzureStorageLocationToAzureObjectStore) {
  LoopbackHttpServer server;
  server.respond(
      {http_ok_json(kAzureTableInfoJson),
       http_ok_json(R"({"azure_user_delegation_sas": {"sas_token": "sv=2024-01-01&sig=vended"}})")});

  AzureSection azure;
  azure.options.account_name = "devstoreaccount1";
  azure.options.blob_storage_authority = "127.0.0.1:1";  // nothing listens here
  azure.options.dfs_storage_authority = "127.0.0.1:1";
  azure.options.blob_storage_scheme = "http";
  azure.options.dfs_storage_scheme = "http";

  UnityCatalogSourceResolver resolver =
      make_resolver(single_instance_config(server.base_url()), {}, {}, {}, azure);
  LocalObjectStore store;
  try {
    (void)resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {});
    FAIL() << "expected resolve() to throw once it tries to actually list abfss://... with no reachable "
              "Azure endpoint";
  } catch (const StorageError& e) {
    const std::string message = e.what();
    EXPECT_EQ(message.find("azure_sas_token"), std::string::npos) << message;
    EXPECT_NE(message.find("azure:"), std::string::npos) << message;
  }
  server.join();
}

// data_source_format "ICEBERG" (unity_catalog_source_resolver.cpp
// :185-198): no temporary-credential fetch on this path at all (see the
// resolver's own class comment), so only one HTTP request hits the fake
// Unity Catalog server -- the second, to the Iceberg REST dispatch, must
// land on "{uc_url}/iceberg", proving IcebergCatalogSection::catalog_uri is
// built by appending "/iceberg" to the configured instance's own uc_url,
// not some other URL. Both requests land on the same fake server (the same
// base_url is used for both), the same way a real Unity Catalog server
// fronts its own Iceberg-REST-compatible endpoint under its own host.
TEST(UnityCatalogSourceResolver, DispatchesIcebergFormatWithCorrectCatalogUriPrefixAndBearerToken) {
  constexpr const char* kIcebergTableInfoJson = R"({
    "table_id": "table-uuid-iceberg",
    "table_type": "EXTERNAL",
    "data_source_format": "ICEBERG",
    "storage_location": "s3://warehouse/db/orders"
  })";
  // No "current-snapshot-id"/"snapshots" -- resolves to zero files without
  // needing any further (manifest-list) requests, keeping this test
  // scoped to just the dispatch/URL-construction behavior.
  constexpr const char* kIcebergLoadTableResultJson = R"({
    "metadata": {
      "format-version": 2,
      "location": "s3://warehouse/db/orders",
      "current-schema-id": 0,
      "schemas": [{"schema-id": 0, "fields": [{"id": 1, "name": "id", "required": true, "type": "long"}]}]
    }
  })";

  LoopbackHttpServer server;
  std::vector<std::string> requests(2);
  server.respond({http_ok_json(kIcebergTableInfoJson), http_ok_json(kIcebergLoadTableResultJson)}, &requests);

  UnityCatalogInstanceSection instance;
  instance.uc_url = server.base_url();
  instance.credentials_kind = "bearer_token";
  instance.bearer_token = "static-secret-token";
  UnityCatalogSection instances;
  instances.instances["prod"] = instance;

  UnityCatalogSourceResolver resolver = make_resolver(instances);
  LocalObjectStore store;
  const ResolvedTable resolved = resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {});
  server.join();

  EXPECT_TRUE(resolved.files.empty());
  ASSERT_EQ(resolved.schema.field_count(), 1u);

  EXPECT_NE(requests[0].find("GET /tables/main.db.orders"), std::string::npos);
  // "main" (the qualified name's catalog part) as the REST Catalog API's
  // own "{prefix}" path segment, under "/iceberg" appended to uc_url.
  EXPECT_NE(requests[1].find("GET /iceberg/v1/main/namespaces/db/tables/orders"), std::string::npos);
  EXPECT_NE(requests[1].find("Authorization: Bearer static-secret-token"), std::string::npos);
}

// Regression test for the exact dangling-string_view bug
// strip_file_scheme()'s own comment (unity_catalog_source_resolver.cpp
// :63-79) describes: only exercised via a schemeless path until now. A
// real local directory with a real Parquet file, addressed through an
// explicit "file://" storage_location, must resolve successfully -- if
// the "file://" prefix weren't actually stripped, LocalObjectStore would
// try to open a path literally starting with "file://" and fail.
TEST(UnityCatalogSourceResolver, StripsExplicitFileSchemePrefixFromStorageLocation) {
  const fs::path dir =
      fs::temp_directory_path() / fs::path("kernellake_unity_catalog_source_resolver_test_" +
                                           std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  fs::create_directories(dir);

  arrow::Int64Builder id_builder;
  ASSERT_TRUE(id_builder.Append(1).ok());
  ASSERT_TRUE(id_builder.Append(2).ok());
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(schema, {id_array});
  auto sink_result = arrow::io::FileOutputStream::Open((dir / "data-0.parquet").string());
  ASSERT_TRUE(sink_result.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, 2).ok());

  const std::string table_info_json = fmt::format(R"({{
    "table_id": "table-uuid-file",
    "table_type": "EXTERNAL",
    "data_source_format": "PARQUET",
    "storage_location": "file://{}"
  }})",
                                                  dir.string());

  LoopbackHttpServer server;
  server.respond({http_ok_json(table_info_json)});

  UnityCatalogSourceResolver resolver = make_resolver(single_instance_config(server.base_url()));
  LocalObjectStore store;
  const ResolvedTable resolved = resolver.resolve(store, {"unitycatalog://prod.main.db.orders"}, {});
  server.join();

  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_EQ(resolved.files[0].metadata.row_count, 2);
  ASSERT_EQ(resolved.schema.field_count(), 1u);
  EXPECT_EQ(resolved.schema.field(0).name, "id");

  fs::remove_all(dir);
}

}  // namespace
}  // namespace kernellake::unitycatalog
