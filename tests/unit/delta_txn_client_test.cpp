#include "kernellake/delta/delta_txn_client.hpp"

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "delta_txn.grpc.pb.h"
#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/observability/query_tracing.hpp"

namespace kernellake::delta {
namespace {

class FakeDeltaTxnService final : public ::delta::txn::v1::DeltaTxnService::Service {
 public:
  bool get_table_should_fail = false;
  ::delta::txn::v1::GetTableResponse get_table_response;
  std::vector<::delta::txn::v1::ListActiveFilesResponse> list_active_files_responses;
  bool list_active_files_should_fail_stream = false;
  std::string last_seen_api_key;
  std::string last_seen_traceparent;

  grpc::Status GetTable(grpc::ServerContext* context, const ::delta::txn::v1::GetTableRequest* /*request*/,
                        ::delta::txn::v1::GetTableResponse* response) override {
    last_seen_api_key = extract_api_key(context);
    last_seen_traceparent = extract_header(context, "traceparent");
    if (get_table_should_fail) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "no such table");
    }
    *response = get_table_response;
    return grpc::Status::OK;
  }

  grpc::Status ListActiveFiles(
      grpc::ServerContext* context, const ::delta::txn::v1::ListActiveFilesRequest* /*request*/,
      grpc::ServerWriter<::delta::txn::v1::ListActiveFilesResponse>* writer) override {
    last_seen_api_key = extract_api_key(context);
    for (const ::delta::txn::v1::ListActiveFilesResponse& response : list_active_files_responses) {
      writer->Write(response);
    }
    if (list_active_files_should_fail_stream) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "boom");
    }
    return grpc::Status::OK;
  }

 private:
  static std::string extract_api_key(grpc::ServerContext* context) { return extract_header(context, "x-api-key"); }

  static std::string extract_header(grpc::ServerContext* context, const char* name) {
    const auto it = context->client_metadata().find(name);
    return it != context->client_metadata().end() ? std::string(it->second.data(), it->second.length()) : "";
  }
};

// A real in-process gRPC server (not a mock) bound to an ephemeral
// loopback port -- same "test against the real thing, not a fake
// abstraction" bar this project's other client tests hold to (see
// rest_catalog_client_test.cpp's raw-socket HTTP stub for the analogous
// Iceberg case; gRPC's own ServerBuilder makes a real service registration
// simpler here than reimplementing HTTP/2 framing by hand would be).
class LocalDeltaTxnServer {
 public:
  LocalDeltaTxnServer() {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    port_ = port;
  }

  ~LocalDeltaTxnServer() { server_->Shutdown(); }

  FakeDeltaTxnService& service() { return service_; }
  [[nodiscard]] std::string endpoint() const { return "127.0.0.1:" + std::to_string(port_); }

 private:
  FakeDeltaTxnService service_;
  std::unique_ptr<grpc::Server> server_;
  int port_ = 0;
};

DeltaSection endpoint_config(const std::string& endpoint) {
  DeltaSection config;
  config.grpc_endpoint = endpoint;
  return config;
}

TEST(DeltaTxnClient, GetTableReturnsTranslatedMetadata) {
  LocalDeltaTxnServer server;
  ::delta::txn::v1::TableMetadata* metadata = server.service().get_table_response.mutable_metadata();
  metadata->set_schema_string(R"({"type":"struct","fields":[]})");
  metadata->add_partition_columns("region");
  server.service().get_table_response.set_version(42);
  server.service().get_table_response.mutable_protocol()->set_min_reader_version(1);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  const DeltaTableInfo info = client.get_table("s3://bucket/table");

  EXPECT_EQ(info.version, 42);
  EXPECT_EQ(info.schema_string, R"({"type":"struct","fields":[]})");
  ASSERT_EQ(info.partition_columns.size(), 1u);
  EXPECT_EQ(info.partition_columns[0], "region");
}

TEST(DeltaTxnClient, GetTableThrowsOnNonOkStatus) {
  LocalDeltaTxnServer server;
  server.service().get_table_should_fail = true;

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(client.get_table("s3://bucket/missing")), StorageError);
}

TEST(DeltaTxnClient, GetTableSendsApiKeyHeader) {
  LocalDeltaTxnServer server;
  server.service().get_table_response.mutable_metadata();

  DeltaSection config = endpoint_config(server.endpoint());
  config.api_key = "secret";
  DeltaTxnClient client(config);
  (void)client.get_table("s3://bucket/table");

  EXPECT_EQ(server.service().last_seen_api_key, "secret");
}

// Proves the actual point of ClientSpan support, not just that it compiles:
// under a real OTel build, get_table() must send a well-formed W3C
// traceparent header (see kernellake::observability::ClientSpan::inject(),
// query_tracing_otel.cpp) -- what delta-txn-service's own TraceContextLayer
// (telemetry/trace_context.rs) extracts to parent its span under this
// call. Under the stub build (KERNELLAKE_ENABLE_OTEL=OFF, the default),
// inject() never calls its setter at all, so no header should be sent --
// asserting on that absence matters too: a stub silently sending a
// meaningless header would be its own (different) bug.
#ifdef KERNELLAKE_ENABLE_OTEL
// observability::init() (not the test-only init_for_testing(), which needs
// otel-internal in-memory-exporter headers this otherwise otel-agnostic
// test file has no other reason to include) enables tracing for exactly
// this test's scope via RAII -- deliberately not relying on some *other*
// test file (query_tracing_test.cpp) having already flipped the module's
// internal g_enabled flag via its own init_for_testing() call earlier in
// the same test binary run, which would make this test's outcome depend
// on gtest execution order rather than on anything this test itself sets
// up. init() builds real (but real-collector-optional -- gRPC channel
// construction is lazy, nothing here ever needs a reachable OTLP
// endpoint) OTLP exporters; nothing in this test asserts on export
// success, only on ClientSpan::inject() actually running instead of
// short-circuiting on a disabled tracer.
class ScopedObservability {
 public:
  ScopedObservability() {
    ObservabilitySection config;
    config.enabled = true;
    config.service_name = "delta-txn-client-test";
    observability::init(config);
  }
  ~ScopedObservability() { observability::shutdown(); }
};
#endif

TEST(DeltaTxnClient, GetTableTraceContextHeader) {
  LocalDeltaTxnServer server;
  server.service().get_table_response.mutable_metadata();

#ifdef KERNELLAKE_ENABLE_OTEL
  const ScopedObservability observability_guard;
#endif

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  (void)client.get_table("s3://bucket/table");

#ifdef KERNELLAKE_ENABLE_OTEL
  // W3C format: "{version:2}-{trace-id:32}-{parent-id:16}-{flags:2}", e.g.
  // "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".
  const std::string& traceparent = server.service().last_seen_traceparent;
  ASSERT_FALSE(traceparent.empty());
  const std::vector<std::string> parts = [&traceparent] {
    std::vector<std::string> result;
    std::string current;
    for (const char c : traceparent) {
      if (c == '-') {
        result.push_back(current);
        current.clear();
      } else {
        current += c;
      }
    }
    result.push_back(current);
    return result;
  }();
  ASSERT_EQ(parts.size(), 4u);
  EXPECT_EQ(parts[0], "00");
  EXPECT_EQ(parts[1].size(), 32u);
  EXPECT_EQ(parts[2].size(), 16u);
  EXPECT_NE(parts[1], "00000000000000000000000000000000");  // a real, non-zero trace id
  EXPECT_NE(parts[2], "0000000000000000");                  // a real, non-zero span id
#else
  EXPECT_TRUE(server.service().last_seen_traceparent.empty());
#endif
}

TEST(DeltaTxnClient, ListActiveFilesCollectsHeaderAndBatches) {
  LocalDeltaTxnServer server;

  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(7);
  header.mutable_header()->mutable_metadata()->set_schema_string("schema");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch1;
  ::delta::txn::v1::AddFile* file1 = batch1.mutable_batch()->add_files();
  file1->set_path("part-0.parquet");
  file1->set_size(1000);
  (*file1->mutable_partition_values())["region"] = "US";
  file1->mutable_stats()->set_num_records(5);
  server.service().list_active_files_responses.push_back(batch1);

  ::delta::txn::v1::ListActiveFilesResponse batch2;
  ::delta::txn::v1::AddFile* file2 = batch2.mutable_batch()->add_files();
  file2->set_path("part-1.parquet");
  file2->set_size(2000);
  server.service().list_active_files_responses.push_back(batch2);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  const DeltaActiveFileListing listing = client.list_active_files("s3://bucket/table");

  EXPECT_EQ(listing.table.version, 7);
  EXPECT_EQ(listing.table.schema_string, "schema");
  ASSERT_EQ(listing.files.size(), 2u);
  EXPECT_EQ(listing.files[0].path, "part-0.parquet");
  EXPECT_EQ(listing.files[0].size, 1000);
  EXPECT_EQ(listing.files[0].partition_values.at("region"), "US");
  EXPECT_EQ(listing.files[0].record_count, 5);
  EXPECT_EQ(listing.files[1].path, "part-1.parquet");
  EXPECT_EQ(listing.files[1].record_count, 0);
}

TEST(DeltaTxnClient, ListActiveFilesThrowsWhenStreamEndsWithoutHeader) {
  LocalDeltaTxnServer server;
  // No responses configured -- stream ends immediately, no header ever sent.

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(client.list_active_files("s3://bucket/table")), StorageError);
}

TEST(DeltaTxnClient, ListActiveFilesThrowsOnStreamFailure) {
  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata();
  server.service().list_active_files_responses.push_back(header);
  server.service().list_active_files_should_fail_stream = true;

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(client.list_active_files("s3://bucket/table")), StorageError);
}

TEST(DeltaTxnClient, ConstructorThrowsWhenEndpointIsEmpty) {
  EXPECT_THROW((void)(DeltaTxnClient(DeltaSection{})), ConfigurationError);
}

TEST(DeltaTxnClient, ThrowsOnConnectionFailure) {
  DeltaTxnClient client(endpoint_config("127.0.0.1:1"));  // nothing listens here
  EXPECT_THROW((void)(client.get_table("s3://bucket/table")), StorageError);
}

}  // namespace
}  // namespace kernellake::delta
