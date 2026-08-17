#include <arrow/api.h>
#include <arrow/io/file.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "delta_txn.grpc.pb.h"
#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/planner/logical_plan.hpp"

// This is the capstone test for the read_delta(...) SQL surface: a real
// QueryEngine, constructed the same way any caller (CLI, Flight SQL
// server) would, planning, physically resolving, and (unlike
// query_engine_iceberg_test.cpp -- see its own header comment) actually
// *executing* a genuine `SELECT ... FROM read_delta('<table_uri>')` query
// end to end -- SQL text -> parser preprocessing -> binder -> logical plan
// -> physical plan -> real Acero CPU execution -- against a fake in-process
// delta-txn-service and real Parquet data fixtures. The execute() case
// specifically proves a real bug this same session found and fixed:
// QueryEngine::execute() (query_engine_execute_stub.cpp/
// query_engine_execute_gpu.cpp) used to call build_physical_plan() with no
// resolver at all, so `read_iceberg(...)`/`read_delta(...)` could plan and
// explain but not actually execute -- see those files' own comments.
// Everything below this level (DeltaTxnClient, schema translation,
// resolve_delta_table()) is already covered by its own dedicated test
// file; this one exists purely to prove the wiring at the SQL/QueryEngine
// layer is correct.
namespace kernellake {
namespace {

namespace fs = std::filesystem;

// Same purpose-built in-process gRPC server as delta_txn_client_test.cpp/
// delta_table_resolution_test.cpp -- duplicated for the same "no shared
// test-utility target in this project" reason.
class FakeDeltaTxnService final : public ::delta::txn::v1::DeltaTxnService::Service {
 public:
  std::vector<::delta::txn::v1::ListActiveFilesResponse> list_active_files_responses;

  grpc::Status GetTable(grpc::ServerContext* /*context*/,
                        const ::delta::txn::v1::GetTableRequest* /*request*/,
                        ::delta::txn::v1::GetTableResponse* /*response*/) override {
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "not used by this test");
  }

  grpc::Status ListActiveFiles(
      grpc::ServerContext* /*context*/, const ::delta::txn::v1::ListActiveFilesRequest* /*request*/,
      grpc::ServerWriter<::delta::txn::v1::ListActiveFilesResponse>* writer) override {
    // Unlike query_engine_iceberg_test.cpp's LoopbackHttpServer (which
    // consumes one queued response per connection), this fake simply
    // replays the same configured responses on every call -- explain()
    // and execute() each re-resolve independently (see
    // QueryEngine::explain()'s own comment), issuing multiple
    // ListActiveFiles calls against this same fake, and a real
    // delta-txn-service would answer every one of them identically too.
    for (const ::delta::txn::v1::ListActiveFilesResponse& response : list_active_files_responses) {
      writer->Write(response);
    }
    return grpc::Status::OK;
  }
};

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

class QueryEngineDeltaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_engine_delta_test_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                    ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  // "id: long not-null, amount: double nullable" -- matches
  // kSchemaJson below.
  void write_data_file(const std::string& file_name, int64_t id_start, int64_t count) {
    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    for (int64_t i = 0; i < count; ++i) {
      ASSERT_TRUE(id_builder.Append(id_start + i).ok());
      ASSERT_TRUE(amount_builder.Append(static_cast<double>(id_start + i)).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    std::shared_ptr<arrow::Array> amount_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    const auto schema = arrow::schema(
        {arrow::field("id", arrow::int64(), false), arrow::field("amount", arrow::float64(), true)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array});
    auto sink_result = arrow::io::FileOutputStream::Open((dir_ / file_name).string());
    ASSERT_TRUE(sink_result.ok()) << sink_result.status().ToString();
    const arrow::Status write_status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, /*chunk_size=*/count);
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();
  }

  static constexpr const char* kSchemaJson = R"json({"type":"struct","fields":[
    {"name":"id","type":"long","nullable":false,"metadata":{}},
    {"name":"amount","type":"double","nullable":true,"metadata":{}}
  ]})json";

  void configure_active_files(FakeDeltaTxnService& service, int64_t version,
                              const std::vector<std::string>& file_names) {
    ::delta::txn::v1::ListActiveFilesResponse header;
    header.mutable_header()->set_version(version);
    header.mutable_header()->mutable_metadata()->set_schema_string(kSchemaJson);
    service.list_active_files_responses.push_back(header);

    if (!file_names.empty()) {
      ::delta::txn::v1::ListActiveFilesResponse batch;
      for (const std::string& file_name : file_names) {
        batch.mutable_batch()->add_files()->set_path(file_name);
      }
      service.list_active_files_responses.push_back(batch);
    }
  }

  EngineConfig config_with_delta_endpoint(const std::string& endpoint) {
    EngineConfig config = default_config();
    config.delta.grpc_endpoint = endpoint;
    return config;
  }

  fs::path dir_;
};

TEST_F(QueryEngineDeltaTest, ExplainLogicalBindsAgainstDeltaSchema) {
  LocalDeltaTxnServer server;
  configure_active_files(server.service(), /*version=*/1, {});

  QueryEngine engine(config_with_delta_endpoint(server.endpoint()));
  const LogicalPlanPtr plan =
      engine.explain_logical("SELECT id, amount FROM read_delta('" + dir_.string() + "') WHERE id < 100");

  ASSERT_EQ(plan->output_schema().field_count(), 2u);
  EXPECT_EQ(plan->output_schema().field(0).name, "id");
  EXPECT_EQ(plan->output_schema().field(1).name, "amount");
}

TEST_F(QueryEngineDeltaTest, ExplainProducesFullPhysicalPlanOverRealDataFiles) {
  write_data_file("data-0.parquet", 0, 5);
  write_data_file("data-1.parquet", 5, 5);
  LocalDeltaTxnServer server;
  // Named local, not an inline brace-init temporary passed directly as the
  // argument -- GCC 15's -Wfree-nonheap-object otherwise misfires here
  // (confirmed false positive: real double-free would need a mismatched
  // delete, not a std::vector<std::string> destroying normally). The
  // "offset 32" in that warning matches sizeof(std::string) on libstdc++
  // exactly -- GCC's escape analysis loses track of which object's buffer
  // is being freed when the vector-of-strings destructor chain gets
  // heavily inlined right at this temporary's destruction point. A named
  // local sidesteps the specific inlining shape that trips it, with no
  // behavior change.
  const std::vector<std::string> active_files = {"data-0.parquet", "data-1.parquet"};
  configure_active_files(server.service(), /*version=*/1, active_files);

  QueryEngine engine(config_with_delta_endpoint(server.endpoint()));
  const PhysicalPlanPtr plan = engine.explain("SELECT id FROM read_delta('" + dir_.string() + "')");

  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("ArrowResult"), std::string::npos);
  EXPECT_NE(text.find("ParquetScan"), std::string::npos);
}

// Proves the QueryEngine::execute() resolver fix (see this file's own
// header comment): without it, this would throw at physical-plan time
// instead of returning real rows.
TEST_F(QueryEngineDeltaTest, ExecuteRunsQueryEndToEndOverRealDataFiles) {
  write_data_file("data-0.parquet", 0, 5);
  write_data_file("data-1.parquet", 5, 3);
  LocalDeltaTxnServer server;
  // Named local -- see the identical comment on this same pattern in
  // ExplainProducesFullPhysicalPlanOverRealDataFiles above.
  const std::vector<std::string> active_files = {"data-0.parquet", "data-1.parquet"};
  configure_active_files(server.service(), /*version=*/1, active_files);

  EngineConfig config = config_with_delta_endpoint(server.endpoint());
  config.engine.backend = "cpu";
  QueryEngine engine(config);

  const QueryResult result =
      engine.execute("SELECT SUM(amount) AS total FROM read_delta('" + dir_.string() + "')");

  ASSERT_EQ(result.rows_returned, 1);
  ASSERT_EQ(result.batches.size(), 1u);
  const auto* total_array = static_cast<arrow::DoubleArray*>(result.batches[0]->column(0).get());
  // sum(0..7) = 28.
  EXPECT_DOUBLE_EQ(total_array->Value(0), 28.0);
}

TEST_F(QueryEngineDeltaTest, ExplainLogicalThrowsOnUnknownColumn) {
  LocalDeltaTxnServer server;
  configure_active_files(server.service(), /*version=*/1, {});

  QueryEngine engine(config_with_delta_endpoint(server.endpoint()));
  EXPECT_THROW((void)(engine.explain_logical("SELECT nonexistent FROM read_delta('" + dir_.string() + "')")),
               BindingError);
}

TEST_F(QueryEngineDeltaTest, ExplainLogicalThrowsWhenGrpcEndpointIsNotConfigured) {
  QueryEngine engine(default_config());  // delta.grpc_endpoint left unset
  EXPECT_THROW((void)(engine.explain_logical("SELECT id FROM read_delta('" + dir_.string() + "')")),
               ConfigurationError);
}

}  // namespace
}  // namespace kernellake
