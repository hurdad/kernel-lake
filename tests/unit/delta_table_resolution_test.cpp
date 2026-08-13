#include "kernellake/delta/delta_table_resolution.hpp"

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
#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

// Same purpose-built in-process gRPC server as delta_txn_client_test.cpp --
// duplicated for the same "no shared test-utility target in this project"
// reason iceberg_table_resolution_test.cpp's own LoopbackHttpServer
// duplicates rest_catalog_client_test.cpp's.
namespace kernellake::delta {
namespace {

namespace fs = std::filesystem;

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

DeltaSection endpoint_config(const std::string& endpoint) {
  DeltaSection config;
  config.grpc_endpoint = endpoint;
  return config;
}

class DeltaTableResolutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_delta_table_resolution_test_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                    ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  // Two physical columns (id: long not-null, amount: double nullable) --
  // matching every fixture schema_string below, which also always adds a
  // "region" partition column.
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

  static constexpr const char* kSchemaWithRegionPartitionJson = R"json({"type":"struct","fields":[
    {"name":"id","type":"long","nullable":false,"metadata":{}},
    {"name":"amount","type":"double","nullable":true,"metadata":{}},
    {"name":"region","type":"string","nullable":true,"metadata":{}}
  ]})json";

  fs::path dir_;
  LocalObjectStore store_;
};

TEST_F(DeltaTableResolutionTest, ResolvesActiveFilesAndSplitsOutPartitionColumns) {
  write_data_file("data-0.parquet", 0, 5);
  write_data_file("data-1.parquet", 5, 3);

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(kSchemaWithRegionPartitionJson);
  header.mutable_header()->mutable_metadata()->add_partition_columns("region");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");
  (*file0->mutable_partition_values())["region"] = "US";
  ::delta::txn::v1::AddFile* file1 = batch.mutable_batch()->add_files();
  file1->set_path("data-1.parquet");
  (*file1->mutable_partition_values())["region"] = "EU";
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  const ResolvedTable resolved = resolve_delta_table(store_, client, dir_.string());

  ASSERT_EQ(resolved.schema.field_count(), 3u);
  EXPECT_EQ(resolved.schema.field(0).name, "id");
  EXPECT_EQ(resolved.schema.field(1).name, "amount");
  EXPECT_EQ(resolved.schema.field(2).name, "region");  // partition column appended last

  ASSERT_EQ(resolved.partition_columns.size(), 1u);
  EXPECT_EQ(resolved.partition_columns[0].name, "region");
  EXPECT_EQ(resolved.partition_columns[0].type.id, TypeId::String);

  ASSERT_EQ(resolved.files.size(), 2u);
  // Physical footer schema (via inspect_parquet_file()) must never include
  // the partition column -- proves it was split out, not just appended.
  EXPECT_EQ(resolved.files[0].metadata.schema.field_count(), 2u);
  ASSERT_EQ(resolved.files[0].partition_values.size(), 1u);
  EXPECT_EQ(std::get<std::string>(resolved.files[0].partition_values[0]), "US");
  ASSERT_EQ(resolved.files[1].partition_values.size(), 1u);
  EXPECT_EQ(std::get<std::string>(resolved.files[1].partition_values[0]), "EU");

  int64_t total_rows = 0;
  for (const ResolvedFile& file : resolved.files) {
    total_rows += file.metadata.row_count;
  }
  EXPECT_EQ(total_rows, 8);
}

TEST_F(DeltaTableResolutionTest, NoActiveFilesResolvesToZeroFilesWithFullSchema) {
  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(kSchemaWithRegionPartitionJson);
  header.mutable_header()->mutable_metadata()->add_partition_columns("region");
  server.service().list_active_files_responses.push_back(header);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  const ResolvedTable resolved = resolve_delta_table(store_, client, dir_.string());

  EXPECT_TRUE(resolved.files.empty());
  ASSERT_EQ(resolved.schema.field_count(), 3u);
}

TEST_F(DeltaTableResolutionTest, ThrowsWhenPartitionColumnIsMissingFromSchema) {
  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(
      R"json({"type":"struct","fields":[{"name":"id","type":"long","nullable":false,"metadata":{}}]})json");
  header.mutable_header()->mutable_metadata()->add_partition_columns("region");  // not in schema above
  server.service().list_active_files_responses.push_back(header);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

TEST_F(DeltaTableResolutionTest, ThrowsWhenFileIsMissingAPartitionValue) {
  write_data_file("data-0.parquet", 0, 5);

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(kSchemaWithRegionPartitionJson);
  header.mutable_header()->mutable_metadata()->add_partition_columns("region");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");  // no partition_values["region"] set
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

// Regression test for the real bug this file's parse_delta_partition_value()
// used to have: std::stoll()/std::stod() on a malformed numeric partition
// value threw a raw, uncaught std::invalid_argument instead of this
// project's own StorageError -- any caller expecting to catch StorageError
// (every other error path in this file, and every other resolve_*_table()
// across the codebase) would see it escape uncaught instead.
TEST_F(DeltaTableResolutionTest, ThrowsStorageErrorOnMalformedNumericPartitionValue) {
  write_data_file("data-0.parquet", 0, 5);

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(
      R"json({"type":"struct","fields":[
    {"name":"id","type":"long","nullable":false,"metadata":{}},
    {"name":"amount","type":"double","nullable":true,"metadata":{}},
    {"name":"day","type":"integer","nullable":true,"metadata":{}}
  ]})json");
  header.mutable_header()->mutable_metadata()->add_partition_columns("day");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");
  (*file0->mutable_partition_values())["day"] = "abc";  // not a valid integer
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

TEST_F(DeltaTableResolutionTest, ThrowsStorageErrorOnMalformedFloatingPointPartitionValue) {
  write_data_file("data-0.parquet", 0, 5);

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(
      R"json({"type":"struct","fields":[
    {"name":"id","type":"long","nullable":false,"metadata":{}},
    {"name":"amount","type":"double","nullable":true,"metadata":{}},
    {"name":"weight","type":"double","nullable":true,"metadata":{}}
  ]})json");
  header.mutable_header()->mutable_metadata()->add_partition_columns("weight");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");
  (*file0->mutable_partition_values())["weight"] = "not-a-number";
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

TEST_F(DeltaTableResolutionTest, ThrowsWhenBooleanPartitionValueIsNeitherTrueNorFalse) {
  write_data_file("data-0.parquet", 0, 5);

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(
      R"json({"type":"struct","fields":[
    {"name":"id","type":"long","nullable":false,"metadata":{}},
    {"name":"amount","type":"double","nullable":true,"metadata":{}},
    {"name":"active","type":"boolean","nullable":true,"metadata":{}}
  ]})json");
  header.mutable_header()->mutable_metadata()->add_partition_columns("active");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");
  (*file0->mutable_partition_values())["active"] = "1";  // neither "true" nor "false"
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

TEST_F(DeltaTableResolutionTest, ThrowsOnTimestampTypedPartitionColumn) {
  write_data_file("data-0.parquet", 0, 5);

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(
      R"json({"type":"struct","fields":[
    {"name":"id","type":"long","nullable":false,"metadata":{}},
    {"name":"amount","type":"double","nullable":true,"metadata":{}},
    {"name":"created_at","type":"timestamp","nullable":true,"metadata":{}}
  ]})json");
  header.mutable_header()->mutable_metadata()->add_partition_columns("created_at");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");
  (*file0->mutable_partition_values())["created_at"] = "2026-01-01 00:00:00";
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

TEST_F(DeltaTableResolutionTest, ThrowsWhenFileSchemaHasSameColumnCountButDifferentType) {
  // Same column count (2) as the declared table schema (id: long, amount:
  // double) but "amount" is written as float32, not float64 -- must still
  // be rejected as an evolved/incompatible schema, not silently accepted
  // just because the column counts line up.
  arrow::Int64Builder id_builder;
  arrow::FloatBuilder amount_builder;
  ASSERT_TRUE(id_builder.Append(1).ok());
  ASSERT_TRUE(amount_builder.Append(1.5F).ok());
  std::shared_ptr<arrow::Array> id_array;
  std::shared_ptr<arrow::Array> amount_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
  const auto mismatched_schema = arrow::schema(
      {arrow::field("id", arrow::int64(), false), arrow::field("amount", arrow::float32(), true)});
  const auto table = arrow::Table::Make(mismatched_schema, {id_array, amount_array});
  auto sink_result = arrow::io::FileOutputStream::Open((dir_ / "data-0.parquet").string());
  ASSERT_TRUE(sink_result.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, 1).ok());

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(kSchemaWithRegionPartitionJson);
  header.mutable_header()->mutable_metadata()->add_partition_columns("region");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");
  (*file0->mutable_partition_values())["region"] = "US";
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

// AddFile.path is normally table-root-relative, joined onto table_uri --
// but the Delta spec also allows a writer to record an absolute URI
// directly (detected by the presence of "://", same as kernellake::Uri::
// scheme() itself), which must be used as-is rather than joined. There's no
// real "s3://..." object for LocalObjectStore to actually open in this test
// environment, so what's asserted is the *path* the resulting failure names
// -- it must be the absolute URI unmodified, never table_uri (dir_) joined
// onto it.
TEST_F(DeltaTableResolutionTest, UsesAbsoluteAddFilePathAsIsWithoutJoiningTableUri) {
  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(kSchemaWithRegionPartitionJson);
  header.mutable_header()->mutable_metadata()->add_partition_columns("region");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  const std::string absolute_path = "s3://some-bucket/table/data-0.parquet";
  file0->set_path(absolute_path);
  (*file0->mutable_partition_values())["region"] = "US";
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  try {
    (void)resolve_delta_table(store_, client, dir_.string());
    FAIL() << "expected resolve_delta_table() to throw trying to open a nonexistent absolute path";
  } catch (const StorageError& e) {
    EXPECT_NE(std::string(e.what()).find(absolute_path), std::string::npos) << e.what();
    EXPECT_EQ(std::string(e.what()).find(dir_.string() + "/" + absolute_path), std::string::npos) << e.what();
  }
}

TEST_F(DeltaTableResolutionTest, ThrowsWhenFilePhysicalSchemaDoesNotMatchTableSchema) {
  // Only one column, unlike the declared two-physical-column (id, amount)
  // schema above.
  arrow::Int64Builder id_builder;
  ASSERT_TRUE(id_builder.Append(1).ok());
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  const auto mismatched_schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(mismatched_schema, {id_array});
  auto sink_result = arrow::io::FileOutputStream::Open((dir_ / "data-0.parquet").string());
  ASSERT_TRUE(sink_result.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, 1).ok());

  LocalDeltaTxnServer server;
  ::delta::txn::v1::ListActiveFilesResponse header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->mutable_metadata()->set_schema_string(kSchemaWithRegionPartitionJson);
  header.mutable_header()->mutable_metadata()->add_partition_columns("region");
  server.service().list_active_files_responses.push_back(header);

  ::delta::txn::v1::ListActiveFilesResponse batch;
  ::delta::txn::v1::AddFile* file0 = batch.mutable_batch()->add_files();
  file0->set_path("data-0.parquet");
  (*file0->mutable_partition_values())["region"] = "US";
  server.service().list_active_files_responses.push_back(batch);

  DeltaTxnClient client(endpoint_config(server.endpoint()));
  EXPECT_THROW((void)(resolve_delta_table(store_, client, dir_.string())), StorageError);
}

}  // namespace
}  // namespace kernellake::delta
