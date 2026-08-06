#include <arpa/inet.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <avro.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <parquet/arrow/writer.h>
#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/planner/logical_plan.hpp"

// This is the capstone test for the read_iceberg(...) SQL surface: a real
// QueryEngine, constructed the same way any caller (CLI, Flight SQL
// server) would, planning and physically resolving a genuine
// `SELECT ... FROM read_iceberg('catalog.namespace.table')` query end to
// end -- SQL text -> parser preprocessing -> binder -> logical plan ->
// physical plan -- against a fake HTTP REST catalog and real Avro
// manifest/Parquet data fixtures. Everything below this level (the REST
// client, manifest reader, schema translation, resolve_iceberg_table()) is
// already covered by its own dedicated test file; this one exists purely
// to prove the wiring at the SQL/QueryEngine layer is correct.
namespace kernellake {
namespace {

namespace fs = std::filesystem;

// Same purpose-built loopback HTTP stub as rest_catalog_client_test.cpp/
// iceberg_table_resolution_test.cpp -- duplicated for the same "no shared
// test-utility target in this project" reason.
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
    if (thread_.joinable()) thread_.join();
    ::close(listen_fd_);
  }

  [[nodiscard]] std::string base_url() const { return fmt::format("http://127.0.0.1:{}", port_); }

  // Serves `responses.size()` connections in order -- this test issues two
  // requests per query (explain_logical() at bind time, explain() at
  // physical-plan time both re-resolve independently, see
  // physical_planner.cpp's own comment on why), so callers pass the same
  // response twice.
  void respond(std::vector<std::string> responses) {
    thread_ = std::thread([this, responses = std::move(responses)] {
      for (const std::string& response : responses) {
        const int conn_fd = ::accept(listen_fd_, nullptr, nullptr);
        if (conn_fd < 0) return;
        timeval timeout{5, 0};
        ::setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        std::string request(65536, '\0');
        const ssize_t received = ::recv(conn_fd, request.data(), request.size(), 0);
        (void)received;
        ::send(conn_fd, response.data(), response.size(), 0);
        ::close(conn_fd);
      }
    });
  }

  void join() {
    if (thread_.joinable()) thread_.join();
  }

 private:
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread thread_;
};

std::string http_ok_json(const std::string& json_body) {
  return fmt::format(
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
      json_body.size(), json_body);
}

constexpr const char* kManifestListSchemaJson = R"({
  "type": "record",
  "name": "manifest_file",
  "fields": [
    {"name": "manifest_path", "type": "string"},
    {"name": "manifest_length", "type": "long"},
    {"name": "partition_spec_id", "type": "int"},
    {"name": "content", "type": "int"},
    {"name": "added_snapshot_id", "type": "long"}
  ]
})";

constexpr const char* kManifestSchemaJson = R"({
  "type": "record",
  "name": "manifest_entry",
  "fields": [
    {"name": "status", "type": "int"},
    {"name": "data_file", "type": {
      "type": "record",
      "name": "r2",
      "fields": [
        {"name": "file_path", "type": "string"},
        {"name": "file_format", "type": "string"},
        {"name": "record_count", "type": "long"},
        {"name": "file_size_in_bytes", "type": "long"},
        {"name": "partition", "type": {"type": "record", "name": "r102", "fields": []}}
      ]
    }}
  ]
})";

class AvroFixtureWriter {
 public:
  explicit AvroFixtureWriter(const std::string& schema_json) {
    avro_schema_from_json_length(schema_json.c_str(), schema_json.size(), &schema_);
    iface_ = avro_generic_class_from_schema(schema_);
  }
  ~AvroFixtureWriter() {
    if (iface_ != nullptr) avro_value_iface_decref(iface_);
    if (schema_ != nullptr) avro_schema_decref(schema_);
  }

  void write(const fs::path& path, const std::vector<std::function<void(avro_value_t&)>>& rows) {
    avro_file_writer_t writer = nullptr;
    avro_file_writer_create(path.c_str(), schema_, &writer);
    avro_value_t value;
    avro_generic_value_new(iface_, &value);
    for (const auto& fill_row : rows) {
      fill_row(value);
      avro_file_writer_append_value(writer, &value);
      avro_value_reset(&value);
    }
    avro_value_decref(&value);
    avro_file_writer_close(writer);
  }

 private:
  avro_schema_t schema_ = nullptr;
  avro_value_iface_t* iface_ = nullptr;
};

void set_string_field(avro_value_t& record, const char* name, const std::string& s) {
  avro_value_t field;
  avro_value_get_by_name(&record, name, &field, nullptr);
  avro_value_set_string(&field, s.c_str());
}
void set_long_field(avro_value_t& record, const char* name, int64_t v) {
  avro_value_t field;
  avro_value_get_by_name(&record, name, &field, nullptr);
  avro_value_set_long(&field, v);
}
void set_int_field(avro_value_t& record, const char* name, int32_t v) {
  avro_value_t field;
  avro_value_get_by_name(&record, name, &field, nullptr);
  avro_value_set_int(&field, v);
}

class QueryEngineIcebergTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_engine_iceberg_test_" +
                     std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                     ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  // "id: long required, amount: double optional", matching every
  // LoadTableResult fixture below.
  void write_data_file(const fs::path& path, int64_t id_start, int64_t count) {
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
    auto sink_result = arrow::io::FileOutputStream::Open(path.string());
    ASSERT_TRUE(sink_result.ok()) << sink_result.status().ToString();
    ASSERT_TRUE(
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, /*chunk_size=*/count)
            .ok());
  }

  std::string build_table_and_get_manifest_list_json() {
    write_data_file(dir_ / "data-0.parquet", 0, 5);
    write_data_file(dir_ / "data-1.parquet", 5, 5);

    AvroFixtureWriter manifest_writer(kManifestSchemaJson);
    const fs::path manifest_path = dir_ / "m0.avro";
    manifest_writer.write(
        manifest_path, {[this](avro_value_t& v) {
                          set_int_field(v, "status", 1);
                          avro_value_t data_file;
                          avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
                          set_string_field(data_file, "file_path", (dir_ / "data-0.parquet").string());
                          set_string_field(data_file, "file_format", "PARQUET");
                          set_long_field(data_file, "record_count", 5);
                          set_long_field(data_file, "file_size_in_bytes", 1000);
                        },
                        [this](avro_value_t& v) {
                          set_int_field(v, "status", 1);
                          avro_value_t data_file;
                          avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
                          set_string_field(data_file, "file_path", (dir_ / "data-1.parquet").string());
                          set_string_field(data_file, "file_format", "PARQUET");
                          set_long_field(data_file, "record_count", 5);
                          set_long_field(data_file, "file_size_in_bytes", 1000);
                        }});

    AvroFixtureWriter list_writer(kManifestListSchemaJson);
    const fs::path list_path = dir_ / "snap-42.avro";
    list_writer.write(list_path, {[&manifest_path](avro_value_t& v) {
                        set_string_field(v, "manifest_path", manifest_path.string());
                        set_long_field(v, "manifest_length", 1234);
                        set_int_field(v, "partition_spec_id", 0);
                        set_int_field(v, "content", 0);
                        set_long_field(v, "added_snapshot_id", 42);
                      }});

    return fmt::format(R"json({{
      "metadata": {{
        "format-version": 2,
        "location": "{0}",
        "current-schema-id": 0,
        "schemas": [{{"schema-id": 0, "fields": [
          {{"id": 1, "name": "id", "required": true, "type": "long"}},
          {{"id": 2, "name": "amount", "required": false, "type": "double"}}
        ]}}],
        "current-snapshot-id": 42,
        "snapshots": [{{"snapshot-id": 42, "manifest-list": "{1}"}}]
      }}
    }})json",
        dir_.string(), list_path.string());
  }

  EngineConfig config_with_catalog(const std::string& catalog_uri) {
    EngineConfig config = default_config();
    IcebergCatalogSection catalog;
    catalog.catalog_uri = catalog_uri;
    config.iceberg.catalogs["prod"] = catalog;
    return config;
  }

  fs::path dir_;
};

TEST_F(QueryEngineIcebergTest, ExplainLogicalBindsAgainstIcebergSchema) {
  const std::string load_table_result = http_ok_json(build_table_and_get_manifest_list_json());
  LoopbackHttpServer server;
  server.respond({load_table_result});  // explain_logical() alone makes one request.

  QueryEngine engine(config_with_catalog(server.base_url()));
  const LogicalPlanPtr plan =
      engine.explain_logical("SELECT id, amount FROM read_iceberg('prod.db.orders') WHERE id < 100");
  server.join();

  ASSERT_EQ(plan->output_schema().field_count(), 2u);
  EXPECT_EQ(plan->output_schema().field(0).name, "id");
  EXPECT_EQ(plan->output_schema().field(1).name, "amount");
}

TEST_F(QueryEngineIcebergTest, ExplainProducesFullPhysicalPlanOverRealDataFiles) {
  const std::string load_table_result = http_ok_json(build_table_and_get_manifest_list_json());
  LoopbackHttpServer server;
  // explain() re-resolves independently at physical-planning time (see
  // physical_planner.cpp's own comment) -- two requests total.
  server.respond({load_table_result, load_table_result});

  QueryEngine engine(config_with_catalog(server.base_url()));
  const PhysicalPlanPtr plan = engine.explain("SELECT id FROM read_iceberg('prod.db.orders')");
  server.join();

  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("ArrowResult"), std::string::npos);
  EXPECT_NE(text.find("ParquetScan"), std::string::npos);
}

TEST_F(QueryEngineIcebergTest, ExplainLogicalThrowsOnUnknownColumn) {
  const std::string load_table_result = http_ok_json(build_table_and_get_manifest_list_json());
  LoopbackHttpServer server;
  server.respond({load_table_result});

  QueryEngine engine(config_with_catalog(server.base_url()));
  EXPECT_THROW(
      (void)(engine.explain_logical("SELECT nonexistent FROM read_iceberg('prod.db.orders')")), BindingError);
  server.join();
}

TEST_F(QueryEngineIcebergTest, ExplainLogicalThrowsOnUnknownCatalogName) {
  QueryEngine engine(default_config());  // no "prod" catalog configured
  EXPECT_THROW((void)(engine.explain_logical("SELECT id FROM read_iceberg('prod.db.orders')")),
               ConfigurationError);
}

}  // namespace
}  // namespace kernellake
