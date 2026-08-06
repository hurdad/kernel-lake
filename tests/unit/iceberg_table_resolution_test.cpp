#include "kernellake/iceberg/iceberg_table_resolution.hpp"

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

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake::iceberg {
namespace {

namespace fs = std::filesystem;

// A minimal single-connection-at-a-time loopback HTTP stub for the fake
// REST catalog -- see rest_catalog_client_test.cpp's identical helper for
// why this is a purpose-built stand-in rather than a real HTTP server or a
// mocking framework (none exists in this test tree); duplicated here
// rather than shared since this project has no shared test-utility target.
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

// Small avro-c writer helper -- same shape as manifest_reader_test.cpp's
// AvroFixtureWriter, duplicated for the same "no shared test-utility
// target" reason as LoopbackHttpServer above.
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

class IcebergTableResolutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_iceberg_table_resolution_test_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                    ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  // Matches the "id: long required, amount: double optional" schema used
  // by every LoadTableResult fixture below.
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
    const arrow::Status write_status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, /*chunk_size=*/count);
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();
  }

  fs::path write_manifest(const std::string& name, const std::vector<std::pair<std::string, int64_t>>& files,
                          int32_t status) {
    AvroFixtureWriter writer(kManifestSchemaJson);
    std::vector<std::function<void(avro_value_t&)>> rows;
    for (const auto& [path, count] : files) {
      rows.push_back([path, count, status](avro_value_t& v) {
        set_int_field(v, "status", status);
        avro_value_t data_file;
        avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
        set_string_field(data_file, "file_path", path);
        set_string_field(data_file, "file_format", "PARQUET");
        set_long_field(data_file, "record_count", count);
        set_long_field(data_file, "file_size_in_bytes", 1000);
      });
    }
    const fs::path manifest_path = dir_ / name;
    writer.write(manifest_path, rows);
    return manifest_path;
  }

  fs::path write_manifest_list(const std::string& name,
                               const std::vector<std::pair<std::string, int32_t>>& manifests) {
    AvroFixtureWriter writer(kManifestListSchemaJson);
    std::vector<std::function<void(avro_value_t&)>> rows;
    for (const auto& [path, content] : manifests) {
      rows.push_back([path, content](avro_value_t& v) {
        set_string_field(v, "manifest_path", path);
        set_long_field(v, "manifest_length", 1234);
        set_int_field(v, "partition_spec_id", 0);
        set_int_field(v, "content", content);
        set_long_field(v, "added_snapshot_id", 42);
      });
    }
    const fs::path list_path = dir_ / name;
    writer.write(list_path, rows);
    return list_path;
  }

  std::string load_table_result_json(const std::string& manifest_list_path) {
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
                       dir_.string(), manifest_list_path);
  }

  fs::path dir_;
  LocalObjectStore store_;
};

TEST_F(IcebergTableResolutionTest, ResolvesLiveDataFilesWithCurrentSchema) {
  write_data_file(dir_ / "data-0.parquet", 0, 5);
  write_data_file(dir_ / "data-1.parquet", 5, 3);
  const fs::path manifest = write_manifest(
      "m0.avro", {{(dir_ / "data-0.parquet").string(), 5}, {(dir_ / "data-1.parquet").string(), 3}},
      /*status=*/1);
  const fs::path manifest_list = write_manifest_list("snap-42.avro", {{manifest.string(), 0}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders");
  server.join();

  EXPECT_TRUE(resolved.partition_columns.empty());
  ASSERT_EQ(resolved.schema.field_count(), 2u);
  EXPECT_EQ(resolved.schema.field(0).name, "id");
  EXPECT_FALSE(resolved.schema.field(0).type.nullable);
  EXPECT_EQ(resolved.schema.field(1).name, "amount");
  EXPECT_TRUE(resolved.schema.field(1).type.nullable);

  ASSERT_EQ(resolved.files.size(), 2u);
  int64_t total_rows = 0;
  for (const ResolvedFile& file : resolved.files) {
    total_rows += file.metadata.row_count;
    EXPECT_TRUE(file.partition_values.empty());
  }
  EXPECT_EQ(total_rows, 8);
}

TEST_F(IcebergTableResolutionTest, SkipsDeletedStatusEntries) {
  write_data_file(dir_ / "data-0.parquet", 0, 5);
  write_data_file(dir_ / "data-1.parquet", 5, 3);
  const fs::path manifest = write_manifest(
      "m0.avro", {{(dir_ / "data-0.parquet").string(), 5}, {(dir_ / "data-1.parquet").string(), 3}},
      /*status=*/2 /* DELETED */);
  // Second manifest re-adds only data-0 as live.
  const fs::path manifest2 =
      write_manifest("m1.avro", {{(dir_ / "data-0.parquet").string(), 5}}, /*status=*/1);
  const fs::path manifest_list =
      write_manifest_list("snap-42.avro", {{manifest.string(), 0}, {manifest2.string(), 0}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders");
  server.join();

  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_EQ(resolved.files[0].metadata.row_count, 5);
}

TEST_F(IcebergTableResolutionTest, ThrowsOnLiveDeleteManifest) {
  write_data_file(dir_ / "data-0.parquet", 0, 5);
  const fs::path data_manifest = write_manifest("m0.avro", {{(dir_ / "data-0.parquet").string(), 5}}, 1);
  const fs::path delete_manifest = write_manifest("del-0.avro", {{(dir_ / "data-0.parquet").string(), 5}}, 1);
  const fs::path manifest_list =
      write_manifest_list("snap-42.avro", {{data_manifest.string(), 0}, {delete_manifest.string(), 1}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders")), StorageError);
  server.join();
}

TEST_F(IcebergTableResolutionTest, ThrowsOnNonParquetDataFile) {
  const fs::path manifest = write_manifest("m0.avro", {}, 1);
  // Hand-write one ORC entry directly (write_manifest() always fills
  // "PARQUET"), reusing the same writer machinery.
  AvroFixtureWriter writer(kManifestSchemaJson);
  writer.write(manifest, {[](avro_value_t& v) {
                 set_int_field(v, "status", 1);
                 avro_value_t data_file;
                 avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
                 set_string_field(data_file, "file_path", "s3://warehouse/db/orders/data/part-0.orc");
                 set_string_field(data_file, "file_format", "ORC");
                 set_long_field(data_file, "record_count", 5);
                 set_long_field(data_file, "file_size_in_bytes", 1000);
               }});
  const fs::path manifest_list = write_manifest_list("snap-42.avro", {{manifest.string(), 0}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders")), StorageError);
  server.join();
}

TEST_F(IcebergTableResolutionTest, ThrowsWhenDataFileSchemaDoesNotMatchTableSchema) {
  // Only one column, unlike the declared two-column (id, amount) schema.
  arrow::Int64Builder id_builder;
  ASSERT_TRUE(id_builder.Append(1).ok());
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  const auto mismatched_schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(mismatched_schema, {id_array});
  const fs::path data_path = dir_ / "data-0.parquet";
  auto sink_result = arrow::io::FileOutputStream::Open(data_path.string());
  ASSERT_TRUE(sink_result.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, 1).ok());

  const fs::path manifest = write_manifest("m0.avro", {{data_path.string(), 1}}, 1);
  const fs::path manifest_list = write_manifest_list("snap-42.avro", {{manifest.string(), 0}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders")), StorageError);
  server.join();
}

TEST_F(IcebergTableResolutionTest, TableWithNoCurrentSnapshotResolvesToZeroFiles) {
  const std::string json = R"json({
    "metadata": {
      "format-version": 2,
      "location": "s3://warehouse/db/orders",
      "current-schema-id": 0,
      "schemas": [{"schema-id": 0, "fields": [
        {"id": 1, "name": "id", "required": true, "type": "long"}
      ]}]
    }
  })json";

  LoopbackHttpServer server;
  server.respond({http_ok_json(json)});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders");
  server.join();

  EXPECT_TRUE(resolved.files.empty());
  ASSERT_EQ(resolved.schema.field_count(), 1u);
}

}  // namespace
}  // namespace kernellake::iceberg
