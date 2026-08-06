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

// Same shape as kManifestSchemaJson, but with one populated partition
// field -- for the partition-pruning test below, which needs a real
// (non-empty) manifest "partition" struct to decode a value from.
constexpr const char* kManifestSchemaWithRegionPartitionJson = R"({
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
        {"name": "partition", "type": {"type": "record", "name": "r102", "fields": [
          {"name": "region", "type": "string"}
        ]}}
      ]
    }}
  ]
})";

// Same shape as kManifestSchemaJson, but with data_file.content present --
// for the row-level-delete tests below, which need to write POSITION/
// EQUALITY_DELETES entries (data_file.content == 1/2) into a delete
// manifest.
constexpr const char* kManifestSchemaWithContentJson = R"({
  "type": "record",
  "name": "manifest_entry",
  "fields": [
    {"name": "status", "type": "int"},
    {"name": "data_file", "type": {
      "type": "record",
      "name": "r2",
      "fields": [
        {"name": "content", "type": "int"},
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

  // "id: long, amount: double, region: string" -- for the
  // partition-pruning test below, where an identity-transform partition
  // column is (correctly, per Iceberg semantics -- see
  // iceberg_table_resolution.hpp's own comment) physically present in the
  // data file itself, unlike Delta's partition columns.
  void write_data_file_with_region(const fs::path& path, int64_t id_start, int64_t count,
                                   const std::string& region) {
    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::StringBuilder region_builder;
    for (int64_t i = 0; i < count; ++i) {
      ASSERT_TRUE(id_builder.Append(id_start + i).ok());
      ASSERT_TRUE(amount_builder.Append(static_cast<double>(id_start + i)).ok());
      ASSERT_TRUE(region_builder.Append(region).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    std::shared_ptr<arrow::Array> amount_array;
    std::shared_ptr<arrow::Array> region_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false),
                                       arrow::field("amount", arrow::float64(), true),
                                       arrow::field("region", arrow::utf8(), true)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array, region_array});
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

  // One data file per (path, record_count, region) triple, `region` set
  // as the manifest entry's single partition value (see
  // kManifestSchemaWithRegionPartitionJson above) -- for the
  // partition-pruning test below.
  fs::path write_manifest_with_region_partition(
      const std::string& name, const std::vector<std::tuple<std::string, int64_t, std::string>>& files) {
    AvroFixtureWriter writer(kManifestSchemaWithRegionPartitionJson);
    std::vector<std::function<void(avro_value_t&)>> rows;
    for (const auto& [path, count, region] : files) {
      rows.push_back([path, count, region](avro_value_t& v) {
        set_int_field(v, "status", /*ADDED=*/1);
        avro_value_t data_file;
        avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
        set_string_field(data_file, "file_path", path);
        set_string_field(data_file, "file_format", "PARQUET");
        set_long_field(data_file, "record_count", count);
        set_long_field(data_file, "file_size_in_bytes", 1000);
        avro_value_t partition;
        avro_value_get_by_name(&data_file, "partition", &partition, nullptr);
        set_string_field(partition, "region", region);
      });
    }
    const fs::path manifest_path = dir_ / name;
    writer.write(manifest_path, rows);
    return manifest_path;
  }

  // A real "file_path: string, pos: long" Parquet file -- the Iceberg
  // spec's Position Delete Files schema -- for the row-level-delete tests
  // below. Same shape as position_delete_reader_test.cpp's own helper,
  // duplicated per this project's usual test-file convention.
  fs::path write_position_delete_file(const std::string& name,
                                      const std::vector<std::pair<std::string, int64_t>>& deleted_positions) {
    arrow::StringBuilder file_path_builder;
    arrow::Int64Builder pos_builder;
    for (const auto& [path, pos] : deleted_positions) {
      EXPECT_TRUE(file_path_builder.Append(path).ok());
      EXPECT_TRUE(pos_builder.Append(pos).ok());
    }
    std::shared_ptr<arrow::Array> file_path_array;
    std::shared_ptr<arrow::Array> pos_array;
    EXPECT_TRUE(file_path_builder.Finish(&file_path_array).ok());
    EXPECT_TRUE(pos_builder.Finish(&pos_array).ok());
    const auto schema = arrow::schema(
        {arrow::field("file_path", arrow::utf8(), false), arrow::field("pos", arrow::int64(), false)});
    const auto table = arrow::Table::Make(schema, {file_path_array, pos_array});
    const fs::path path = dir_ / name;
    auto sink_result = arrow::io::FileOutputStream::Open(path.string());
    EXPECT_TRUE(sink_result.ok());
    EXPECT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result,
                                           /*chunk_size=*/deleted_positions.size())
                    .ok());
    return path;
  }

  // One delete-manifest entry per (delete_file_path, content) pair --
  // content is 1 (POSITION_DELETES) or 2 (EQUALITY_DELETES). record_count
  // is the delete file's own row count (unused by resolve_iceberg_table()
  // for position deletes -- it re-reads the delete file's real content
  // instead -- but still a required data_file field).
  fs::path write_delete_manifest(const std::string& name,
                                 const std::vector<std::tuple<std::string, int32_t, int64_t>>& files) {
    AvroFixtureWriter writer(kManifestSchemaWithContentJson);
    std::vector<std::function<void(avro_value_t&)>> rows;
    for (const auto& [path, content, record_count] : files) {
      rows.push_back([path, content, record_count](avro_value_t& v) {
        set_int_field(v, "status", /*ADDED=*/1);
        avro_value_t data_file;
        avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
        set_int_field(data_file, "content", content);
        set_string_field(data_file, "file_path", path);
        set_string_field(data_file, "file_format", "PARQUET");
        set_long_field(data_file, "record_count", record_count);
        set_long_field(data_file, "file_size_in_bytes", 500);
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

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders", {});
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

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders", {});
  server.join();

  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_EQ(resolved.files[0].metadata.row_count, 5);
}

// write_manifest() (kManifestSchemaJson) has no data_file.content field at
// all, so this delete-manifest entry decodes with content defaulting to 0
// (DATA) -- rejected as malformed rather than silently treated as an
// ordinary data file. Real position/equality delete handling (content ==
// 1/2, via kManifestSchemaWithContentJson) is covered by the tests below.
TEST_F(IcebergTableResolutionTest, ThrowsOnDeleteManifestEntryWithoutAContentField) {
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

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders", {})), StorageError);
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

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders", {})), StorageError);
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

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders", {})), StorageError);
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

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders", {});
  server.join();

  EXPECT_TRUE(resolved.files.empty());
  ASSERT_EQ(resolved.schema.field_count(), 1u);
}

TEST_F(IcebergTableResolutionTest, WholeFilePositionDeleteDropsTheFileEntirely) {
  write_data_file(dir_ / "data-0.parquet", 0, 5);
  write_data_file(dir_ / "data-1.parquet", 5, 3);
  const fs::path delete_file = write_position_delete_file(
      "delete-0.parquet", {{(dir_ / "data-0.parquet").string(), 0},
                           {(dir_ / "data-0.parquet").string(), 1},
                           {(dir_ / "data-0.parquet").string(), 2},
                           {(dir_ / "data-0.parquet").string(), 3},
                           {(dir_ / "data-0.parquet").string(), 4}});  // all 5 rows of data-0

  const fs::path data_manifest = write_manifest(
      "m0.avro", {{(dir_ / "data-0.parquet").string(), 5}, {(dir_ / "data-1.parquet").string(), 3}}, 1);
  const fs::path delete_manifest =
      write_delete_manifest("del-0.avro", {{delete_file.string(), /*content=*/1, /*record_count=*/5}});
  const fs::path manifest_list =
      write_manifest_list("snap-42.avro", {{data_manifest.string(), 0}, {delete_manifest.string(), 1}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders", {});
  server.join();

  // data-0.parquet (fully deleted) is gone; data-1.parquet (untouched)
  // survives.
  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_EQ(resolved.files[0].metadata.row_count, 3);
}

TEST_F(IcebergTableResolutionTest, AccumulatesDeletedPositionsAcrossMultipleDeleteFiles) {
  write_data_file(dir_ / "data-0.parquet", 0, 4);
  // Two separate delete files (e.g. two compaction passes), each covering
  // half the rows -- neither alone is a whole-file delete, but together
  // they are.
  const fs::path delete_file_1 = write_position_delete_file(
      "delete-0.parquet", {{(dir_ / "data-0.parquet").string(), 0}, {(dir_ / "data-0.parquet").string(), 1}});
  const fs::path delete_file_2 = write_position_delete_file(
      "delete-1.parquet", {{(dir_ / "data-0.parquet").string(), 2}, {(dir_ / "data-0.parquet").string(), 3}});

  const fs::path data_manifest = write_manifest("m0.avro", {{(dir_ / "data-0.parquet").string(), 4}}, 1);
  const fs::path delete_manifest =
      write_delete_manifest("del-0.avro", {{delete_file_1.string(), /*content=*/1, /*record_count=*/2},
                                           {delete_file_2.string(), /*content=*/1, /*record_count=*/2}});
  const fs::path manifest_list =
      write_manifest_list("snap-42.avro", {{data_manifest.string(), 0}, {delete_manifest.string(), 1}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders", {});
  server.join();

  EXPECT_TRUE(resolved.files.empty());
}

TEST_F(IcebergTableResolutionTest, ThrowsOnPartialPositionDelete) {
  write_data_file(dir_ / "data-0.parquet", 0, 5);
  // Only 2 of the file's 5 rows are deleted -- real per-row filtering
  // isn't implemented (see resolve_iceberg_table()'s own comment), so
  // this must fail loudly rather than silently return all 5 rows.
  const fs::path delete_file = write_position_delete_file(
      "delete-0.parquet", {{(dir_ / "data-0.parquet").string(), 0}, {(dir_ / "data-0.parquet").string(), 1}});

  const fs::path data_manifest = write_manifest("m0.avro", {{(dir_ / "data-0.parquet").string(), 5}}, 1);
  const fs::path delete_manifest =
      write_delete_manifest("del-0.avro", {{delete_file.string(), /*content=*/1, /*record_count=*/2}});
  const fs::path manifest_list =
      write_manifest_list("snap-42.avro", {{data_manifest.string(), 0}, {delete_manifest.string(), 1}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders", {})), StorageError);
  server.join();
}

TEST_F(IcebergTableResolutionTest, ThrowsOnEqualityDeleteFile) {
  write_data_file(dir_ / "data-0.parquet", 0, 5);
  const fs::path data_manifest = write_manifest("m0.avro", {{(dir_ / "data-0.parquet").string(), 5}}, 1);
  // content=2 (EQUALITY_DELETES) -- the delete file's own path is never
  // actually opened/read here; the type check happens first, from
  // manifest metadata alone.
  const fs::path delete_manifest = write_delete_manifest(
      "del-0.avro", {{"s3://warehouse/db/orders/eq-delete-0.parquet", /*content=*/2, /*record_count=*/1}});
  const fs::path manifest_list =
      write_manifest_list("snap-42.avro", {{data_manifest.string(), 0}, {delete_manifest.string(), 1}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders", {})), StorageError);
  server.join();
}

TEST_F(IcebergTableResolutionTest, ThrowsOnUnexpectedDeleteFileContentValue) {
  write_data_file(dir_ / "data-0.parquet", 0, 5);
  const fs::path data_manifest = write_manifest("m0.avro", {{(dir_ / "data-0.parquet").string(), 5}}, 1);
  // Neither a recognized DATA/POSITION_DELETES/EQUALITY_DELETES value --
  // must be rejected, not guessed at as one of the two delete kinds.
  const fs::path delete_manifest = write_delete_manifest(
      "del-0.avro", {{"s3://warehouse/db/orders/weird-0.parquet", /*content=*/99, /*record_count=*/1}});
  const fs::path manifest_list =
      write_manifest_list("snap-42.avro", {{data_manifest.string(), 0}, {delete_manifest.string(), 1}});

  LoopbackHttpServer server;
  server.respond({http_ok_json(load_table_result_json(manifest_list.string()))});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  EXPECT_THROW((void)(resolve_iceberg_table(store_, catalog, {"db"}, "orders", {})), StorageError);
  server.join();
}

// Proves partition pruning actually skips a file, not just that it
// *would*: the "EU" data file's path points at a Parquet file that's
// never written to disk at all -- if resolve_iceberg_table() called
// inspect_parquet_file() on it (i.e. pruning silently didn't run), this
// would throw a real "failed to open" StorageError instead of succeeding
// with just the "US" file's results.
TEST_F(IcebergTableResolutionTest, PartitionPruningSkipsFilesWithoutOpeningThem) {
  write_data_file_with_region(dir_ / "data-us.parquet", 0, 5, "US");
  const fs::path nonexistent_eu_file = dir_ / "data-eu-never-written.parquet";

  const fs::path manifest = write_manifest_with_region_partition(
      "m0.avro", {{(dir_ / "data-us.parquet").string(), 5, "US"}, {nonexistent_eu_file.string(), 3, "EU"}});
  const fs::path manifest_list = write_manifest_list("snap-42.avro", {{manifest.string(), 0}});

  const std::string json = fmt::format(R"json({{
    "metadata": {{
      "format-version": 2,
      "location": "{0}",
      "current-schema-id": 0,
      "schemas": [{{"schema-id": 0, "fields": [
        {{"id": 1, "name": "id", "required": true, "type": "long"}},
        {{"id": 2, "name": "amount", "required": false, "type": "double"}},
        {{"id": 3, "name": "region", "required": false, "type": "string"}}
      ]}}],
      "current-snapshot-id": 42,
      "snapshots": [{{"snapshot-id": 42, "manifest-list": "{1}"}}],
      "partition-specs": [
        {{"spec-id": 0, "fields": [
          {{"source-id": 3, "field-id": 1000, "name": "region", "transform": "identity"}}
        ]}}
      ]
    }}
  }})json",
                                       dir_.string(), manifest_list.string());

  LoopbackHttpServer server;
  server.respond({http_ok_json(json)});

  IcebergCatalogSection config;
  config.catalog_uri = server.base_url();
  IcebergRestCatalogClient catalog(config);

  const std::vector<PushablePredicate> predicates = {
      PushablePredicate{"region", BinaryOperator::Equal,
                        std::make_shared<LiteralExpression>(LiteralExpression::make_string("US"))}};
  const ResolvedTable resolved = resolve_iceberg_table(store_, catalog, {"db"}, "orders", predicates);
  server.join();

  ASSERT_EQ(resolved.files.size(), 1u);
  EXPECT_EQ(resolved.files[0].metadata.row_count, 5);
}

}  // namespace
}  // namespace kernellake::iceberg
