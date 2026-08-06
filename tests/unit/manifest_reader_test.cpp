#include "kernellake/iceberg/manifest_reader.hpp"

#include <avro.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake::iceberg {
namespace {

namespace fs = std::filesystem;

// Simplified stand-ins for the real Iceberg v2 Avro schemas: same field
// names/nesting this reader looks up by name/position, minus the
// attributes (field-id, logicalType, the count/stats maps) it doesn't
// read -- so these fixtures are realistic enough to exercise the reader
// without needing the full spec schema.
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
        {"name": "partition", "type": {
          "type": "record",
          "name": "r102",
          "fields": [
            {"name": "region", "type": ["null", "string"]},
            {"name": "day", "type": ["null", "int"]}
          ]
        }}
      ]
    }}
  ]
})";

class AvroFixtureWriter {
 public:
  explicit AvroFixtureWriter(const std::string& schema_json) {
    if (avro_schema_from_json_length(schema_json.c_str(), schema_json.size(), &schema_) != 0) {
      throw std::runtime_error("test fixture: invalid schema JSON");
    }
    iface_ = avro_generic_class_from_schema(schema_);
  }

  ~AvroFixtureWriter() {
    if (iface_ != nullptr) {
      avro_value_iface_decref(iface_);
    }
    if (schema_ != nullptr) {
      avro_schema_decref(schema_);
    }
  }

  // Writes `rows` (each a callback that fills one avro_value_t) to a new
  // Object Container File at `path`, returning the raw bytes written.
  std::string write(const fs::path& path, const std::vector<std::function<void(avro_value_t&)>>& rows) {
    avro_file_writer_t writer = nullptr;
    if (avro_file_writer_create(path.c_str(), schema_, &writer) != 0) {
      throw std::runtime_error(std::string("test fixture: avro_file_writer_create failed: ") + avro_strerror());
    }
    avro_value_t value;
    avro_generic_value_new(iface_, &value);
    for (const auto& fill_row : rows) {
      fill_row(value);
      if (avro_file_writer_append_value(writer, &value) != 0) {
        throw std::runtime_error(std::string("test fixture: append failed: ") + avro_strerror());
      }
      avro_value_reset(&value);
    }
    avro_value_decref(&value);
    avro_file_writer_close(writer);

    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
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

void set_optional_string_field(avro_value_t& record, const char* name, const std::string& s) {
  avro_value_t field;
  avro_value_get_by_name(&record, name, &field, nullptr);
  avro_value_t branch;
  avro_value_set_branch(&field, 1, &branch);
  avro_value_set_string(&branch, s.c_str());
}

void set_optional_int_field(avro_value_t& record, const char* name, int32_t v) {
  avro_value_t field;
  avro_value_get_by_name(&record, name, &field, nullptr);
  avro_value_t branch;
  avro_value_set_branch(&field, 1, &branch);
  avro_value_set_int(&branch, v);
}

class ManifestReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_manifest_reader_test_" +
                     std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                     ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
};

TEST_F(ManifestReaderTest, ReadsManifestListEntriesInOrder) {
  AvroFixtureWriter writer(kManifestListSchemaJson);
  const std::string bytes = writer.write(dir_ / "snap-1.avro",
      {
          [](avro_value_t& v) {
            set_string_field(v, "manifest_path", "s3://warehouse/db/orders/metadata/m0.avro");
            set_long_field(v, "manifest_length", 1234);
            set_int_field(v, "partition_spec_id", 0);
            set_int_field(v, "content", 0);
            set_long_field(v, "added_snapshot_id", 42);
          },
          [](avro_value_t& v) {
            set_string_field(v, "manifest_path", "s3://warehouse/db/orders/metadata/m1.avro");
            set_long_field(v, "manifest_length", 5678);
            set_int_field(v, "partition_spec_id", 0);
            set_int_field(v, "content", 1);
            set_long_field(v, "added_snapshot_id", 42);
          },
      });

  const std::vector<ManifestListEntry> entries = read_manifest_list_bytes(bytes);
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].manifest_path, "s3://warehouse/db/orders/metadata/m0.avro");
  EXPECT_EQ(entries[0].manifest_length, 1234);
  EXPECT_EQ(entries[0].content, 0);
  EXPECT_EQ(entries[0].added_snapshot_id, 42);
  EXPECT_EQ(entries[1].manifest_path, "s3://warehouse/db/orders/metadata/m1.avro");
  EXPECT_EQ(entries[1].content, 1);
}

TEST_F(ManifestReaderTest, ReadsManifestDataFileEntriesWithPartitionValues) {
  AvroFixtureWriter writer(kManifestSchemaJson);
  const std::string bytes = writer.write(dir_ / "m0.avro",
      {
          [](avro_value_t& v) {
            set_int_field(v, "status", 1);
            avro_value_t data_file;
            avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
            set_string_field(data_file, "file_path", "s3://warehouse/db/orders/data/region=US/part-0.parquet");
            set_string_field(data_file, "file_format", "PARQUET");
            set_long_field(data_file, "record_count", 1000);
            set_long_field(data_file, "file_size_in_bytes", 999999);
            avro_value_t partition;
            avro_value_get_by_name(&data_file, "partition", &partition, nullptr);
            set_optional_string_field(partition, "region", "US");
            set_optional_int_field(partition, "day", 19000);
          },
      });

  const std::vector<ManifestDataFileEntry> entries = read_manifest_bytes(bytes);
  ASSERT_EQ(entries.size(), 1u);
  const ManifestDataFileEntry& entry = entries[0];
  EXPECT_EQ(entry.status, 1);
  EXPECT_EQ(entry.file_path, "s3://warehouse/db/orders/data/region=US/part-0.parquet");
  EXPECT_EQ(entry.file_format, "PARQUET");
  EXPECT_EQ(entry.record_count, 1000);
  EXPECT_EQ(entry.file_size_in_bytes, 999999);
  ASSERT_EQ(entry.partition_values.size(), 2u);
  ASSERT_TRUE(std::holds_alternative<std::string>(entry.partition_values[0]));
  EXPECT_EQ(std::get<std::string>(entry.partition_values[0]), "US");
  ASSERT_TRUE(std::holds_alternative<int64_t>(entry.partition_values[1]));
  EXPECT_EQ(std::get<int64_t>(entry.partition_values[1]), 19000);
}

TEST_F(ManifestReaderTest, DecodesNullPartitionFieldAsMonostate) {
  AvroFixtureWriter writer(kManifestSchemaJson);
  const std::string bytes = writer.write(dir_ / "m0.avro",
      {
          [](avro_value_t& v) {
            set_int_field(v, "status", 1);
            avro_value_t data_file;
            avro_value_get_by_name(&v, "data_file", &data_file, nullptr);
            set_string_field(data_file, "file_path", "s3://warehouse/db/orders/data/part-0.parquet");
            set_string_field(data_file, "file_format", "PARQUET");
            set_long_field(data_file, "record_count", 1);
            set_long_field(data_file, "file_size_in_bytes", 1);
            avro_value_t partition;
            avro_value_get_by_name(&data_file, "partition", &partition, nullptr);
            avro_value_t region_field;
            avro_value_get_by_name(&partition, "region", &region_field, nullptr);
            avro_value_t branch;
            avro_value_set_branch(&region_field, 0, &branch);  // null branch
            avro_value_t day_field;
            avro_value_get_by_name(&partition, "day", &day_field, nullptr);
            avro_value_set_branch(&day_field, 0, &branch);
          },
      });

  const std::vector<ManifestDataFileEntry> entries = read_manifest_bytes(bytes);
  ASSERT_EQ(entries.size(), 1u);
  ASSERT_EQ(entries[0].partition_values.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(entries[0].partition_values[0]));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(entries[0].partition_values[1]));
}

TEST_F(ManifestReaderTest, ThrowsOnNonAvroBytes) {
  EXPECT_THROW((void)(read_manifest_list_bytes("this is not an avro object container file")), StorageError);
}

TEST_F(ManifestReaderTest, ThrowsOnEmptyBytes) {
  EXPECT_THROW((void)(read_manifest_list_bytes("")), StorageError);
}

TEST_F(ManifestReaderTest, ReadsManifestListThroughObjectStore) {
  AvroFixtureWriter writer(kManifestListSchemaJson);
  const fs::path path = dir_ / "snap-1.avro";
  writer.write(path,
      {
          [](avro_value_t& v) {
            set_string_field(v, "manifest_path", "s3://warehouse/db/orders/metadata/m0.avro");
            set_long_field(v, "manifest_length", 1234);
            set_int_field(v, "partition_spec_id", 0);
            set_int_field(v, "content", 0);
            set_long_field(v, "added_snapshot_id", 42);
          },
      });

  LocalObjectStore store;
  const std::vector<ManifestListEntry> entries = read_manifest_list(store, Uri(path.string()));
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].manifest_path, "s3://warehouse/db/orders/metadata/m0.avro");
  EXPECT_EQ(entries[0].added_snapshot_id, 42);
}

}  // namespace
}  // namespace kernellake::iceberg
