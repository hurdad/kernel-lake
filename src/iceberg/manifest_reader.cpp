#include "kernellake/iceberg/manifest_reader.hpp"

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <avro.h>
#include <fmt/format.h>

#include <cstdio>
#include <memory>

#include "kernellake/common/errors.hpp"

namespace kernellake::iceberg {

namespace {

struct AvroFileReaderDeleter {
  void operator()(avro_file_reader_t reader) const {
    if (reader != nullptr) {
      avro_file_reader_close(reader);
    }
  }
};

// avro_file_reader_t is already a pointer typedef (to an opaque struct), so
// unique_ptr's template argument is the pointee type it points at.
using AvroFileReaderPtr = std::unique_ptr<std::remove_pointer_t<avro_file_reader_t>, AvroFileReaderDeleter>;

struct AvroValueIfaceDeleter {
  void operator()(avro_value_iface_t* iface) const { avro_value_iface_decref(iface); }
};
using AvroValueIfacePtr = std::unique_ptr<avro_value_iface_t, AvroValueIfaceDeleter>;

// avro-c's file-container reader (avro_file_reader/avro_file_reader_fp) only
// takes a local path or FILE*, never an in-memory buffer directly -- but
// fmemopen() (POSIX) wraps an already-in-memory buffer as a FILE*, letting
// this reader work directly off bytes an ObjectStore already fetched
// (S3/GCS/Azure/HDFS/local) without writing a real temp file to disk.
AvroFileReaderPtr open_avro_reader(const std::string& avro_bytes, const char* debug_name) {
  // fmemopen's buffer isn't modified in "rb" mode, but the API doesn't
  // accept a const pointer.
  FILE* fp = fmemopen(const_cast<char*>(avro_bytes.data()), avro_bytes.size(), "rb");
  if (fp == nullptr) {
    throw StorageError(fmt::format("iceberg {}: fmemopen() failed", debug_name));
  }
  avro_file_reader_t raw_reader = nullptr;
  // should_close = 1: avro_file_reader_close() below takes ownership of fp.
  //
  // Verified under ASan+LSan: on this branch (malformed/truncated input --
  // avro_file_reader_fp() itself failed to parse the container header),
  // system libavro-c 1.12.0 (Ubuntu 26.04's libavro24 package) leaks a
  // fixed ~40 bytes it allocated internally (avro_reader_memory(), called
  // from inside avro_file_reader_fp()) before detecting the error and
  // returning nonzero -- never on a successful open. There is no handle to
  // that allocation in the public API to free ourselves: avro_reader_t is
  // constructed and owned entirely inside avro_file_reader_fp() on this
  // path. A one-time, bounded, per-call leak strictly on the "someone
  // handed us corrupt bytes" error path (immediately followed by throwing
  // below) -- a real upstream bug, not fixable from this call site.
  if (avro_file_reader_fp(fp, debug_name, /*should_close=*/1, &raw_reader) != 0) {
    throw StorageError(
        fmt::format("iceberg {}: failed to open as an Avro container file: {}", debug_name, avro_strerror()));
  }
  return AvroFileReaderPtr(raw_reader);
}

std::string read_all_bytes(ObjectStore& store, const Uri& uri) {
  const std::unique_ptr<RandomAccessObject> object = store.open(uri);
  const std::shared_ptr<arrow::io::RandomAccessFile> file = object->as_arrow_file();
  const arrow::Result<std::shared_ptr<arrow::Buffer>> result =
      file->ReadAt(0, static_cast<int64_t>(object->size()));
  if (!result.ok()) {
    throw StorageError(
        fmt::format("iceberg: failed to read '{}': {}", uri.value(), result.status().ToString()));
  }
  const std::shared_ptr<arrow::Buffer>& buffer = *result;
  return std::string(reinterpret_cast<const char*>(buffer->data()), static_cast<size_t>(buffer->size()));
}

// Fetches `field_name` off `record` as a required int32. Handles both a
// bare "int" field and a ["null", "int"]-shaped union with a non-null
// value present (Iceberg schemas mix both styles for "required" fields
// depending on spec version); a null branch is treated as 0, matching the
// spec's convention of defaulting these particular counters.
int32_t get_required_int(avro_value_t* record, const char* field_name, const char* debug_name) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, nullptr) != 0) {
    throw StorageError(fmt::format("iceberg {}: missing required field '{}'", debug_name, field_name));
  }
  avro_value_t target = field;
  if (avro_value_get_type(&field) == AVRO_UNION) {
    avro_value_t branch;
    if (avro_value_get_current_branch(&field, &branch) != 0) {
      throw StorageError(
          fmt::format("iceberg {}: couldn't resolve union branch for '{}'", debug_name, field_name));
    }
    target = branch;
  }
  if (avro_value_get_type(&target) == AVRO_NULL) {
    return 0;
  }
  int32_t value = 0;
  if (avro_value_get_int(&target, &value) != 0) {
    throw StorageError(fmt::format("iceberg {}: field '{}' is not an int", debug_name, field_name));
  }
  return value;
}

// Like get_required_int(), but for a field the Avro *schema* itself may
// not declare at all (not just a present-but-null value) -- v1 manifests,
// and every pre-existing test fixture's manifest schema, predate
// data_file.content, so avro_value_get_by_name() failing to even find the
// field is expected, not an error; `default_value` covers that case the
// same way a present-but-null field is already covered below.
int32_t get_optional_int(avro_value_t* record, const char* field_name, int32_t default_value) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, nullptr) != 0) {
    return default_value;
  }
  avro_value_t target = field;
  if (avro_value_get_type(&field) == AVRO_UNION) {
    avro_value_t branch;
    if (avro_value_get_current_branch(&field, &branch) != 0) {
      return default_value;
    }
    target = branch;
  }
  if (avro_value_get_type(&target) == AVRO_NULL) {
    return default_value;
  }
  int32_t value = 0;
  if (avro_value_get_int(&target, &value) != 0) {
    return default_value;
  }
  return value;
}

int64_t get_required_long(avro_value_t* record, const char* field_name, const char* debug_name) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, nullptr) != 0) {
    throw StorageError(fmt::format("iceberg {}: missing required field '{}'", debug_name, field_name));
  }
  avro_value_t target = field;
  if (avro_value_get_type(&field) == AVRO_UNION) {
    avro_value_t branch;
    if (avro_value_get_current_branch(&field, &branch) != 0) {
      throw StorageError(
          fmt::format("iceberg {}: couldn't resolve union branch for '{}'", debug_name, field_name));
    }
    target = branch;
  }
  if (avro_value_get_type(&target) == AVRO_NULL) {
    return 0;
  }
  int64_t value = 0;
  if (avro_value_get_long(&target, &value) != 0) {
    throw StorageError(fmt::format("iceberg {}: field '{}' is not a long", debug_name, field_name));
  }
  return value;
}

std::string get_required_string(avro_value_t* record, const char* field_name, const char* debug_name) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, nullptr) != 0) {
    throw StorageError(fmt::format("iceberg {}: missing required field '{}'", debug_name, field_name));
  }
  avro_value_t target = field;
  if (avro_value_get_type(&field) == AVRO_UNION) {
    avro_value_t branch;
    if (avro_value_get_current_branch(&field, &branch) != 0) {
      throw StorageError(
          fmt::format("iceberg {}: couldn't resolve union branch for '{}'", debug_name, field_name));
    }
    target = branch;
  }
  const char* str = nullptr;
  size_t size = 0;
  if (avro_value_get_string(&target, &str, &size) != 0) {
    throw StorageError(fmt::format("iceberg {}: field '{}' is not a string", debug_name, field_name));
  }
  // avro-c's generic string values are always NUL-terminated internally;
  // using strlen (via the std::string(const char*) ctor) instead of `size`
  // sidesteps needing to know whether `size` includes that terminator.
  return std::string(str);
}

// Decodes one field of a manifest entry's "partition" struct by its Avro
// type, resolving a ["null", T] union to its live branch first. Iceberg
// partition transforms only ever produce int/long/string/null at this
// layer (dates and timestamps are encoded as int/long; only their
// *meaning*, which needs the partition spec this reader doesn't have,
// distinguishes them) -- any other Avro type is a schema this reader
// doesn't understand and is reported as null rather than guessed at.
PartitionFieldValue decode_partition_field(avro_value_t* field) {
  avro_value_t target = *field;
  if (avro_value_get_type(field) == AVRO_UNION) {
    avro_value_t branch;
    if (avro_value_get_current_branch(field, &branch) != 0) {
      return std::monostate{};
    }
    target = branch;
  }
  switch (avro_value_get_type(&target)) {
    case AVRO_INT32: {
      int32_t value = 0;
      avro_value_get_int(&target, &value);
      return static_cast<int64_t>(value);
    }
    case AVRO_INT64: {
      int64_t value = 0;
      avro_value_get_long(&target, &value);
      return value;
    }
    case AVRO_STRING: {
      const char* str = nullptr;
      size_t size = 0;
      avro_value_get_string(&target, &str, &size);
      return std::string(str);
    }
    default:
      return std::monostate{};
  }
}

std::vector<PartitionFieldValue> decode_partition_struct(avro_value_t* data_file, const char* debug_name) {
  avro_value_t partition;
  if (avro_value_get_by_name(data_file, "partition", &partition, nullptr) != 0) {
    throw StorageError(fmt::format("iceberg {}: data_file is missing 'partition'", debug_name));
  }
  size_t field_count = 0;
  if (avro_value_get_size(&partition, &field_count) != 0) {
    throw StorageError(fmt::format("iceberg {}: couldn't size 'partition' struct", debug_name));
  }
  std::vector<PartitionFieldValue> values;
  values.reserve(field_count);
  for (size_t i = 0; i < field_count; ++i) {
    avro_value_t field;
    if (avro_value_get_by_index(&partition, static_cast<int>(i), &field, nullptr) != 0) {
      throw StorageError(fmt::format("iceberg {}: couldn't read partition field {}", debug_name, i));
    }
    values.push_back(decode_partition_field(&field));
  }
  return values;
}

}  // namespace

std::vector<ManifestListEntry> read_manifest_list_bytes(const std::string& avro_bytes) {
  constexpr const char* kDebugName = "manifest-list";
  const AvroFileReaderPtr reader = open_avro_reader(avro_bytes, kDebugName);

  avro_schema_t schema = avro_file_reader_get_writer_schema(reader.get());
  const AvroValueIfacePtr iface(avro_generic_class_from_schema(schema));
  avro_schema_decref(schema);
  if (!iface) {
    throw StorageError(
        fmt::format("iceberg {}: couldn't build a value class from the writer schema", kDebugName));
  }

  avro_value_t value;
  if (avro_generic_value_new(iface.get(), &value) != 0) {
    throw StorageError(fmt::format("iceberg {}: couldn't instantiate a value", kDebugName));
  }

  std::vector<ManifestListEntry> entries;
  while (avro_file_reader_read_value(reader.get(), &value) == 0) {
    ManifestListEntry entry;
    entry.manifest_path = get_required_string(&value, "manifest_path", kDebugName);
    entry.manifest_length = get_required_long(&value, "manifest_length", kDebugName);
    entry.partition_spec_id = get_required_int(&value, "partition_spec_id", kDebugName);
    entry.content = get_required_int(&value, "content", kDebugName);
    entry.added_snapshot_id = get_required_long(&value, "added_snapshot_id", kDebugName);
    entries.push_back(std::move(entry));
    avro_value_reset(&value);
  }
  avro_value_decref(&value);
  return entries;
}

std::vector<ManifestDataFileEntry> read_manifest_bytes(const std::string& avro_bytes) {
  constexpr const char* kDebugName = "manifest";
  const AvroFileReaderPtr reader = open_avro_reader(avro_bytes, kDebugName);

  avro_schema_t schema = avro_file_reader_get_writer_schema(reader.get());
  const AvroValueIfacePtr iface(avro_generic_class_from_schema(schema));
  avro_schema_decref(schema);
  if (!iface) {
    throw StorageError(
        fmt::format("iceberg {}: couldn't build a value class from the writer schema", kDebugName));
  }

  avro_value_t value;
  if (avro_generic_value_new(iface.get(), &value) != 0) {
    throw StorageError(fmt::format("iceberg {}: couldn't instantiate a value", kDebugName));
  }

  std::vector<ManifestDataFileEntry> entries;
  while (avro_file_reader_read_value(reader.get(), &value) == 0) {
    ManifestDataFileEntry entry;
    entry.status = get_required_int(&value, "status", kDebugName);

    avro_value_t data_file;
    if (avro_value_get_by_name(&value, "data_file", &data_file, nullptr) != 0) {
      throw StorageError(fmt::format("iceberg {}: entry is missing 'data_file'", kDebugName));
    }
    entry.file_path = get_required_string(&data_file, "file_path", kDebugName);
    entry.file_format = get_required_string(&data_file, "file_format", kDebugName);
    entry.record_count = get_required_long(&data_file, "record_count", kDebugName);
    entry.file_size_in_bytes = get_required_long(&data_file, "file_size_in_bytes", kDebugName);
    entry.partition_values = decode_partition_struct(&data_file, kDebugName);
    entry.content = get_optional_int(&data_file, "content", /*default_value=*/0);

    entries.push_back(std::move(entry));
    avro_value_reset(&value);
  }
  avro_value_decref(&value);
  return entries;
}

std::vector<ManifestListEntry> read_manifest_list(ObjectStore& store, const Uri& uri) {
  return read_manifest_list_bytes(read_all_bytes(store, uri));
}

std::vector<ManifestDataFileEntry> read_manifest(ObjectStore& store, const Uri& uri) {
  return read_manifest_bytes(read_all_bytes(store, uri));
}

}  // namespace kernellake::iceberg
