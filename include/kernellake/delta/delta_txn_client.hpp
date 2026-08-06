#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernellake/common/config.hpp"

namespace kernellake::delta {

// One data/delete-adjacent file entry from ListActiveFiles, mirroring the
// wire AddFile message's fields this client actually needs.
// Column-level stats aren't extracted -- nothing consumes them yet, same
// scope decision as kernellake::iceberg::ManifestDataFileEntry.
struct DeltaActiveFile {
  std::string path;
  int64_t size = 0;
  int64_t modification_time = 0;
  std::unordered_map<std::string, std::string> partition_values;
  int64_t record_count = 0;  // 0 when the file has no stats (FileStats.num_records)
};

// The table-level info both GetTable and ListActiveFiles' header message
// return. `schema_string` is Delta's own JSON schema encoding (Spark-
// compatible) -- see schema_translation.hpp for how it becomes a
// kernellake::Schema.
struct DeltaTableInfo {
  int64_t version = 0;
  std::string schema_string;
  std::vector<std::string> partition_columns;
};

struct DeltaActiveFileListing {
  DeltaTableInfo table;
  std::vector<DeltaActiveFile> files;
};

// A gRPC client for one delta-txn-service deployment (DeltaSection,
// kernellake/common/config.hpp) -- table inspection (GetTable) and
// active-file listing (ListActiveFiles) reads. `config` is assumed
// non-empty (grpc_endpoint set) -- callers check that themselves (see
// DeltaSourceResolver) since an empty DeltaSection is a valid "not
// configured" state elsewhere in this codebase, not something this class
// should silently tolerate; the constructor throws ConfigurationError if
// grpc_endpoint is empty.
//
// Deliberately pimpl'd: the generated delta_txn.pb.h/delta_txn.grpc.pb.h
// and <grpcpp/grpcpp.h> stay out of this header so consumers don't need
// them just to hold a DeltaTxnClient by pointer/reference.
class DeltaTxnClient final {
 public:
  explicit DeltaTxnClient(DeltaSection config);
  ~DeltaTxnClient();

  DeltaTxnClient(const DeltaTxnClient&) = delete;
  DeltaTxnClient& operator=(const DeltaTxnClient&) = delete;

  // Throws StorageError on any transport failure, non-OK gRPC status, or
  // a response missing its required `metadata` field.
  [[nodiscard]] DeltaTableInfo get_table(const std::string& table_uri);

  // Throws StorageError on any transport failure, non-OK gRPC status, or
  // a stream that never sends its required header message.
  [[nodiscard]] DeltaActiveFileListing list_active_files(const std::string& table_uri);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernellake::delta
