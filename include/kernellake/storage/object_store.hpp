#pragma once

#include <arrow/io/interfaces.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace kernellake {

// A storage location. For the MVP this is always a local filesystem path;
// the scheme() accessor exists so a future S3ObjectStore ("s3://bucket/key")
// can share the same type without every caller needing to know which
// backend is in play.
class Uri {
 public:
  Uri() = default;
  explicit Uri(std::string value) : value_(std::move(value)) {}

  [[nodiscard]] const std::string& value() const noexcept { return value_; }

  [[nodiscard]] std::string_view scheme() const noexcept {
    const std::size_t pos = value_.find("://");
    return pos == std::string::npos ? std::string_view("file") : std::string_view(value_).substr(0, pos);
  }

  [[nodiscard]] bool operator==(const Uri& other) const noexcept { return value_ == other.value_; }
  [[nodiscard]] bool operator<(const Uri& other) const noexcept { return value_ < other.value_; }

 private:
  std::string value_;
};

struct ObjectInfo {
  Uri uri;
  std::uint64_t size_bytes = 0;
};

// Thin, backend-independent wrapper around random-access byte reads.
// as_arrow_file() hands back Arrow's own RandomAccessFile interface (which
// Parquet reading needs) so the Parquet-scanning code never has to know
// which ObjectStore backend produced the bytes.
class RandomAccessObject {
 public:
  virtual ~RandomAccessObject() = default;

  [[nodiscard]] virtual std::uint64_t size() const = 0;
  [[nodiscard]] virtual std::shared_ptr<arrow::io::RandomAccessFile> as_arrow_file() const = 0;
};

class ObjectStore {
 public:
  virtual ~ObjectStore() = default;

  // Lists objects addressed by `prefix`, which may be a single object path,
  // a glob pattern, or (backend-permitting) a directory. Throws
  // StorageError if nothing matches -- an empty result is never returned
  // silently, per KernelLake's "explicit errors for missing paths" rule.
  [[nodiscard]] virtual std::vector<ObjectInfo> list(const Uri& prefix) = 0;

  [[nodiscard]] virtual std::unique_ptr<RandomAccessObject> open(const Uri& uri) = 0;
};

}  // namespace kernellake
