#include "kernellake/execution_gpu/object_store_datasource.hpp"

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <fmt/format.h>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

// Wraps an already-read arrow::Buffer directly, rather than copying into a
// std::vector<uint8_t> via datasource::buffer::create() -- avoids an extra
// host-memory copy on every cudf-driven ranged read.
class ArrowBufferDatasourceBuffer final : public cudf::io::datasource::buffer {
 public:
  explicit ArrowBufferDatasourceBuffer(std::shared_ptr<arrow::Buffer> buffer) : buffer_(std::move(buffer)) {}

  [[nodiscard]] std::size_t size() const override { return static_cast<std::size_t>(buffer_->size()); }
  [[nodiscard]] const std::uint8_t* data() const override { return buffer_->data(); }

 private:
  std::shared_ptr<arrow::Buffer> buffer_;
};

}  // namespace

ObjectStoreDatasource::ObjectStoreDatasource(std::unique_ptr<RandomAccessObject> object)
    : object_(std::move(object)) {}

std::size_t ObjectStoreDatasource::size() const {
  return static_cast<std::size_t>(object_->size());
}

std::unique_ptr<cudf::io::datasource::buffer> ObjectStoreDatasource::host_read(std::size_t offset,
                                                                               std::size_t size) {
  const arrow::Result<std::shared_ptr<arrow::Buffer>> result =
      object_->as_arrow_file()->ReadAt(static_cast<std::int64_t>(offset), static_cast<std::int64_t>(size));
  if (!result.ok()) {
    throw StorageError(
        fmt::format("failed to read {} bytes at offset {}: {}", size, offset, result.status().ToString()));
  }
  return std::make_unique<ArrowBufferDatasourceBuffer>(*result);
}

std::size_t ObjectStoreDatasource::host_read(std::size_t offset, std::size_t size, std::uint8_t* dst) {
  const arrow::Result<std::int64_t> result = object_->as_arrow_file()->ReadAt(
      static_cast<std::int64_t>(offset), static_cast<std::int64_t>(size), dst);
  if (!result.ok()) {
    throw StorageError(
        fmt::format("failed to read {} bytes at offset {}: {}", size, offset, result.status().ToString()));
  }
  return static_cast<std::size_t>(*result);
}

// std::launch::async, not the default (cudf::io::datasource's own base
// implementation, which these two overrides replace, uses
// std::launch::deferred, which would serialize every "concurrent" read a
// caller issues instead of actually overlapping them). ReadAt() is safe to
// call concurrently from multiple threads against the same
// arrow::io::RandomAccessFile (an explicit Arrow guarantee for the
// ReadAt-style API specifically, unlike stateful Read()), so no locking is
// needed here beyond that.
//
// Confirmed *not* the fix for the real-S3 scan-throughput regression these
// overrides were originally added for (see docs/GPU_OPTIMIZATIONS.md):
// instrumented for real against a live S3 bucket and cudf's
// chunked_parquet_reader (the multi-datasource constructor, used for any
// non-local scan) never calls host_read_async() at all here -- every one
// of 156 real reads for one query went through the plain synchronous
// host_read(), issued one at a time from a single thread inside
// has_next(), not read_chunk(). That serialization is what actually
// dominates wall time; these async overrides are still a real fix for a
// real bug (a caller that *does* use host_read_async() would otherwise get
// silently-serialized reads), just not the one that mattered here. See
// ParquetScanOperator's own decode_seconds_ comment for where the real
// time goes instead.
std::future<std::unique_ptr<cudf::io::datasource::buffer>> ObjectStoreDatasource::host_read_async(
    std::size_t offset, std::size_t size) {
  return std::async(std::launch::async, [this, offset, size] { return host_read(offset, size); });
}

std::future<std::size_t> ObjectStoreDatasource::host_read_async(std::size_t offset, std::size_t size,
                                                                 std::uint8_t* dst) {
  return std::async(std::launch::async, [this, offset, size, dst] { return host_read(offset, size, dst); });
}

}  // namespace kernellake
