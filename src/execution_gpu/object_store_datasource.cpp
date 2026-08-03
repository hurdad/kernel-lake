#include "kernellake/execution_gpu/object_store_datasource.hpp"

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>

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
    throw StorageError("failed to read " + std::to_string(size) + " bytes at offset " +
                       std::to_string(offset) + ": " + result.status().ToString());
  }
  return std::make_unique<ArrowBufferDatasourceBuffer>(*result);
}

std::size_t ObjectStoreDatasource::host_read(std::size_t offset, std::size_t size, std::uint8_t* dst) {
  const arrow::Result<std::int64_t> result = object_->as_arrow_file()->ReadAt(
      static_cast<std::int64_t>(offset), static_cast<std::int64_t>(size), dst);
  if (!result.ok()) {
    throw StorageError("failed to read " + std::to_string(size) + " bytes at offset " +
                       std::to_string(offset) + ": " + result.status().ToString());
  }
  return static_cast<std::size_t>(*result);
}

}  // namespace kernellake
