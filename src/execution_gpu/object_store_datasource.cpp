#include "kernellake/execution_gpu/object_store_datasource.hpp"

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <fmt/format.h>

#include <cuda/memory_resource>
#include <rmm/device_buffer.hpp>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"

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

// Wraps device memory for device_read()'s buffer-returning overload -- the
// same pattern as ArrowBufferDatasourceBuffer above, just device- instead
// of host-resident. data() returns a device pointer here; callers of
// device_read() (as opposed to host_read()) already know that.
class DeviceBufferDatasourceBuffer final : public cudf::io::datasource::buffer {
 public:
  explicit DeviceBufferDatasourceBuffer(rmm::device_buffer buffer) : buffer_(std::move(buffer)) {}

  [[nodiscard]] std::size_t size() const override { return buffer_.size(); }
  [[nodiscard]] const std::uint8_t* data() const override {
    return static_cast<const std::uint8_t*>(buffer_.data());
  }

 private:
  rmm::device_buffer buffer_;
};

}  // namespace

ObjectStoreDatasource::ObjectStoreDatasource(std::unique_ptr<RandomAccessObject> object,
                                             const rmm::device_async_resource_ref& mr)
    : object_(std::move(object)), mr_(mr) {}

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

std::unique_ptr<cudf::io::datasource::buffer> ObjectStoreDatasource::device_read(
    std::size_t offset, std::size_t size, rmm::cuda_stream_view stream) {
  // mr_ explicitly, not the (size, stream) overload's implicit default
  // resource -- see this class's own constructor comment for why.
  // device_buffer's mr parameter is a cuda::mr::any_resource (owning,
  // type-erased), not a device_async_resource_ref (non-owning) like mr_
  // itself -- wrapping mr_ in any_resource here type-erases a *copy of
  // the reference* (cheap: a vtable pointer + object pointer, same as
  // rmm_environment.cpp's own any_resource<...>{limiter} pattern, just
  // wrapping a reference-to-a-resource here instead of a resource
  // directly), not a copy of whatever it points to -- every allocate/
  // deallocate call through it still reaches the real, single upstream
  // resource mr_ was constructed from.
  rmm::device_buffer buffer(size, stream, cuda::mr::any_resource<cuda::mr::device_accessible>{mr_});
  const std::size_t bytes_read = device_read(offset, size, static_cast<std::uint8_t*>(buffer.data()), stream);
  if (bytes_read != size) {
    // A short read (offset+size ran past the real object size -- callers do
    // ask for this deliberately sometimes, e.g. a generous end-of-file
    // range guess) -- shrink to what was actually written so size()
    // reports the truth, same as host_read()'s ArrowBufferDatasourceBuffer
    // already does implicitly (its wrapped arrow::Buffer is sized to
    // exactly what ReadAt() returned).
    buffer.resize(bytes_read, stream);
  }
  return std::make_unique<DeviceBufferDatasourceBuffer>(std::move(buffer));
}

std::size_t ObjectStoreDatasource::device_read(std::size_t offset, std::size_t size, std::uint8_t* dst,
                                               rmm::cuda_stream_view stream) {
  return device_read_async(offset, size, dst, stream).get();
}

// This is the override that actually matters -- see class comment.
// std::launch::async (not deferred): the whole point is that this starts
// running -- including the real, blocking network host_read() below --
// the moment this function is called, not when the returned future is
// waited on. cudf's own read_column_chunks_async() (confirmed by reading
// v26.06.00's actual source) calls this once per coalesced column-chunk
// group in a plain loop, collecting every returned future before waiting
// on any of them -- so by the time it starts waiting, every one of these
// host_read() calls is already running concurrently on its own thread.
std::future<std::size_t> ObjectStoreDatasource::device_read_async(std::size_t offset, std::size_t size,
                                                                  std::uint8_t* dst,
                                                                  rmm::cuda_stream_view stream) {
  return std::async(std::launch::async, [this, offset, size, dst, stream] {
    const std::unique_ptr<cudf::io::datasource::buffer> host_buffer = host_read(offset, size);
    const std::size_t bytes_read = host_buffer->size();
    // cudaMemcpyAsync, not a plain memcpy-then-return: this enqueues the
    // host-to-device copy on `stream` -- the same stream every other decode
    // operation for this scan uses (see ParquetScanOperator's
    // decode_stream_) -- so any kernel cudf launches on that stream *after*
    // this future's .get() returns is correctly ordered after this copy,
    // with no separate host-side synchronize needed here. Safe regarding
    // host_buffer's lifetime specifically because it is *pageable* memory
    // (ReadAt() returns a plain heap buffer, not a pinned one) -- CUDA's
    // own documented behavior is that a host-to-device cudaMemcpyAsync from
    // pageable memory blocks the calling thread until the copy completes
    // (the driver has to stage it through an internal pinned buffer first),
    // so host_buffer is never touched-after-free even though this call is
    // nominally "async".
    check_cuda(cudaMemcpyAsync(dst, host_buffer->data(), bytes_read, cudaMemcpyHostToDevice, stream.value()),
               "cudaMemcpyAsync in ObjectStoreDatasource::device_read_async");
    return bytes_read;
  });
}

}  // namespace kernellake
