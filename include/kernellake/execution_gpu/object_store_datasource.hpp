#pragma once

#include <cudf/io/datasource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <future>
#include <memory>

#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Wraps a RandomAccessObject (i.e. whatever ObjectStore::open() returned --
// local, S3, GCS, or Azure, this class doesn't know or care which) as a
// cudf::io::datasource, so cudf's chunked_parquet_reader can read from any
// ObjectStore backend, not just local files. host_read() forwards directly
// to arrow::io::RandomAccessFile::ReadAt(), so cudf's own pass-budgeted
// reads (see ParquetScanOperator's pass_read_limit_bytes) still drive
// bounded, ranged reads against the backend -- this never pre-loads a whole
// remote object into host memory.
//
// host_read_async() is overridden below for its own sake (cudf's base-class
// default wraps host_read() in std::launch::deferred, which is a real bug
// for any caller that actually invokes it), but it turned out **not** to be
// what fixes this class's own real-S3 scan-throughput regression (0.07-0.17
// GB/s vs. 1.5-2.7 GB/s locally, same SF10 queries -- see
// docs/GPU_OPTIMIZATIONS.md). Confirmed by reading cudf v26.06.00's actual
// source (cpp/src/io/parquet/reader_impl_preprocess_utils.cu,
// read_column_chunks_async()): for any datasource with
// supports_device_read() == false (the base-class default), cudf never
// calls host_read_async() at all -- it calls the plain synchronous
// host_read() directly, wraps *that specific call* in its own
// std::launch::deferred future, then waits on every chunk's future in
// sequence, one at a time. host_read_async() being real or fake is simply
// never consulted on that path.
//
// The actual fix: for a *device*-preferred source, cudf takes whatever
// future device_read_async() returns completely unmodified -- if that
// future represents work already launched (not deferred), every
// column-chunk read for a pass gets kicked off concurrently in cudf's own
// coalescing loop, and the later sequential .get() calls just wait for
// work already in flight (wall time -> max() of the reads instead of
// sum()). supports_device_read() = true routes calls through that branch;
// device_read_async() launches a real host_read() concurrently
// (std::launch::async, same reasoning as host_read_async() above) and
// copies the result into device memory via a stream-ordered
// cudaMemcpyAsync -- for the *pageable* host buffer ReadAt() returns, that
// call blocks the launching thread until the copy is done (CUDA's own
// documented behavior for non-pinned sources), so the host buffer's
// lifetime is never at risk even though the copy is issued "async".
class ObjectStoreDatasource final : public cudf::io::datasource {
 public:
  // `mr` is the calling query's own memory resource (ExecutionContext::
  // memory_resource) -- device_read()'s buffer-returning overload below
  // needs it explicitly rather than falling back to cudf's default
  // (cudf::get_current_device_resource_ref(), the ambient process-wide
  // resource): harmless when only one query ever runs at a time, but once
  // GpuExecutionCoordinator can run several concurrently (see that
  // class's own comment), the ambient default no longer means "this
  // query's resource" -- it means "whatever's globally current right
  // now," which could belong to any of them.
  ObjectStoreDatasource(std::unique_ptr<RandomAccessObject> object, const rmm::device_async_resource_ref& mr);

  [[nodiscard]] std::size_t size() const override;
  [[nodiscard]] std::unique_ptr<datasource::buffer> host_read(std::size_t offset, std::size_t size) override;
  std::size_t host_read(std::size_t offset, std::size_t size, std::uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(std::size_t offset,
                                                                   std::size_t size) override;
  std::future<std::size_t> host_read_async(std::size_t offset, std::size_t size, std::uint8_t* dst) override;

  // See class comment: this is the override that actually routes cudf's
  // Parquet reader through a concurrency-capable path for this datasource.
  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] std::unique_ptr<datasource::buffer> device_read(std::size_t offset, std::size_t size,
                                                                rmm::cuda_stream_view stream) override;
  std::size_t device_read(std::size_t offset, std::size_t size, std::uint8_t* dst,
                          rmm::cuda_stream_view stream) override;
  std::future<std::size_t> device_read_async(std::size_t offset, std::size_t size, std::uint8_t* dst,
                                             rmm::cuda_stream_view stream) override;

 private:
  std::unique_ptr<RandomAccessObject> object_;
  rmm::device_async_resource_ref mr_;
};

}  // namespace kernellake
