#pragma once

#include <cudf/io/datasource.hpp>

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
// host_read_async() overrides matter, not just an optimization: cudf's own
// base-class default (undeclared here before -- see git history) wraps the
// synchronous host_read() in `std::async(std::launch::deferred, ...)`,
// which does not run until the returned future is waited on -- so a caller
// that kicks off N "async" reads (cudf's own Parquet reader does this once
// per column chunk it needs) and then waits on each future in turn gets N
// fully *serialized* reads, no overlap at all, each paying this backend's
// own full round-trip latency back-to-back. Confirmed for real: invisible
// against local files or MinIO-on-localhost (near-zero per-request
// latency either way), but a real ~15-20x scan-throughput regression
// against real S3 (0.07-0.17 GB/s vs. 1.5-2.7 GB/s locally, same queries,
// same SF10 data) -- see docs/GPU_OPTIMIZATIONS.md. Overriding these to
// launch on a real thread (std::launch::async) instead is what actually
// lets multiple column-chunk reads overlap their network wait time.
class ObjectStoreDatasource final : public cudf::io::datasource {
 public:
  explicit ObjectStoreDatasource(std::unique_ptr<RandomAccessObject> object);

  [[nodiscard]] std::size_t size() const override;
  [[nodiscard]] std::unique_ptr<datasource::buffer> host_read(std::size_t offset, std::size_t size) override;
  std::size_t host_read(std::size_t offset, std::size_t size, std::uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(std::size_t offset, std::size_t size) override;
  std::future<std::size_t> host_read_async(std::size_t offset, std::size_t size, std::uint8_t* dst) override;

 private:
  std::unique_ptr<RandomAccessObject> object_;
};

}  // namespace kernellake
