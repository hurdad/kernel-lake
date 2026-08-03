#pragma once

#include <cudf/io/datasource.hpp>

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
class ObjectStoreDatasource final : public cudf::io::datasource {
 public:
  explicit ObjectStoreDatasource(std::unique_ptr<RandomAccessObject> object);

  [[nodiscard]] std::size_t size() const override;
  [[nodiscard]] std::unique_ptr<datasource::buffer> host_read(std::size_t offset, std::size_t size) override;
  std::size_t host_read(std::size_t offset, std::size_t size, std::uint8_t* dst) override;

 private:
  std::unique_ptr<RandomAccessObject> object_;
};

}  // namespace kernellake
