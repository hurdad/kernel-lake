#pragma once

#include <cudf/io/parquet.hpp>

#include <memory>
#include <vector>

#include "kernellake/execution/operator.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Reads the files/row-groups selected by pruning (PhysicalFileFragment,
// from ParquetScanNode) via cudf::io::chunked_parquet_reader, which bounds
// GPU memory per read regardless of dataset size -- "datasets larger than
// GPU memory through iterative processing" from the spec.
//
// `pass_read_limit_bytes` (0 = unlimited) bounds the temporary
// decompression memory used per internal read pass; there is no exact
// "rows per batch" knob on cudf's chunked reader (it is byte-budget based,
// not row-count based), so this is the closest available control and is
// deliberately named for what it actually does rather than implying an
// exact row count.
//
// `store` resolves each fragment's Uri to bytes. Fragments whose scheme is
// "file" use cudf's own local-path source_info constructor directly (no
// extra indirection); any other scheme (s3/gs/gcs/abfs/abfss/az) routes
// through store.open() and an ObjectStoreDatasource wrapper instead -- see
// object_store_datasource.hpp. `store` must outlive this operator.
class ParquetScanOperator final : public PhysicalOperator {
 public:
  ParquetScanOperator(OperatorId id, std::vector<PhysicalFileFragment> fragments,
                      std::vector<std::string> columns, std::shared_ptr<const Schema> schema,
                      ObjectStore& store, std::size_t pass_read_limit_bytes = 0);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "ParquetScan"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  OperatorId id_;
  std::vector<PhysicalFileFragment> fragments_;
  std::vector<std::string> columns_;
  std::shared_ptr<const Schema> schema_;
  ObjectStore& store_;
  std::size_t pass_read_limit_bytes_;
  std::unique_ptr<cudf::io::chunked_parquet_reader> reader_;
};

}  // namespace kernellake
