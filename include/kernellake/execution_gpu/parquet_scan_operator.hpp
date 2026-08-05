#pragma once

#include <cudf/io/parquet.hpp>

#include <memory>
#include <vector>

#include "kernellake/execution_gpu/operator.hpp"
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
//
// `partition_columns` (empty for a plain, non-partitioned scan -- the
// common case, unaffected by anything below) names columns whose values
// come from each fragment's file location (Hive-style `key=value`
// directory segments, see kernellake/io/table_resolution.hpp) rather than
// being physically present in the file, using PhysicalFileFragment's own
// parallel `partition_values`. When non-empty, this operator reads one
// fragment at a time via its own chunked_parquet_reader (rather than one
// reader spanning every fragment, this operator's normal fast path) --
// cudf's chunked reader can legitimately batch rows from *multiple* source
// files into a single returned chunk when they fit within
// pass_read_limit_bytes together, and there is no way to recover, after
// the fact, how many of a chunk's rows came from which file, which a
// per-file constant partition value absolutely needs to know. This trades
// away cross-file pass batching specifically for partitioned scans (a
// single large partition's own file can still stream across multiple
// passes/chunks normally) in exchange for provable correctness, rather
// than guessing at a chunk-to-file boundary that isn't actually exposed.
class ParquetScanOperator final : public PhysicalOperator {
 public:
  ParquetScanOperator(OperatorId id, std::vector<PhysicalFileFragment> fragments,
                      std::vector<std::string> columns, std::shared_ptr<const Schema> schema,
                      ObjectStore& store, std::size_t pass_read_limit_bytes = 0,
                      std::vector<PartitionColumn> partition_columns = {});

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "ParquetScan"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  // Opens fragments_[current_fragment_index_] as a fresh single-file
  // chunked_parquet_reader, used only in the partition_columns_-non-empty
  // path (see class comment).
  void open_current_fragment(ExecutionContext& context);

  OperatorId id_;
  std::vector<PhysicalFileFragment> fragments_;
  std::vector<std::string> columns_;
  std::shared_ptr<const Schema> schema_;
  ObjectStore& store_;
  std::size_t pass_read_limit_bytes_;
  std::vector<PartitionColumn> partition_columns_;
  std::unique_ptr<cudf::io::chunked_parquet_reader> reader_;
  std::size_t current_fragment_index_ = 0;
};

}  // namespace kernellake
