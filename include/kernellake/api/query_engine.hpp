#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/planner/logical_plan.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/local_object_store.hpp"
#include "kernellake/types/schema.hpp"

namespace arrow {
class Schema;
class RecordBatch;
}  // namespace arrow

namespace kernellake {

// Execution and I/O metrics for one query. Every metric KernelLake cannot
// yet measure (because execution requires GPU/libcudf, not yet built --
// see docs/architecture.md) stays std::nullopt rather than being guessed at,
// per the spec's "documented null value, never an invented measurement"
// rule.
struct QueryResult {
  std::shared_ptr<arrow::Schema> schema;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;

  std::optional<double> elapsed_wall_seconds;
  std::optional<std::int64_t> rows_scanned;
  std::optional<std::int64_t> rows_returned;
  std::optional<std::int64_t> files_considered;
  std::optional<std::int64_t> files_scanned;
  std::optional<std::int64_t> row_groups_considered;
  std::optional<std::int64_t> row_groups_scanned;
  std::optional<std::int64_t> compressed_bytes_read;
  std::optional<std::int64_t> estimated_uncompressed_bytes;
  std::optional<double> metadata_inspection_seconds;
  std::optional<double> parquet_decoding_seconds;
  std::optional<double> gpu_execution_seconds;
  std::optional<double> host_to_device_seconds;
  std::optional<double> device_to_host_seconds;
  std::optional<std::int64_t> peak_gpu_memory_bytes;
};

// The top-level entry point described in the spec: SQL in, either a plan
// (for explain_logical/explain) or a result (for execute) out.
//
// explain_logical() and explain() bind against the real Parquet schema of
// the query's FROM read_parquet(...) source (discovered and inspected via
// LocalObjectStore), so they fully exercise parsing, binding, logical
// planning, optimization, file discovery, and pruning -- everything short
// of actually decoding column data and running GPU operators.
//
// execute() actually runs the query on the GPU when built with
// KERNELLAKE_WITH_CUDA=ON (see query_engine_execute_gpu.cpp): it builds the
// physical-plan operator tree (kernellake/execution/operator_builder.hpp),
// pulls DeviceBatch results through it, and converts each to a host-side
// Arrow RecordBatch. In a CPU-only build (KERNELLAKE_WITH_CUDA=OFF, the
// `dev` preset), execute() throws ExecutionError instead
// (query_engine_execute_stub.cpp) -- KernelLake never substitutes a CPU
// implementation for GPU execution without saying so explicitly.
class QueryEngine {
public:
  explicit QueryEngine(EngineConfig config);

  // Returns the optimized logical plan (i.e. the plan the physical planner
  // would actually receive), not the pre-optimization one -- use this to
  // see what KernelLake decided the query means after its rewrite rules.
  [[nodiscard]] LogicalPlanPtr explain_logical(std::string_view sql) const;

  [[nodiscard]] PhysicalPlanPtr explain(std::string_view sql) const;

  [[nodiscard]] QueryResult execute(std::string_view sql) const;

private:
  [[nodiscard]] LogicalPlanPtr plan_logical(std::string_view sql) const;

  EngineConfig config_;
  mutable LocalObjectStore store_;
};

}  // namespace kernellake
