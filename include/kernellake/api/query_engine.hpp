#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/planner/logical_plan.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/object_store_registry.hpp"
#include "kernellake/types/schema.hpp"

namespace arrow {
class Schema;
class RecordBatch;
}  // namespace arrow

namespace kernellake {

class RmmEnvironment;

// Execution and I/O metrics for one query. Every metric KernelLake cannot
// yet measure (because execution requires GPU/libcudf, not yet built --
// see docs/ARCHITECTURE.md) stays std::nullopt rather than being guessed at,
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
  // Populated only by the CPU (Acero) execution backend -- kept as its own
  // field rather than overloading gpu_execution_seconds, since the two
  // backends measure genuinely different work (a single Acero
  // DeclarationToTable() call vs. GPU operator pull-loop + device-to-host
  // transfer) and conflating them under one name would misrepresent which
  // backend actually ran.
  std::optional<double> cpu_execution_seconds;
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

  // Convenience entry point for one-shot callers (the CLI): plans and
  // executes `sql` end to end, building and tearing down its own
  // RmmEnvironment for the duration of this call. Fine for a one-query-
  // per-process model; a long-lived caller (e.g. a server handling many
  // requests) should prefer explain() + the execute(physical, rmm_environment)
  // overload below instead, reusing one RmmEnvironment across requests --
  // see docs/ARCHITECTURE.md's Concurrency notes for why rebuilding the RMM
  // pool per request is both wrong under concurrency and wasteful even
  // single-threaded.
  [[nodiscard]] QueryResult execute(std::string_view sql) const;

  // Runs an already-built physical plan's GPU execution against an
  // *externally owned* RmmEnvironment (constructed and torn down by the
  // caller, once, not per call). `result.metadata_inspection_seconds` stays
  // nullopt here -- planning (where that time is actually spent) already
  // happened before this call, in whatever produced `physical` (e.g.
  // explain()); a caller that wants it should time its own explain() call.
  [[nodiscard]] QueryResult execute(const PhysicalPlanPtr& physical, RmmEnvironment& rmm_environment) const;

  // Runs an already-built physical plan on the Apache Arrow Acero CPU
  // execution backend (see docs/ARCHITECTURE.md's CPU backend section).
  // Always available, in both the `dev` and `gpu-dev` presets -- unlike the
  // execute(physical, rmm_environment) overload above, this needs no CUDA
  // and takes no external resource, since Acero owns its own thread pool
  // internally. Throws PlanningError/ExecutionError for physical plan nodes
  // this backend doesn't yet support (e.g. HashJoin). Like the GPU overload,
  // leaves metadata_inspection_seconds null (planning already happened
  // before this call).
  [[nodiscard]] QueryResult execute_cpu(const PhysicalPlanPtr& physical) const;

 private:
  // `metadata_inspection_seconds_out`, when non-null, accumulates the time
  // spent discovering/inspecting each FROM source's Parquet metadata (the
  // JOIN case inspects two sources, hence "accumulates" rather than
  // "assigns"). Left null by explain_logical()/explain(), which don't
  // return a QueryResult to put it in.
  [[nodiscard]] LogicalPlanPtr plan_logical(std::string_view sql,
                                            double* metadata_inspection_seconds_out = nullptr) const;

  EngineConfig config_;
  // Declared after config_ (member init order follows declaration order):
  // ObjectStoreRegistry keeps a reference to config_.storage, valid for
  // QueryEngine's whole lifetime since both are members of the same object.
  mutable ObjectStoreRegistry store_;
};

}  // namespace kernellake
