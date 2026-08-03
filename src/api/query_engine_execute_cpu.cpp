// Provides QueryEngine::execute_cpu() -- always built, in both the `dev`
// and `gpu-dev` presets, since the Apache Arrow Acero backend it wraps
// needs no CUDA at all. Unlike query_engine_execute_gpu.cpp/
// query_engine_execute_stub.cpp (exactly one of which provides
// QueryEngine::execute()), this file always exists alongside whichever of
// those two is selected.
#include <arrow/table.h>
#include <fmt/format.h>

#include <chrono>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/execution_cpu/acero_query_executor.hpp"
#include "kernellake/planner/physical_plan.hpp"

namespace kernellake {

namespace {

const ParquetScanNode* find_parquet_scan(const PhysicalPlanNode& node) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(&node)) {
    return scan;
  }
  for (const PhysicalPlanPtr& child : node.children()) {
    if (const ParquetScanNode* found = find_parquet_scan(*child)) {
      return found;
    }
  }
  return nullptr;
}

}  // namespace

QueryResult QueryEngine::execute_cpu(const PhysicalPlanPtr& physical) const {
  const auto wall_start = std::chrono::steady_clock::now();

  const CpuQueryExecutionResult cpu_result = execute_physical_plan_cpu(physical, store_);

  QueryResult result;
  result.schema = cpu_result.table->schema();
  result.rows_returned = cpu_result.table->num_rows();
  result.cpu_execution_seconds = cpu_result.execution_seconds;

  arrow::TableBatchReader batch_reader(*cpu_result.table);
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    const arrow::Status status = batch_reader.ReadNext(&batch);
    if (!status.ok()) {
      throw ExecutionError(fmt::format(
          "failed to split the CPU execution backend's result table into batches: {}", status.ToString()));
    }
    if (batch == nullptr) {
      break;
    }
    result.batches.push_back(std::move(batch));
  }

  if (const ParquetScanNode* scan = find_parquet_scan(*physical)) {
    result.files_considered = scan->files_considered();
    result.files_scanned = static_cast<std::int64_t>(scan->files_scanned());
    std::int64_t row_groups_considered = 0;
    std::int64_t row_groups_scanned = 0;
    for (const PhysicalFileFragment& fragment : scan->fragments()) {
      row_groups_considered += fragment.total_row_groups;
      row_groups_scanned += static_cast<std::int64_t>(fragment.selected_row_groups.size());
    }
    result.row_groups_considered = row_groups_considered;
    result.row_groups_scanned = row_groups_scanned;
    // parquet_decoding_seconds is not tracked for the CPU backend: Acero's
    // Declaration tree runs as one opaque DeclarationToTable() call with no
    // per-node instrumentation hook wired up (unlike the GPU path's
    // MetricsRegistry-wrapped operator tree) -- documented null, not an
    // invented measurement.
  }

  result.elapsed_wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  return result;
}

}  // namespace kernellake
