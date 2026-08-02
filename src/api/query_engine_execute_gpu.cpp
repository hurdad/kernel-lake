// Provides the real QueryEngine::execute() for GPU-enabled
// (KERNELLAKE_WITH_CUDA=ON) builds. Mutually exclusive with
// query_engine_execute_stub.cpp -- see that file's comment.
#include <rmm/mr/per_device_resource.hpp>

#include <chrono>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/execution/arrow_bridge.hpp"
#include "kernellake/execution/cuda_utils.hpp"
#include "kernellake/execution/execution_context.hpp"
#include "kernellake/execution/operator_builder.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/types/arrow_adapter.hpp"

namespace kernellake {

namespace {

const ParquetScanNode* find_parquet_scan(const PhysicalPlanNode& node) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(&node)) return scan;
  for (const PhysicalPlanPtr& child : node.children()) {
    if (const ParquetScanNode* found = find_parquet_scan(*child)) return found;
  }
  return nullptr;
}

QueryId make_query_id() {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  return "q-" + std::to_string(ticks);
}

}  // namespace

QueryResult QueryEngine::execute(std::string_view sql) const {
  const auto wall_start = std::chrono::steady_clock::now();

  double metadata_inspection_seconds = 0.0;
  const LogicalPlanPtr logical = plan_logical(sql, &metadata_inspection_seconds);
  const PhysicalPlanPtr physical = build_physical_plan(logical, store_);

  RmmEnvironment rmm_environment(config_);
  QueryResult result = execute(physical, rmm_environment);
  result.metadata_inspection_seconds = metadata_inspection_seconds;
  // Overwrite the inner call's own elapsed_wall_seconds (which only covers
  // its own scope) with the full convenience-call duration, including
  // parsing/binding/planning above -- matches this overload's documented
  // "plans and executes end to end" contract.
  result.elapsed_wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  return result;
}

QueryResult QueryEngine::execute(const PhysicalPlanPtr& physical, RmmEnvironment& rmm_environment) const {
  const auto wall_start = std::chrono::steady_clock::now();

  const CudaDeviceGuard device_guard(config_.engine.device_id);
  const CudaStream stream;

  MetricsRegistry metrics;
  ExecutionContext context{make_query_id(), config_.engine.device_id,
                           stream.get(),    rmm::mr::get_current_device_resource_ref(),
                           nullptr,         &metrics,
                           nullptr};

  // Half the configured pool ceiling, leaving headroom for filter/
  // projection/aggregation intermediates above the scan itself.
  const std::size_t pass_read_limit_bytes = config_.memory.pool_max_bytes / 2;
  const std::unique_ptr<PhysicalOperator> root =
      build_operator_tree(physical, pass_read_limit_bytes, config_.profiling.nvtx);

  QueryResult result;
  std::int64_t rows_returned = 0;
  double device_to_host_seconds = 0.0;
  const auto gpu_execution_start = std::chrono::steady_clock::now();

  const MemoryUsage usage = rmm_environment.track_query([&] {
    root->open(context);
    while (std::optional<DeviceBatch> batch = root->next(context)) {
      rows_returned += static_cast<std::int64_t>(batch->row_count());
      const auto d2h_start = std::chrono::steady_clock::now();
      result.batches.push_back(to_arrow_record_batch(*batch));
      device_to_host_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - d2h_start).count();
    }
    root->close(context);
  });
  result.gpu_execution_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_execution_start).count();
  result.device_to_host_seconds = device_to_host_seconds;

  result.schema =
      result.batches.empty() ? to_arrow_schema(physical->output_schema()) : result.batches.front()->schema();
  result.rows_returned = rows_returned;
  result.peak_gpu_memory_bytes = usage.peak_bytes;

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
    // ParquetScanOperator's own accumulated time (a leaf operator, so its
    // MetricsRegistry total is its true self time, not inclusive of
    // anything else -- see MetricsRegistry's own doc comment).
    result.parquet_decoding_seconds = metrics.total_seconds(scan->node_name());
    // compressed_bytes_read/rows_scanned and host_to_device_seconds are not
    // tracked yet: the latter has no natural boundary in this architecture
    // today (cudf's chunked Parquet reader decodes host file bytes directly
    // into device memory in one call -- there is no separate "stage data
    // host-side, then copy to device" step to time). Documented null,
    // rather than an invented measurement.
  }

  result.elapsed_wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  return result;
}

}  // namespace kernellake
