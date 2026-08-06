// Provides the real QueryEngine::execute() for GPU-enabled
// (KERNELLAKE_WITH_CUDA=ON) builds. Mutually exclusive with
// query_engine_execute_stub.cpp -- see that file's comment.
#include <rmm/mr/per_device_resource.hpp>

#include <chrono>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/delta/delta_source_resolver.hpp"
#include "kernellake/execution_gpu/arrow_bridge.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/execution_gpu/execution_context.hpp"
#include "kernellake/execution_gpu/operator_builder.hpp"
#include "kernellake/iceberg/iceberg_source_resolver.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/types/arrow_adapter.hpp"

#include "composite_source_resolver.hpp"

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
  // Real execution re-resolves independently of plan_logical()'s own
  // schema-discovery resolve, same as explain() -- see
  // QueryEngine::explain()'s own comment (query_engine.cpp) for why.
  // Without this, `SELECT ... FROM read_iceberg(...)`/`read_delta(...)`
  // would plan fine but fail at physical-plan time here, since
  // build_physical_plan()'s `extra_resolver` defaults to nullptr.
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver);
  const PhysicalPlanPtr physical = build_physical_plan(logical, store_, &resolver);

  QueryResult result;
  if (config_.engine.backend == "cpu") {
    // Deliberately skips constructing an RmmEnvironment at all -- the CPU
    // backend touches no CUDA state, so a GPU-enabled build requesting the
    // CPU backend shouldn't pay for (or risk failing on) RMM/device setup
    // it doesn't need.
    result = execute_cpu(physical);
  } else {
    RmmEnvironment rmm_environment(config_);
    result = execute(physical, rmm_environment);
  }
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

  // A quarter of the *actually enforced* memory ceiling --
  // rmm_environment.query_memory_limit_bytes() (the exact value this
  // instance's limiting_resource_adaptor was built with), not
  // config_.engine.query_memory_limit_bytes directly (that field's own "0
  // means auto-detect from GPU VRAM" convention would otherwise make this
  // quarter of *zero* whenever auto-detection is in effect) and not a
  // fresh resolve_query_memory_limit_bytes(config_) call either: for a
  // long-lived RmmEnvironment (kernellake-server keeps one for the whole
  // process -- see GpuExecutionCoordinator), auto-detection was resolved
  // once, at that instance's construction, from whatever free VRAM looked
  // like *then*; a fresh call here would read *current* free VRAM instead,
  // which can have drifted since -- sizing this against a value the
  // limiter doesn't actually enforce. Not memory.pool_max_bytes, which is
  // dead config whenever memory.use_async_allocator is true, the default:
  // it only sizes rmm::mr::pool_memory_resource, never constructed in that
  // case; see rmm_environment.cpp's build_base_resource()). Using
  // pool_max_bytes here was a real, distinct bug from the divisor below --
  // silent unless a caller happens to set both fields to the same value.
  //
  // A real SF100 run (600M-row lineitem, TPC-H-derived Q1: GROUP BY
  // returnflag/linestatus over an almost-unfiltered scan, materializing two
  // extra derived DOUBLE columns per pass for
  // SUM(extendedprice*(1-discount)) and SUM(extendedprice*(1-discount)*
  // (1+tax)) on top of the 7 scanned columns) hit a genuine RMM OOM under
  // the previous `/ 2` divisor: measured peak need was a consistent ~1.196x
  // the configured ceiling at two different ceiling sizes (6.04 GiB ->
  // needed +1.18 GiB; 3 GiB -> needed +0.59 GiB) -- i.e. ~2.4x
  // pass_read_limit_bytes, not the ~2x of headroom `/ 2` provides. `/ 4`
  // leaves about 40% margin above that measured ratio, for the retained
  // original columns, the derived projection columns, and hash-aggregate
  // working state all live at once within one pass -- not just the scan
  // itself.
  const std::size_t pass_read_limit_bytes = rmm_environment.query_memory_limit_bytes() / 4;
  const std::unique_ptr<PhysicalOperator> root =
      build_operator_tree(physical, store_, pass_read_limit_bytes, config_.profiling.nvtx);

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
