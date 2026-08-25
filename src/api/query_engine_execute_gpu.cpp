// Provides the real QueryEngine::execute() for GPU-enabled
// (KERNELLAKE_WITH_CUDA=ON) builds. Mutually exclusive with
// query_engine_execute_stub.cpp -- see that file's comment.
#include <fmt/format.h>

#include <chrono>
#include <filesystem>
#include <string>

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
#include "kernellake/unitycatalog/unity_catalog_source_resolver.hpp"

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
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(
      config_.unity_catalog, config_.delta, config_.storage.s3, config_.storage.gcs, config_.storage.azure,
      &unity_catalog_token_cache_);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver, unity_catalog_resolver);
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

  // rmm_environment.device_id(), not config_.engine.device_id: once
  // GpuExecutionCoordinator owns one RmmEnvironment per visible GPU (see
  // docs/MULTI_GPU_SCALING.md's Tier 1), config_.engine.device_id is just
  // the process's originally configured value and no longer identifies
  // which specific device this call's rmm_environment argument is actually
  // bound to -- the CLI's one-shot execute(sql) overload above still
  // constructs its RmmEnvironment straight from config_, so device_id()
  // there is config_.engine.device_id anyway and this is a no-op change for
  // that caller.
  const CudaDeviceGuard device_guard(rmm_environment.device_id());
  const CudaStream stream;

  // Per-query, not the process-wide ambient default -- see
  // RmmEnvironment::make_query_tracker()'s own comment for why this
  // matters once GpuExecutionCoordinator can run more than one query
  // concurrently: rmm::mr::get_current_device_resource_ref() (this
  // context's memory_resource before this change) points at whatever's
  // globally current for the whole process, shared by every concurrently
  // in-flight query alike -- fine for tracking/limiting purposes (the
  // limiter underneath is deliberately shared, see that same comment),
  // but wrong for this query's own peak_gpu_memory_bytes reporting below.
  QueryMemoryTracker memory_tracker = rmm_environment.make_query_tracker();

  MetricsRegistry metrics;
  ExecutionContext context{make_query_id(), rmm_environment.device_id(),
                           stream.get(),    memory_tracker.resource_ref(),
                           nullptr,         &metrics,
                           &memory_tracker};

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
  // 1/8th (not half) of the same actually-enforced ceiling. A partitioned
  // HashJoinOperator's build bucket and pass_read_limit_bytes-sized probe
  // batches are both alive against the *same* ceiling at once during
  // probing -- and pass_read_limit_bytes above is already sized assuming
  // it alone may need up to ~2.4x itself (~0.6x the whole ceiling; the
  // real ratio measured for HashAggregateOperator's own derived columns at
  // SF100, see that divisor's own comment -- HashJoinOperator's gather()
  // producing a concatenated left+right output can plausibly need
  // comparable headroom for the same reason: real width growth beyond the
  // raw scanned/materialized columns). A first attempt at half the
  // ceiling here (leaving too little of that headroom free) OOM'd for
  // real: a real SF1000 TPC-H Q14 run (build side `part`, small enough to
  // only need partition_count=3) failed with "Exceeded memory limit
  // (failed to allocate 1.35 GiB)" -- reproduced twice, including against
  // a freshly-restarted server, ruling out cross-query state. 1/8th costs
  // more, smaller buckets (choose_partition_count() picks a larger
  // partition_count for the same estimated build size), not a free
  // lunch, but erring toward more partitions is cheap; erring toward too
  // few risks exactly this failure (see choose_partition_count()'s own
  // comment in hash_join_operator.hpp/.cpp for how this decides whether
  // -- and how finely -- a join's build side gets partitioned instead of
  // materialized whole).
  const std::size_t build_side_budget_bytes = rmm_environment.query_memory_limit_bytes() / 8;
  // A partitioned HashJoinOperator spills whole buckets to *disk*, not
  // host RAM (see that class's own doc comment for why: even after the
  // build side is bounded, the probe side alone -- e.g. a real SF1000
  // TPC-H `lineitem`, ~6B rows -- can still exceed host RAM outright,
  // confirmed for real by a kernellake-server OOM-kill at ~75 GiB RSS
  // before this existed). storage.cache.directory is reused rather than a
  // new config field: it's already required to be a real, disk-backed
  // directory whenever it's configured at all (see NvmeObjectCache's own
  // docs), so it's already the right kind of place.
  //
  // Deliberately keyed on cache.directory being non-empty alone, *not*
  // cache.enabled too (fixed 2026-08-17 -- the original version here
  // gated on both, coupling two independent concerns: whether S3 reads
  // get cached has nothing to do with whether a real disk-backed
  // directory exists for join spilling). Real, reproducible bug this
  // caused: a worst-case/no-cache SF1000 benchmark run
  // (kernellake_nvme_cache_enabled=false, but the same real 521GB NVMe
  // volume still mounted and still configured as cache.directory) hit Q3's
  // hash join needing to spill, fell through to the system temp directory
  // since cache.enabled was false, and that turned out to be a small
  // tmpfs-backed /tmp on that instance -- "No space left on device"
  // partway through spilling, the exact host-RAM-shaped failure this
  // design already knew to worry about (see the temp-directory fallback's
  // own remaining comment below), just reached through a gap in the
  // condition rather than truly having no directory configured at all.
  // Falls back to the system temp directory only when no cache directory
  // is configured at all -- see HashJoinOperator's own doc comment for
  // the real risk that still carries whenever that fallback is actually
  // hit (a tmpfs-backed /tmp would silently reintroduce the exact
  // host-RAM problem this exists to avoid).
  const std::string spill_directory = !config_.storage.cache.directory.empty()
                                          ? config_.storage.cache.directory
                                          : std::filesystem::temp_directory_path().string();
  const std::unique_ptr<PhysicalOperator> root =
      build_operator_tree(physical, store_, pass_read_limit_bytes, config_.profiling.nvtx,
                          build_side_budget_bytes, spill_directory, config_.engine.max_distinct_keys);

  QueryResult result;
  std::int64_t rows_returned = 0;
  double device_to_host_seconds = 0.0;
  const auto gpu_execution_start = std::chrono::steady_clock::now();

  // No push/pop-stack protection needed here the way the old
  // rmm_environment.track_query(lambda) had: memory_tracker above is this
  // call's own, unshared QueryMemoryTracker instance, not a frame on a
  // structure any other concurrently-running query could also be touching
  // -- if the block below throws, memory_tracker's destructor just runs
  // normally during unwind, same as any other local RAII object, and the
  // exception propagates exactly as it did before.
  root->open(context);
  while (std::optional<DeviceBatch> batch = root->next(context)) {
    rows_returned += static_cast<std::int64_t>(batch->row_count());
    const auto d2h_start = std::chrono::steady_clock::now();
    result.batches.push_back(to_arrow_record_batch(*batch, context.stream, context.memory_resource));
    device_to_host_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - d2h_start).count();
  }
  root->close(context);
  const MemoryUsage usage = memory_tracker.current_usage();
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
    // Not metrics.total_seconds(scan->node_name()) (ParquetScanOperator's
    // plain next()-call self time): since decode/compute overlap (see that
    // operator's class comment), most of its real decode cost happens on a
    // background thread *between* next() calls, not inside them, so that
    // plain self-time now under-reports real decode cost for the common
    // (non-partitioned) scan path. InstrumentedOperator (operator_builder.cpp)
    // separately records the operator's own resource_seconds() -- real
    // cumulative time inside every reader_->read_chunk() call, regardless
    // of which thread/path made it -- under this derived key instead.
    result.parquet_decoding_seconds =
        metrics.total_seconds(fmt::format("{}.resource_seconds", scan->node_name()));
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
