#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Converts an optimized PhysicalPlan into a tree of concrete
// PhysicalOperator instances, ready for open()/next()/close().
//
// `store` resolves each ParquetScanNode's fragments to bytes (local, S3,
// GCS, or Azure -- see ParquetScanOperator's own docs); it must outlive the
// returned operator tree.
//
// `pass_read_limit_bytes` (0 = unlimited) is forwarded to
// ParquetScanOperator to bound its per-pass decompression memory; see that
// operator's own docs for why this is a byte budget rather than an exact
// row count.
//
// `build_side_budget_bytes` (0 = unlimited, the default) is forwarded to
// every HashJoinNode's HashJoinOperator, which uses it (together with
// HashJoinNode::estimated_build_rows()) to decide whether its build side
// needs a partitioned/spilling join instead of the plain single-table one
// -- see choose_partition_count() in hash_join_operator.hpp/.cpp for the
// exact decision, and query_engine_execute_gpu.cpp for how this value is
// actually computed (roughly half of RmmEnvironment::query_memory_limit_bytes()).
//
// `spill_directory` (empty = HashJoinOperator falls back to the system
// temp directory -- see that class's own doc comment for the real risk
// this can carry, e.g. a tmpfs-backed /tmp reintroducing the exact host-
// RAM problem partitioning exists to avoid) is forwarded the same way, for
// a partitioned HashJoinOperator to spill buckets to real disk instead of
// host RAM. See query_engine_execute_gpu.cpp for how this is chosen
// (prefers storage.cache.directory, a real-disk path this project's own
// NVMe cache already requires whenever it's configured).
//
// `max_distinct_keys` (0 = HashAggregateOperator's own
// kDefaultMaxDistinctKeys) is forwarded to every HashAggregateNode's
// HashAggregateOperator as its own fixed result-cardinality safety cap --
// see EngineConfig::EngineSection::max_distinct_keys for why this is
// config-driven rather than a compile-time constant.
//
// Every node in the returned tree is wrapped (see operator_builder.cpp's
// InstrumentedOperator) to record its own wall-clock next() time into
// ExecutionContext::metrics when non-null, and to emit an NVTX range per
// next() call when `nvtx_enabled` is true (see EngineConfig's
// ProfilingSection::nvtx) -- this is generic instrumentation, not something
// any individual operator implements itself.
[[nodiscard]] std::unique_ptr<PhysicalOperator> build_operator_tree(
    const PhysicalPlanPtr& plan, ObjectStore& store, std::size_t pass_read_limit_bytes,
    bool nvtx_enabled = false, std::size_t build_side_budget_bytes = 0,
    const std::string& spill_directory = "", std::uint64_t max_distinct_keys = 0);

}  // namespace kernellake
