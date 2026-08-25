#pragma once

#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <cudf/table/table.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "kernellake/execution_gpu/operator.hpp"

namespace kernellake {

// Shared grace-hash-join spill machinery: both HashJoinOperator (INNER/LEFT
// OUTER) and SemiAntiJoinOperator (LEFT SEMI/LEFT ANTI) partition a build
// side too large for the GPU-memory budget by hash-partitioning each side
// into buckets and spilling each bucket to a real disk file (Arrow IPC),
// then reloading and probing one bucket at a time -- see
// hash_join_operator.hpp's class-level comment for the full design
// rationale (device memory bounded to ~one batch at a time; host memory
// bounded the same way, not just GPU memory, confirmed necessary by a real
// SF1000 TPC-H OOM). Extracted here (rather than duplicated) once
// SemiAntiJoinOperator needed the identical machinery for its own
// partitioned mode.

[[nodiscard]] std::unique_ptr<cudf::table> concatenate_device_batches(const std::vector<DeviceBatch>& batches,
                                                                      ExecutionContext& context);

[[nodiscard]] std::shared_ptr<arrow::ipc::RecordBatchFileReader> open_spill_partition_reader(
    const std::string& path);

[[nodiscard]] std::vector<DeviceBatch> read_spill_partition_batches(
    const std::string& path, const std::shared_ptr<const Schema>& schema, ExecutionContext& context);

// Streams `child` to exhaustion, hash-partitioning each incoming batch by
// its `key_index` column into `partition_count` buckets (cudf::hash_partition,
// deterministic MURMUR3 with a fixed seed, so equal keys land in the same
// bucket index when called identically on both sides of a join) and
// spilling each bucket's slice to a disk file under `scratch_dir`
// (`{side_prefix}-{i}.arrow`, Arrow IPC file format). See
// hash_join_operator.cpp's original definition (now here) for the full
// per-batch/per-bucket memory-bounding rationale.
void spill_partitioned_to_disk(PhysicalOperator& child, cudf::size_type key_index,
                               std::size_t partition_count, const std::filesystem::path& scratch_dir,
                               const std::string& side_prefix, std::vector<std::string>& paths_out,
                               std::vector<bool>& nonempty_out, std::shared_ptr<const Schema>& schema_out,
                               ExecutionContext& context);

}  // namespace kernellake
