#pragma once

#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kernellake/execution_gpu/operator.hpp"

namespace kernellake {

// Rough, schema-type-only average row width in bytes (no data access --
// this is a plan-time estimate, called before any data has been read) --
// used only by choose_partition_count() below to size a partition count,
// so erring high is safe (just picks more/smaller partitions than
// strictly necessary); erring low risks a repeat OOM, hence the
// conservative per-type constants here and the extra safety factor in
// choose_partition_count() itself.
[[nodiscard]] std::size_t estimate_row_width_bytes(const Schema& schema);

// Decides HashJoinOperator's partition_count from the build side's
// estimated row count (HashJoinNode::estimated_build_rows() -- a rough,
// pre-filter plan-time estimate, see physical_planner.cpp's
// estimate_row_count()) and a byte budget (query_engine_execute_gpu.cpp's
// build_side_budget_bytes). Returns 1 (no partitioning -- the operator's
// original, unconditional-materialization fast path) whenever the
// estimate is missing or already fits comfortably; otherwise a clamped
// [2, 64] partition count sized so each bucket's share of the estimated
// build side comfortably fits the budget.
[[nodiscard]] std::size_t choose_partition_count(std::optional<std::int64_t> estimated_build_rows,
                                                 const Schema& build_side_schema, std::size_t budget_bytes);

// Two-table INNER equi-join (see HashJoinNode / docs/ARCHITECTURE.md's "Hash
// joins" section for the full scope: exactly one equality key, no
// LEFT/RIGHT/FULL, no 3+-way joins).
//
// `right` is the *build* side, `left` the *probe* side -- put the smaller
// table on the right for best performance; the physical planner already
// does this via its own row-count estimate (see estimate_row_count() in
// physical_planner.cpp), not a cost-based optimizer here.
//
// Two modes, chosen once at construction by `partition_count` (see
// operator_builder.cpp's HashJoinNode case, which computes it from
// HashJoinNode::estimated_build_rows() via choose_partition_count() in the
// .cpp -- never decided reactively mid-query):
//
// - `partition_count == 1` (the common case: build side estimated to fit
//   the GPU memory budget): open() pulls `right` to exhaustion and
//   concatenates it into one `right_table_`/`hash_join_` (cudf::hash_join
//   builds its hash table once, up front, from a single materialized
//   cudf::table_view -- there is no way to build it incrementally
//   batch-by-batch), and `left` is streamed through next() batch-by-batch,
//   each batch probed against that one persistent hash_join_. This is the
//   *exact* code path this operator has always had -- zero added overhead.
//
// - `partition_count > 1` (build side estimated too large to fit): a
//   partitioned ("grace") hash join. Both `left` and `right` are hash-
//   partitioned by the join key (cudf::hash_partition(), deterministic
//   MURMUR3 with a fixed seed, so equal keys land in the same bucket index
//   on both sides) into `partition_count` buckets as they stream in from
//   their child operators, and each bucket is spilled to *disk*, not host
//   RAM (real files under `spill_directory`, one per bucket per side, via
//   arrow::ipc::RecordBatchFileWriter/Reader) -- bounding device memory to
//   roughly one incoming batch at a time during this phase, for both
//   sides, regardless of either side's true total size. This is a real,
//   load-bearing distinction, not a style choice: an earlier version of
//   this design spilled to an in-memory std::vector<arrow::RecordBatch>
//   instead, which bounded GPU memory correctly but not host memory --
//   confirmed for real by a SF1000 TPC-H Q12 run that grew
//   kernellake-server to ~75 GiB RSS before the Linux OOM killer SIGKILL'd
//   it (the *probe* side, e.g. `lineitem` at that scale, can itself be
//   larger than host RAM even after column pruning, regardless of how
//   many buckets the *build* side is split into). next() then processes
//   one bucket at a time: reload that bucket's build rows to GPU (safe --
//   by construction, ~1/partition_count of the total, comfortably under
//   budget), build a small hash_join_ over just it, and stream that
//   bucket's spilled probe rows off disk against it batch by batch,
//   reusing the exact same per-batch probe logic (probe_one_batch()) the
//   fast path above already uses -- so no more than one bucket's build
//   table and one probe batch are ever resident on GPU, and no more than
//   one bucket's probe file is ever being read at once. See
//   docs/ROADMAP.md's "HashJoinOperator streaming gap" entry for the real
//   SF1000 TPC-H OOMs (Q12/Q14/Q19) this was built to fix, and
//   choose_partition_count()'s own comment in the .cpp for how
//   `partition_count` is sized. Spilled files are removed in close() and
//   (in case an exception skips close() -- see InstrumentedOperator's own
//   comment on why that can happen) in the destructor too.
//
// Every output batch's rows are the concatenation of a matching (left_row,
// right_row) pair's columns, left columns first then right -- matching
// LogicalJoin's/HashJoinNode's own schema concatenation convention, which
// is what lets every other operator/expression above a join treat it like
// any other input schema.
class HashJoinOperator final : public PhysicalOperator {
 public:
  // `spill_directory` (only consulted when partition_count > 1) must be a
  // real, disk-backed directory -- see query_engine_execute_gpu.cpp for how
  // this is chosen (prefers storage.cache.directory, a real-disk path
  // this project's NVMe cache already requires; falls back to the system
  // temp directory, with a documented risk, only when no cache directory
  // is configured). Ignored entirely when partition_count == 1.
  HashJoinOperator(OperatorId id, std::unique_ptr<PhysicalOperator> left,
                   std::unique_ptr<PhysicalOperator> right, std::size_t left_key_index,
                   std::size_t right_key_index, std::shared_ptr<const Schema> output_schema,
                   std::size_t partition_count = 1, std::string spill_directory = "");
  ~HashJoinOperator() override;

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "HashJoin"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  // Builds hash_join_ over the already-populated right_table_, setting
  // right_is_empty_ accordingly. Shared by the fast path (open()) and the
  // partitioned path (ensure_partition_built(), once per bucket).
  void build_hash_join(ExecutionContext& context);

  // Probes one already-materialized probe-side DeviceBatch against the
  // currently-built hash_join_/right_table_, returning a matched output
  // batch, or nullopt if this specific batch produced zero matches (the
  // caller moves on to the next batch/bucket, same "skip empty-match
  // batches" convention next() has always used).
  std::optional<DeviceBatch> probe_one_batch(const DeviceBatch& left_batch, ExecutionContext& context);

  // Partitioned mode only: makes sure right_table_/hash_join_/
  // right_is_empty_ reflect build_partition_paths_[current_partition_],
  // reconstructing it by reading that bucket's spilled file back from disk
  // the first time each bucket is visited (idempotent afterward via
  // current_partition_built_).
  void ensure_partition_built(ExecutionContext& context);

  // Best-effort removal of scratch_dir_ (if created) and everything under
  // it. Called from both close() and the destructor -- see this class's
  // own doc comment on why relying on close() alone isn't safe.
  void remove_scratch_dir() noexcept;

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> left_;   // probe side
  std::unique_ptr<PhysicalOperator> right_;  // build side
  cudf::size_type left_key_index_;
  cudf::size_type right_key_index_;
  std::shared_ptr<const Schema> output_schema_;
  std::size_t partition_count_;
  std::string spill_directory_;

  // Populated for whichever build side is currently active: the *whole*
  // build side when partition_count_ == 1, or just the current bucket's
  // rows when partition_count_ > 1. right_table_ must outlive hash_join_
  // (the hash_join object only views it, per cudf::hash_join's own
  // documented lifetime requirement).
  std::unique_ptr<cudf::table> right_table_;
  std::unique_ptr<cudf::hash_join> hash_join_;
  bool right_is_empty_ = false;

  // Captured from the first DeviceBatch each side ever produces (neither
  // PhysicalOperator nor its children expose a schema before the first
  // pull) -- only needed by the partitioned path, to reconstruct DeviceBatch
  // objects via from_arrow_record_batch() out of the spilled Arrow batches
  // read back from disk; stays null (and unused) if a side never produces
  // any batch.
  std::shared_ptr<const Schema> right_schema_;
  std::shared_ptr<const Schema> left_schema_;

  // Partitioned-mode-only state (partition_count_ > 1): both sides are
  // fully drained and hash-partitioned into per-bucket files under
  // scratch_dir_ in open(), then processed bucket-by-bucket in next() --
  // see this class's own doc comment above. *_partition_paths_[i] is only
  // meaningful (and *_partition_nonempty_[i] true) if bucket i actually
  // received at least one row for that side.
  std::filesystem::path scratch_dir_;
  std::vector<std::string> build_partition_paths_;
  std::vector<std::string> probe_partition_paths_;
  std::vector<bool> build_partition_nonempty_;
  std::vector<bool> probe_partition_nonempty_;
  // Open only while streaming the current bucket's probe file in next();
  // reset (closing the file) when moving to the next bucket.
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> probe_reader_;
  std::size_t current_partition_ = 0;
  std::size_t probe_batch_index_ = 0;
  bool current_partition_built_ = false;
};

}  // namespace kernellake
