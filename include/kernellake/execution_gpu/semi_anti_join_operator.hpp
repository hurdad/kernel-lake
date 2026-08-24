#pragma once

#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <optional>

#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/types/join_type.hpp"

namespace kernellake {

// A LEFT SEMI/LEFT ANTI equi-join (see LogicalJoin/HashJoinNode's own doc
// comments for the full scope: exactly one equality key, produced only by
// sql::rewrite_exists_subqueries()'s EXISTS/NOT EXISTS -> join-step rewrite,
// never directly from SQL syntax). Unlike HashJoinOperator (INNER/LEFT
// OUTER), this operator's output is *only* `left`'s own columns -- a
// semi/anti join never contributes anything from `right` at all (see
// HashJoinNode::output_schema()), so there's no gather-and-concatenate
// step, no null-extension, and no per-row NULL handling beyond whatever
// cudf::filtered_join itself already does.
//
// Built on cudf::filtered_join, not cudf::hash_join: it builds a hash
// *set* from the build/right side's key column alone (never needing to
// gather anything else from that side, unlike cudf::hash_join, which
// keeps the whole build table alive for inner_join()/left_join()'s own
// gather step), via its `semi_join()`/`anti_join()` member functions,
// which each return only the *left* (probe) row indices that do/don't
// have a match -- no right_indices at all, unlike cudf::hash_join's
// inner_join()/left_join().
//
// Same "build once, probe streaming" shape as HashJoinOperator's own
// non-partitioned fast path (open() pulls `right` to exhaustion,
// concatenates it, builds one filtered_join; `left` streams through
// next() batch by batch) -- but *only* that shape: no size-aware
// build-side swap consideration (physical_planner.cpp never swaps a
// semi/anti join's sides at all, same reasoning as LEFT OUTER JOIN --
// see that file's own comment), and no grace-hash partitioned/disk-
// spilling mode yet. A correlated EXISTS/NOT EXISTS subquery's own build
// side is not, in general, guaranteed small (TPC-H Q4's own `lineitem`
// is not) -- this is a real, deliberate scope limitation for this first
// version, not an oversight: see docs/ROADMAP.md's "not yet started"
// list for the follow-up if a real OOM is hit at scale, the same
// iterative pattern HashJoinOperator's own partitioned mode was itself
// added under.
class SemiAntiJoinOperator final : public PhysicalOperator {
 public:
  // `join_type` must be LeftSemi or LeftAnti -- open() throws otherwise
  // (an internal-error case, never reachable from a real bound query,
  // since operator_builder.cpp only ever constructs this operator for
  // those two join types).
  SemiAntiJoinOperator(OperatorId id, std::unique_ptr<PhysicalOperator> left,
                       std::unique_ptr<PhysicalOperator> right, std::size_t left_key_index,
                       std::size_t right_key_index, std::shared_ptr<const Schema> output_schema,
                       JoinType join_type);
  ~SemiAntiJoinOperator() override;

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "SemiAntiJoin"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  // Probes one already-materialized probe-side DeviceBatch against the
  // already-built filtered_join_, returning a matched (LeftSemi) or
  // unmatched (LeftAnti) output batch, or nullopt if this specific batch
  // produced zero matching rows (the caller moves on to the next batch,
  // same "skip empty-result batches" convention HashJoinOperator's own
  // probe_one_batch() already uses).
  std::optional<DeviceBatch> probe_one_batch(const DeviceBatch& left_batch, ExecutionContext& context);

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> left_;   // probe side
  std::unique_ptr<PhysicalOperator> right_;  // build side
  cudf::size_type left_key_index_;
  cudf::size_type right_key_index_;
  std::shared_ptr<const Schema> output_schema_;
  JoinType join_type_;

  // Populated in open(): the whole build side, concatenated. Must outlive
  // filtered_join_ (which only views it, per cudf::filtered_join's own
  // documented lifetime requirement, same as cudf::hash_join's).
  std::unique_ptr<cudf::table> right_table_;
  std::unique_ptr<cudf::filtered_join> filtered_join_;
  // An empty build side means: LeftSemi can never match anything (result
  // is empty, probe side just gets drained); LeftAnti's every probe row
  // trivially "has no match" (the whole probe side passes through
  // unchanged -- no gather needed at all, since output_schema_ already
  // *is* the probe side's own schema).
  bool right_is_empty_ = false;
};

}  // namespace kernellake
