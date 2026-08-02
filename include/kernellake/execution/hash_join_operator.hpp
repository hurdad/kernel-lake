#pragma once

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <optional>

#include "kernellake/execution/operator.hpp"

namespace kernellake {

// Two-table INNER equi-join (see HashJoinNode / docs/ARCHITECTURE.md's "Hash
// joins" section for the full scope: exactly one equality key, no
// LEFT/RIGHT/FULL, no 3+-way joins).
//
// `right` is the *build* side: pulled to exhaustion and concatenated into a
// single table in open() (the same "consume-then-produce" shape
// SortOperator uses, and for the same reason -- cudf::hash_join builds its
// hash table once, up front, from a single materialized cudf::table_view;
// there is no way to build it incrementally batch-by-batch). `left` is the
// *probe* side: streamed through next() batch-by-batch, each batch probed
// against the one persistent cudf::hash_join object built in open(). Put
// the smaller table on the right for best performance -- there is no
// cost-based optimizer here to choose that automatically.
//
// Every output batch's rows are the concatenation of a matching (left_row,
// right_row) pair's columns, left columns first then right -- matching
// LogicalJoin's/HashJoinNode's own schema concatenation convention, which
// is what lets every other operator/expression above a join treat it like
// any other input schema.
class HashJoinOperator final : public PhysicalOperator {
 public:
  HashJoinOperator(OperatorId id, std::unique_ptr<PhysicalOperator> left,
                   std::unique_ptr<PhysicalOperator> right, std::size_t left_key_index,
                   std::size_t right_key_index, std::shared_ptr<const Schema> output_schema);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "HashJoin"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  OperatorId id_;
  std::unique_ptr<PhysicalOperator> left_;   // probe side
  std::unique_ptr<PhysicalOperator> right_;  // build side
  cudf::size_type left_key_index_;
  cudf::size_type right_key_index_;
  std::shared_ptr<const Schema> output_schema_;

  // Populated once in open(); right_table_ must outlive hash_join_ (the
  // hash_join object only views it, per cudf::hash_join's own
  // documented lifetime requirement).
  std::unique_ptr<cudf::table> right_table_;
  std::unique_ptr<cudf::hash_join> hash_join_;
  bool right_is_empty_ = false;
};

}  // namespace kernellake
