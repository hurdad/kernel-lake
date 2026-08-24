#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernellake/planner/logical_plan.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Physical plan nodes describe *what will actually run*: which concrete
// files and row groups a scan will touch after pruning, and the operator
// pipeline above it. They carry no execution behavior themselves -- that is
// PhysicalOperator's job (kernellake/execution/operator.hpp), whose concrete
// GPU implementations require libcudf/RMM and are built separately (see
// docs/ARCHITECTURE.md) once that dependency is available.
class PhysicalPlanNode;
using PhysicalPlanPtr = std::shared_ptr<PhysicalPlanNode>;

class PhysicalPlanNode {
 public:
  virtual ~PhysicalPlanNode() = default;

  [[nodiscard]] virtual const Schema& output_schema() const = 0;
  [[nodiscard]] virtual std::string_view node_name() const noexcept = 0;
  [[nodiscard]] virtual std::vector<PhysicalPlanPtr> children() const = 0;
  [[nodiscard]] virtual std::vector<std::pair<std::string, std::string>> explain_attributes() const {
    return {};
  }
};

// One file's pruning outcome, ready for the scan operator to act on.
// `partition_values` (empty for a plain, non-partitioned source) is parallel
// to ParquetScanNode::partition_columns() -- constant values this fragment's
// every row gets for those columns, since they're derived from the file's
// location and never physically present in it. See
// kernellake/io/table_resolution.hpp.
struct PhysicalFileFragment {
  Uri file;
  std::int64_t file_row_count = 0;
  int total_row_groups = 0;
  std::vector<int> selected_row_groups;
  std::vector<int> skipped_row_groups;
  std::vector<std::string> pruning_reasons;
  std::vector<LiteralStorage> partition_values;
};

class ParquetScanNode final : public PhysicalPlanNode {
 public:
  ParquetScanNode(std::vector<PhysicalFileFragment> fragments, std::vector<std::string> columns,
                  Schema schema, int files_considered, std::vector<PartitionColumn> partition_columns = {},
                  std::vector<std::optional<std::size_t>> original_column_map = {},
                  std::shared_ptr<ObjectStore> owned_store = nullptr)
      : fragments_(std::move(fragments)),
        columns_(std::move(columns)),
        schema_(std::move(schema)),
        files_considered_(files_considered),
        partition_columns_(std::move(partition_columns)),
        original_column_map_(std::move(original_column_map)),
        owned_store_(std::move(owned_store)) {}

  [[nodiscard]] const std::vector<PhysicalFileFragment>& fragments() const noexcept { return fragments_; }
  // Physical columns to actually read from each fragment's Parquet file --
  // never includes a partition column (see partition_columns() below),
  // since those don't exist in the file itself and cudf's/Arrow's Parquet
  // readers would fail to find them.
  [[nodiscard]] const std::vector<std::string>& columns() const noexcept { return columns_; }
  // Columns whose values must be materialized as a constant per fragment
  // (from PhysicalFileFragment::partition_values) rather than read from the
  // file -- a suffix of output_schema()'s fields not covered by columns().
  [[nodiscard]] const std::vector<PartitionColumn>& partition_columns() const noexcept {
    return partition_columns_;
  }
  [[nodiscard]] int files_considered() const noexcept { return files_considered_; }
  [[nodiscard]] std::size_t files_scanned() const noexcept { return fragments_.size(); }
  // Maps this scan's *original* (pre-narrowing) LogicalScan column index to
  // its position in this node's own (possibly narrowed) output_schema(), or
  // nullopt if that original column was pruned away. A ColumnExpression
  // sitting above this scan still carries the original index the binder
  // resolved it to (see physical_planner.cpp's remap_columns()); this map is
  // what translates that back to a real position without going through
  // Schema::find_field()'s first-name-match lookup, which is ambiguous once
  // a JOIN puts two same-named columns from different sides into one
  // combined schema (see HashJoinNode's own identical field below).
  [[nodiscard]] const std::vector<std::optional<std::size_t>>& original_column_map() const noexcept {
    return original_column_map_;
  }
  // Non-null only when this scan's files must be read through different
  // credentials than the query's own default ObjectStore -- see
  // ResolvedTable::owned_store's own doc comment for why and how this gets
  // populated. Execution (acero_query_executor.cpp's translate(),
  // operator_builder.cpp's build()) must check this per scan node instead
  // of assuming the one ObjectStore threaded through the rest of the
  // physical plan tree is always the right one to read this scan's files
  // with.
  [[nodiscard]] ObjectStore* owned_store() const noexcept { return owned_store_.get(); }

  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "ParquetScan"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override;

 private:
  std::vector<PhysicalFileFragment> fragments_;
  std::vector<std::string> columns_;
  Schema schema_;
  int files_considered_;
  std::vector<PartitionColumn> partition_columns_;
  std::vector<std::optional<std::size_t>> original_column_map_;
  std::shared_ptr<ObjectStore> owned_store_;
};

// A two-table INNER or LEFT OUTER equi-join (see LogicalJoin).
// `left_key_index`/`right_key_index` are into each side's *own already-
// narrowed* physical scan schema (unlike LogicalJoin's, which are into the
// original, pre-pruning logical schema) -- the physical planner translates
// via each side's own original_column_map() when converting a LogicalJoin,
// since narrowing can shift a column's position. Output schema is the
// plain concatenation of the two children's (already narrowed) schemas, in
// that order (right widened to nullable for LEFT OUTER, same as
// LogicalJoin's own build_schema()), matching exactly what
// HashJoinOperator gathers into its output batch.
class HashJoinNode final : public PhysicalPlanNode {
 public:
  HashJoinNode(PhysicalPlanPtr left, PhysicalPlanPtr right, std::size_t left_key_index,
               std::size_t right_key_index, std::vector<std::optional<std::size_t>> original_column_map = {},
               std::optional<std::int64_t> estimated_build_rows = std::nullopt,
               JoinType join_type = JoinType::Inner)
      : left_(std::move(left)),
        right_(std::move(right)),
        left_key_index_(left_key_index),
        right_key_index_(right_key_index),
        join_type_(join_type),
        schema_(build_schema(left_->output_schema(), right_->output_schema(), join_type_)),
        original_column_map_(std::move(original_column_map)),
        estimated_build_rows_(estimated_build_rows) {}

  [[nodiscard]] const PhysicalPlanPtr& left() const noexcept { return left_; }
  [[nodiscard]] const PhysicalPlanPtr& right() const noexcept { return right_; }
  [[nodiscard]] std::size_t left_key_index() const noexcept { return left_key_index_; }
  [[nodiscard]] std::size_t right_key_index() const noexcept { return right_key_index_; }
  [[nodiscard]] JoinType join_type() const noexcept { return join_type_; }
  // Rough, pre-filter row-count estimate of the *build* (right) side --
  // exactly the value physical_planner.cpp's own estimate_row_count()
  // already computes to decide which side to build on (see that
  // function's doc comment for the estimate's caveats), just persisted
  // here instead of only used transiently for the swap decision. nullopt
  // when no estimate was available. Used by operator_builder.cpp to size
  // HashJoinOperator's partition count -- see that file and
  // hash_join_operator.cpp's choose_partition_count().
  [[nodiscard]] std::optional<std::int64_t> estimated_build_rows() const noexcept {
    return estimated_build_rows_;
  }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "HashJoin"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {left_, right_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    return {{"type", std::string(kernellake::to_string(join_type_))},
            {"left_key", left_->output_schema().field(left_key_index_).name},
            {"right_key", right_->output_schema().field(right_key_index_).name}};
  }
  // Maps the *original* combined pre-join logical schema's column index
  // (left's original fields first, then right's -- the domain
  // LogicalJoin's ON condition and everything above this node was resolved
  // against, see LogicalJoin's own doc comment) to this node's own actual
  // output_schema() position, or nullopt if pruned. Built by combining
  // left/right's own original_column_map()s (see the physical planner's
  // JOIN conversion) -- see ParquetScanNode::original_column_map()'s own
  // comment for why this exists instead of Schema::find_field().
  [[nodiscard]] const std::vector<std::optional<std::size_t>>& original_column_map() const noexcept {
    return original_column_map_;
  }

 private:
  static Schema build_schema(const Schema& left, const Schema& right, JoinType join_type) {
    std::vector<Field> fields = left.fields();
    const std::vector<Field>& right_fields = right.fields();
    if (join_type == JoinType::LeftOuter) {
      fields.reserve(fields.size() + right_fields.size());
      for (const Field& field : right_fields) {
        Field widened = field;
        widened.type.nullable = true;
        fields.push_back(std::move(widened));
      }
    } else {
      fields.insert(fields.end(), right_fields.begin(), right_fields.end());
    }
    return Schema(std::move(fields));
  }

  PhysicalPlanPtr left_;
  PhysicalPlanPtr right_;
  std::size_t left_key_index_;
  std::size_t right_key_index_;
  JoinType join_type_;
  Schema schema_;
  std::vector<std::optional<std::size_t>> original_column_map_;
  std::optional<std::int64_t> estimated_build_rows_;
};

class FilterNode final : public PhysicalPlanNode {
 public:
  FilterNode(PhysicalPlanPtr child, ExpressionPtr predicate)
      : child_(std::move(child)), predicate_(std::move(predicate)) {}

  [[nodiscard]] const PhysicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const ExpressionPtr& predicate() const noexcept { return predicate_; }
  [[nodiscard]] const Schema& output_schema() const override { return child_->output_schema(); }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "Filter"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    return {{"predicate", predicate_->to_string()}};
  }

 private:
  PhysicalPlanPtr child_;
  ExpressionPtr predicate_;
};

class ProjectionNode final : public PhysicalPlanNode {
 public:
  ProjectionNode(PhysicalPlanPtr child, std::vector<NamedExpression> items)
      : child_(std::move(child)), items_(std::move(items)), schema_(build_schema(items_)) {}

  [[nodiscard]] const PhysicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const std::vector<NamedExpression>& items() const noexcept { return items_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "Projection"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override;

 private:
  static Schema build_schema(const std::vector<NamedExpression>& items) {
    std::vector<Field> fields;
    fields.reserve(items.size());
    for (const NamedExpression& item : items) {
      fields.push_back(Field{item.name, item.expr->result_type()});
    }
    return Schema(std::move(fields));
  }
  PhysicalPlanPtr child_;
  std::vector<NamedExpression> items_;
  Schema schema_;
};

class HashAggregateNode final : public PhysicalPlanNode {
 public:
  HashAggregateNode(PhysicalPlanPtr child, std::vector<NamedExpression> group_by,
                    std::vector<NamedExpression> aggregates)
      : child_(std::move(child)),
        group_by_(std::move(group_by)),
        aggregates_(std::move(aggregates)),
        schema_(build_schema(group_by_, aggregates_)) {}

  [[nodiscard]] const PhysicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const std::vector<NamedExpression>& group_by() const noexcept { return group_by_; }
  [[nodiscard]] const std::vector<NamedExpression>& aggregates() const noexcept { return aggregates_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "HashAggregate"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override;

 private:
  static Schema build_schema(const std::vector<NamedExpression>& group_by,
                             const std::vector<NamedExpression>& aggregates) {
    std::vector<Field> fields;
    fields.reserve(group_by.size() + aggregates.size());
    for (const NamedExpression& item : group_by) {
      fields.push_back(Field{item.name, item.expr->result_type()});
    }
    for (const NamedExpression& item : aggregates) {
      fields.push_back(Field{item.name, item.expr->result_type()});
    }
    return Schema(std::move(fields));
  }
  PhysicalPlanPtr child_;
  std::vector<NamedExpression> group_by_;
  std::vector<NamedExpression> aggregates_;
  Schema schema_;
};

// The no-GROUP-BY case: a single output row (or zero for an empty input),
// distinct from HashAggregate per the physical operator list in
// docs/ARCHITECTURE.md.
class ScalarAggregateNode final : public PhysicalPlanNode {
 public:
  ScalarAggregateNode(PhysicalPlanPtr child, std::vector<NamedExpression> aggregates)
      : child_(std::move(child)), aggregates_(std::move(aggregates)), schema_(build_schema(aggregates_)) {}

  [[nodiscard]] const PhysicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const std::vector<NamedExpression>& aggregates() const noexcept { return aggregates_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "ScalarAggregate"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override;

 private:
  static Schema build_schema(const std::vector<NamedExpression>& aggregates) {
    std::vector<Field> fields;
    fields.reserve(aggregates.size());
    for (const NamedExpression& item : aggregates) {
      fields.push_back(Field{item.name, item.expr->result_type()});
    }
    return Schema(std::move(fields));
  }
  PhysicalPlanPtr child_;
  std::vector<NamedExpression> aggregates_;
  Schema schema_;
};

class SortNode final : public PhysicalPlanNode {
 public:
  SortNode(PhysicalPlanPtr child, std::vector<LogicalSort::Key> keys)
      : child_(std::move(child)), keys_(std::move(keys)) {}

  [[nodiscard]] const PhysicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const std::vector<LogicalSort::Key>& keys() const noexcept { return keys_; }
  [[nodiscard]] const Schema& output_schema() const override { return child_->output_schema(); }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "Sort"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    std::string rendered;
    for (std::size_t i = 0; i < keys_.size(); ++i) {
      if (i > 0) {
        rendered += ", ";
      }
      rendered += keys_[i].expr->to_string() + (keys_[i].ascending ? " ASC" : " DESC");
    }
    return {{"order_by", rendered}};
  }

 private:
  PhysicalPlanPtr child_;
  std::vector<LogicalSort::Key> keys_;
};

class LimitNode final : public PhysicalPlanNode {
 public:
  LimitNode(PhysicalPlanPtr child, std::int64_t limit) : child_(std::move(child)), limit_(limit) {}

  [[nodiscard]] const PhysicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] std::int64_t limit() const noexcept { return limit_; }
  [[nodiscard]] const Schema& output_schema() const override { return child_->output_schema(); }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "Limit"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    return {{"limit", std::to_string(limit_)}};
  }

 private:
  PhysicalPlanPtr child_;
  std::int64_t limit_;
};

// Terminal node: converts the final device-resident batch stream to Arrow
// RecordBatches. Always the physical plan's root.
class ArrowResultNode final : public PhysicalPlanNode {
 public:
  explicit ArrowResultNode(PhysicalPlanPtr child) : child_(std::move(child)) {}

  [[nodiscard]] const PhysicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const Schema& output_schema() const override { return child_->output_schema(); }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "ArrowResult"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {child_}; }

 private:
  PhysicalPlanPtr child_;
};

[[nodiscard]] std::string explain_text(const PhysicalPlanNode& root);
[[nodiscard]] std::string explain_json(const PhysicalPlanNode& root);

}  // namespace kernellake
