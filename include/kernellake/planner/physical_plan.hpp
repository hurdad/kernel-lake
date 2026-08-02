#pragma once

#include <memory>
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
struct PhysicalFileFragment {
  Uri file;
  std::int64_t file_row_count = 0;
  int total_row_groups = 0;
  std::vector<int> selected_row_groups;
  std::vector<int> skipped_row_groups;
  std::vector<std::string> pruning_reasons;
};

class ParquetScanNode final : public PhysicalPlanNode {
 public:
  ParquetScanNode(std::vector<PhysicalFileFragment> fragments, std::vector<std::string> columns,
                  Schema schema, int files_considered)
      : fragments_(std::move(fragments)),
        columns_(std::move(columns)),
        schema_(std::move(schema)),
        files_considered_(files_considered) {}

  [[nodiscard]] const std::vector<PhysicalFileFragment>& fragments() const noexcept { return fragments_; }
  [[nodiscard]] const std::vector<std::string>& columns() const noexcept { return columns_; }
  [[nodiscard]] int files_considered() const noexcept { return files_considered_; }
  [[nodiscard]] std::size_t files_scanned() const noexcept { return fragments_.size(); }

  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "ParquetScan"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override;

 private:
  std::vector<PhysicalFileFragment> fragments_;
  std::vector<std::string> columns_;
  Schema schema_;
  int files_considered_;
};

// A two-table INNER equi-join (see LogicalJoin). `left_key_index`/
// `right_key_index` are into each side's *own already-narrowed* physical
// scan schema (unlike LogicalJoin's, which are into the original,
// pre-pruning logical schema) -- the physical planner translates by name
// when converting a LogicalJoin, since narrowing can shift a column's
// position. Output schema is the plain concatenation of the two children's
// (already narrowed) schemas, in that order, matching exactly what
// HashJoinOperator gathers into its output batch.
class HashJoinNode final : public PhysicalPlanNode {
 public:
  HashJoinNode(PhysicalPlanPtr left, PhysicalPlanPtr right, std::size_t left_key_index,
               std::size_t right_key_index)
      : left_(std::move(left)),
        right_(std::move(right)),
        left_key_index_(left_key_index),
        right_key_index_(right_key_index),
        schema_(build_schema(left_->output_schema(), right_->output_schema())) {}

  [[nodiscard]] const PhysicalPlanPtr& left() const noexcept { return left_; }
  [[nodiscard]] const PhysicalPlanPtr& right() const noexcept { return right_; }
  [[nodiscard]] std::size_t left_key_index() const noexcept { return left_key_index_; }
  [[nodiscard]] std::size_t right_key_index() const noexcept { return right_key_index_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "HashJoin"; }
  [[nodiscard]] std::vector<PhysicalPlanPtr> children() const override { return {left_, right_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    return {{"type", "INNER"},
            {"left_key", left_->output_schema().field(left_key_index_).name},
            {"right_key", right_->output_schema().field(right_key_index_).name}};
  }

 private:
  static Schema build_schema(const Schema& left, const Schema& right) {
    std::vector<Field> fields = left.fields();
    const std::vector<Field>& right_fields = right.fields();
    fields.insert(fields.end(), right_fields.begin(), right_fields.end());
    return Schema(std::move(fields));
  }

  PhysicalPlanPtr left_;
  PhysicalPlanPtr right_;
  std::size_t left_key_index_;
  std::size_t right_key_index_;
  Schema schema_;
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
    for (const NamedExpression& item : items) fields.push_back(Field{item.name, item.expr->result_type()});
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
    for (const NamedExpression& item : group_by) fields.push_back(Field{item.name, item.expr->result_type()});
    for (const NamedExpression& item : aggregates)
      fields.push_back(Field{item.name, item.expr->result_type()});
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
    for (const NamedExpression& item : aggregates)
      fields.push_back(Field{item.name, item.expr->result_type()});
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
      if (i > 0) rendered += ", ";
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
