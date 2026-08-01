#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernellake/expression/expression.hpp"
#include "kernellake/types/schema.hpp"

namespace kernellake {

struct NamedExpression {
  ExpressionPtr expr;
  std::string name;
};

class LogicalPlanNode;
using LogicalPlanPtr = std::shared_ptr<LogicalPlanNode>;

// Base of every logical plan node. Logical plans are parser-independent:
// nothing here depends on kernellake::sql. Every node carries its output
// schema, an optional estimated row count (filled in once something
// downstream -- e.g. Parquet metadata inspection -- can supply one), and
// enough node-specific detail (via explain_attributes) to render both the
// human-readable and JSON EXPLAIN formats generically.
class LogicalPlanNode {
 public:
  virtual ~LogicalPlanNode() = default;

  [[nodiscard]] virtual const Schema& output_schema() const = 0;
  [[nodiscard]] virtual std::string_view node_name() const noexcept = 0;
  [[nodiscard]] virtual std::vector<LogicalPlanPtr> children() const = 0;
  [[nodiscard]] virtual std::vector<std::pair<std::string, std::string>> explain_attributes() const {
    return {};
  }

  std::optional<std::uint64_t> estimated_rows;
};

// ---------------------------------------------------------------------------

// A comparison conjunct of the shape `column OP literal` extracted from a
// WHERE clause by the optimizer's predicate-pushdown rule. Parquet metadata
// inspection / row-group and file pruning (see kernellake/storage) consumes
// these to decide what can be skipped using min/max statistics, without
// having to re-walk the filter's expression tree itself.
struct PushablePredicate {
  std::string column_name;
  BinaryOperator op;
  ExpressionPtr literal;  // always a LiteralExpression
};

class LogicalScan final : public LogicalPlanNode {
 public:
  LogicalScan(std::vector<std::string> source_paths, Schema schema)
      : source_paths_(std::move(source_paths)),
        schema_(schema),
        required_columns_(all_field_names(schema_)) {}

  [[nodiscard]] const std::vector<std::string>& source_paths() const noexcept { return source_paths_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "LogicalScan"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override;

  // Defaults to every column in the schema until the optimizer's
  // projection-pushdown rule narrows it to what the plan actually
  // references (see docs/ARCHITECTURE.md for why this narrows the column
  // *list* rather than reindexing the scan's schema/expressions).
  [[nodiscard]] const std::vector<std::string>& required_columns() const noexcept {
    return required_columns_;
  }
  void set_required_columns(std::vector<std::string> columns) { required_columns_ = std::move(columns); }

  [[nodiscard]] const std::vector<PushablePredicate>& pushable_predicates() const noexcept {
    return pushable_predicates_;
  }
  void set_pushable_predicates(std::vector<PushablePredicate> predicates) {
    pushable_predicates_ = std::move(predicates);
  }

 private:
  static std::vector<std::string> all_field_names(const Schema& schema) {
    std::vector<std::string> names;
    names.reserve(schema.field_count());
    for (const Field& field : schema.fields()) names.push_back(field.name);
    return names;
  }

  std::vector<std::string> source_paths_;
  Schema schema_;
  std::vector<std::string> required_columns_;
  std::vector<PushablePredicate> pushable_predicates_;
};

// ---------------------------------------------------------------------------

class LogicalFilter final : public LogicalPlanNode {
 public:
  LogicalFilter(LogicalPlanPtr child, ExpressionPtr predicate)
      : child_(std::move(child)), predicate_(std::move(predicate)) {}

  [[nodiscard]] const LogicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const ExpressionPtr& predicate() const noexcept { return predicate_; }
  [[nodiscard]] const Schema& output_schema() const override { return child_->output_schema(); }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "LogicalFilter"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    return {{"predicate", predicate_->to_string()}};
  }

 private:
  LogicalPlanPtr child_;
  ExpressionPtr predicate_;
};

// ---------------------------------------------------------------------------

class LogicalProjection final : public LogicalPlanNode {
 public:
  LogicalProjection(LogicalPlanPtr child, std::vector<NamedExpression> items)
      : child_(std::move(child)), items_(std::move(items)), schema_(build_schema(items_)) {}

  [[nodiscard]] const LogicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const std::vector<NamedExpression>& items() const noexcept { return items_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "LogicalProjection"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {child_}; }
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

  LogicalPlanPtr child_;
  std::vector<NamedExpression> items_;
  Schema schema_;
};

// ---------------------------------------------------------------------------

class LogicalAggregate final : public LogicalPlanNode {
 public:
  LogicalAggregate(LogicalPlanPtr child, std::vector<NamedExpression> group_by,
                   std::vector<NamedExpression> aggregates)
      : child_(std::move(child)),
        group_by_(std::move(group_by)),
        aggregates_(std::move(aggregates)),
        schema_(build_schema(group_by_, aggregates_)) {}

  [[nodiscard]] const LogicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const std::vector<NamedExpression>& group_by() const noexcept { return group_by_; }
  [[nodiscard]] const std::vector<NamedExpression>& aggregates() const noexcept { return aggregates_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "LogicalAggregate"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {child_}; }
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

  LogicalPlanPtr child_;
  std::vector<NamedExpression> group_by_;
  std::vector<NamedExpression> aggregates_;
  Schema schema_;
};

// ---------------------------------------------------------------------------

class LogicalSort final : public LogicalPlanNode {
 public:
  struct Key {
    ExpressionPtr expr;
    bool ascending;
  };

  LogicalSort(LogicalPlanPtr child, std::vector<Key> keys)
      : child_(std::move(child)), keys_(std::move(keys)) {}

  [[nodiscard]] const LogicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] const std::vector<Key>& keys() const noexcept { return keys_; }
  [[nodiscard]] const Schema& output_schema() const override { return child_->output_schema(); }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "LogicalSort"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override;

 private:
  LogicalPlanPtr child_;
  std::vector<Key> keys_;
};

// ---------------------------------------------------------------------------

class LogicalLimit final : public LogicalPlanNode {
 public:
  LogicalLimit(LogicalPlanPtr child, std::int64_t limit) : child_(std::move(child)), limit_(limit) {}

  [[nodiscard]] const LogicalPlanPtr& child() const noexcept { return child_; }
  [[nodiscard]] std::int64_t limit() const noexcept { return limit_; }
  [[nodiscard]] const Schema& output_schema() const override { return child_->output_schema(); }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "LogicalLimit"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {child_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    return {{"limit", std::to_string(limit_)}};
  }

 private:
  LogicalPlanPtr child_;
  std::int64_t limit_;
};

// ---------------------------------------------------------------------------

// Human-readable tree, e.g.:
//   LogicalAggregate
//       group_by: [region]
//       aggregates: [SUM(amount) AS total_amount]
//       └── LogicalFilter
//           predicate: (event_date >= DATE '2026-01-01')
//           └── LogicalScan
//               source: [/data/sales/*.parquet]
//               columns: [region, amount, event_date]
[[nodiscard]] std::string explain_text(const LogicalPlanNode& root);

// JSON explain output suitable for tooling.
[[nodiscard]] std::string explain_json(const LogicalPlanNode& root);

}  // namespace kernellake
