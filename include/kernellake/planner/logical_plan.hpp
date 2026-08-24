#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernellake/expression/expression.hpp"
#include "kernellake/types/join_type.hpp"
#include "kernellake/types/partition_column.hpp"
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
  LogicalScan(std::vector<std::string> source_paths, Schema schema,
              std::vector<PartitionColumn> partition_columns = {})
      : source_paths_(std::move(source_paths)),
        schema_(std::move(schema)),
        required_columns_(all_field_names(schema_)),
        partition_columns_(std::move(partition_columns)) {}

  [[nodiscard]] const std::vector<std::string>& source_paths() const noexcept { return source_paths_; }
  // Columns whose values come from each file's location (Hive-style
  // `key=value` directory segments today) rather than being physically
  // present in it -- a suffix of output_schema()'s fields, in the same
  // order. Empty for a plain, non-partitioned source (the common case,
  // and the only one before this was added). See
  // kernellake/io/table_resolution.hpp for how these get discovered.
  [[nodiscard]] const std::vector<PartitionColumn>& partition_columns() const noexcept {
    return partition_columns_;
  }
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
    for (const Field& field : schema.fields()) {
      names.push_back(field.name);
    }
    return names;
  }

  std::vector<std::string> source_paths_;
  Schema schema_;
  std::vector<std::string> required_columns_;
  std::vector<PushablePredicate> pushable_predicates_;
  std::vector<PartitionColumn> partition_columns_;
};

// ---------------------------------------------------------------------------

// A two-table INNER, LEFT OUTER, LEFT SEMI, or LEFT ANTI JOIN on a single
// equality key (see BoundJoin in binder.hpp -- this is the only shape
// currently supported). For INNER/LEFT OUTER, the output schema is the
// plain concatenation of `left`'s and `right`'s output fields, in that
// order: every ColumnExpression above this node in the bound query already
// indexes into that combined row (see binder.cpp's two-schema bind_query()
// overload), so nothing downstream (optimizer column collection, the
// physical planner's remapping, GPU operators) needs to know a join
// happened at all except the physical planner's own HashJoinNode
// conversion. `left_key_index`/`right_key_index` are each into that side's
// *own* schema (pre-concatenation), matching what HashJoinOperator/
// SemiAntiJoinOperator need to extract the key column from each side's
// (already pruned/remapped) physical scan.
//
// For a LEFT OUTER JOIN, every one of `right`'s fields is widened to
// nullable in the combined schema below, regardless of its own source
// schema's declared nullability -- an unmatched left row null-extends every
// right-side column, so a query reading one downstream must see it as
// nullable even if the underlying Parquet column itself is NOT NULL. `left`
// is never widened: LEFT OUTER JOIN always preserves every left row as-is.
//
// LEFT SEMI/LEFT ANTI never appear directly in SQL syntax (see JoinType's
// own comment) -- both are produced only by the EXISTS/NOT EXISTS ->
// join-step rewrite (sql::rewrite_exists_subqueries()). Their output
// schema is `left`'s schema *only* -- zero fields from `right` at all, not
// even the join key -- since a semi/anti join only ever filters `left`'s
// own rows (keeping ones with/without a match), it never actually produces
// any row *from* `right`. This is why a join step after a LEFT SEMI/ANTI
// step in a chain must bind its own column references against the
// *unchanged* running combined-field-count from before that step (see
// binder.cpp's own join-step loop) -- unlike INNER/LEFT OUTER, a semi/anti
// step contributes zero fields to grow that count by.
class LogicalJoin final : public LogicalPlanNode {
 public:
  LogicalJoin(LogicalPlanPtr left, LogicalPlanPtr right, std::size_t left_key_index,
              std::size_t right_key_index, JoinType join_type = JoinType::Inner)
      : left_(std::move(left)),
        right_(std::move(right)),
        left_key_index_(left_key_index),
        right_key_index_(right_key_index),
        join_type_(join_type),
        schema_(build_schema(left_->output_schema(), right_->output_schema(), join_type_)) {}

  [[nodiscard]] const LogicalPlanPtr& left() const noexcept { return left_; }
  [[nodiscard]] const LogicalPlanPtr& right() const noexcept { return right_; }
  [[nodiscard]] std::size_t left_key_index() const noexcept { return left_key_index_; }
  [[nodiscard]] std::size_t right_key_index() const noexcept { return right_key_index_; }
  [[nodiscard]] JoinType join_type() const noexcept { return join_type_; }
  [[nodiscard]] const Schema& output_schema() const override { return schema_; }
  [[nodiscard]] std::string_view node_name() const noexcept override { return "LogicalJoin"; }
  [[nodiscard]] std::vector<LogicalPlanPtr> children() const override { return {left_, right_}; }
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> explain_attributes() const override {
    return {{"type", std::string(kernellake::to_string(join_type_))},
            {"left_key", left_->output_schema().field(left_key_index_).name},
            {"right_key", right_->output_schema().field(right_key_index_).name}};
  }

 private:
  static Schema build_schema(const Schema& left, const Schema& right, JoinType join_type) {
    if (join_type == JoinType::LeftSemi || join_type == JoinType::LeftAnti) {
      return left;  // see this class's own comment: zero fields from right, ever.
    }
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

  LogicalPlanPtr left_;
  LogicalPlanPtr right_;
  std::size_t left_key_index_;
  std::size_t right_key_index_;
  JoinType join_type_;
  Schema schema_;
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
