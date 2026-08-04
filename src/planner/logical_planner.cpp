#include "kernellake/planner/logical_planner.hpp"

#include <fmt/format.h>

#include <unordered_map>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

std::vector<LogicalSort::Key> to_sort_keys(const std::vector<BoundOrderByItem>& order_by) {
  std::vector<LogicalSort::Key> keys;
  keys.reserve(order_by.size());
  for (const BoundOrderByItem& item : order_by) {
    keys.push_back(LogicalSort::Key{item.expr, item.ascending});
  }
  return keys;
}

// Recursively rewrites `expr` for the post-aggregation LogicalProjection:
// every subtree matching a GROUP BY key (by to_string() -- the same
// convention `LogicalAggregate`'s alias-resolution already uses) becomes a
// ColumnExpression at that key's position; every AggregateExpression
// subtree becomes a ColumnExpression at its (deduplicated, so `SUM(x) /
// SUM(x)` only computes it once) slot in `aggregates`, registered here the
// first time it's seen; everything else is rebuilt with recursively
// rewritten children. This is what makes a SELECT item that *combines*
// multiple aggregates arithmetically (e.g. TPC-H Q14's `100.00 * SUM(CASE
// WHEN ... THEN ... ELSE 0 END) / SUM(...)`, neither a bare aggregate call
// nor a bare GROUP BY reference) work, not just a SELECT item that *is*
// exactly one or the other -- found while adding Q14. A GROUP BY match is
// checked before recursing, and short-circuits recursion entirely: this is
// essential for `GROUP BY <alias>` resolving to a computed SELECT-list
// expression (e.g. a CASE with no column name of its own -- see
// binder.cpp), whose *own* internals (e.g. a column reference in the
// CASE's condition) are deliberately exempted from the ungrouped-column
// check at bind time specifically because the match happens at this whole
// -subtree level, not by decomposing further.
//
// Registers `expr` as a LogicalAggregate output slot (or reuses the
// existing one if this exact aggregate, by to_string(), was already
// registered -- so `SUM(x) / SUM(x)` only computes it once): whichever
// name is passed in the *first* time a given aggregate is registered is
// the one that sticks for its `aggregates`/output-schema field, since a
// later duplicate reference (from a different SELECT item, or nested
// deeper inside one) just reuses that slot rather than renaming it.
// Callers must always read the name back via `aggregates[position].name`
// (never assume their own `name` argument won) when building a
// ColumnExpression that references this slot -- see the two call sites
// below.
std::size_t register_aggregate(const ExpressionPtr& expr, const std::string& name,
                               std::vector<NamedExpression>& aggregates,
                               std::unordered_map<std::string, std::size_t>& aggregate_positions) {
  const std::string key = expr->to_string();
  if (const auto it = aggregate_positions.find(key); it != aggregate_positions.end()) {
    return it->second;
  }
  const std::size_t position = aggregates.size();
  aggregate_positions[key] = position;
  aggregates.push_back(NamedExpression{expr, name});
  return position;
}

// A ColumnExpression or LiteralExpression reaching the fallthrough at the
// bottom means either a pure literal (returned as-is, needs no rewriting)
// or an ungrouped column reference -- the latter already rejected earlier,
// at bind time, by references_ungrouped_column() (see binder.cpp), so it
// cannot actually occur for a query that bound successfully.
ExpressionPtr rewrite_aggregate_refs(const ExpressionPtr& expr, std::vector<NamedExpression>& aggregates,
                                     std::unordered_map<std::string, std::size_t>& aggregate_positions,
                                     const std::unordered_map<std::string, std::size_t>& group_by_positions,
                                     std::size_t group_by_count) {
  const std::string key = expr->to_string();
  if (const auto group_it = group_by_positions.find(key); group_it != group_by_positions.end()) {
    return std::make_shared<ColumnExpression>(key, group_it->second, expr->result_type());
  }
  if (dynamic_cast<const AggregateExpression*>(expr.get()) != nullptr) {
    const std::size_t position = register_aggregate(
        expr, fmt::format("__kernellake_agg_{}", aggregates.size()), aggregates, aggregate_positions);
    return std::make_shared<ColumnExpression>(aggregates[position].name, group_by_count + position,
                                              expr->result_type());
  }
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(expr.get())) {
    return std::make_shared<BinaryExpression>(
        binary->op(),
        rewrite_aggregate_refs(binary->left(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count),
        rewrite_aggregate_refs(binary->right(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count),
        binary->result_type());
  }
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(expr.get())) {
    return std::make_shared<UnaryExpression>(
        unary->op(),
        rewrite_aggregate_refs(unary->operand(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count),
        unary->result_type());
  }
  if (const auto* cast = dynamic_cast<const CastExpression*>(expr.get())) {
    return std::make_shared<CastExpression>(
        rewrite_aggregate_refs(cast->operand(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count),
        cast->result_type());
  }
  if (const auto* between = dynamic_cast<const BetweenExpression*>(expr.get())) {
    return std::make_shared<BetweenExpression>(
        rewrite_aggregate_refs(between->value(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count),
        rewrite_aggregate_refs(between->lower(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count),
        rewrite_aggregate_refs(between->upper(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count));
  }
  if (const auto* like = dynamic_cast<const LikeExpression*>(expr.get())) {
    return std::make_shared<LikeExpression>(
        rewrite_aggregate_refs(like->value(), aggregates, aggregate_positions, group_by_positions,
                               group_by_count),
        like->pattern(), like->negated());
  }
  if (const auto* case_expr = dynamic_cast<const CaseExpression*>(expr.get())) {
    std::vector<CaseExpression::WhenThen> when_then;
    when_then.reserve(case_expr->when_then().size());
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      when_then.push_back(
          CaseExpression::WhenThen{rewrite_aggregate_refs(branch.condition, aggregates, aggregate_positions,
                                                          group_by_positions, group_by_count),
                                   rewrite_aggregate_refs(branch.result, aggregates, aggregate_positions,
                                                          group_by_positions, group_by_count)});
    }
    ExpressionPtr else_branch =
        case_expr->else_branch() != nullptr
            ? rewrite_aggregate_refs(case_expr->else_branch(), aggregates, aggregate_positions,
                                     group_by_positions, group_by_count)
            : nullptr;
    return std::make_shared<CaseExpression>(std::move(when_then), std::move(else_branch),
                                            case_expr->result_type());
  }
  return expr;
}

}  // namespace

LogicalPlanPtr build_logical_plan(const BoundQuery& query, const Schema& source_schema,
                                  const Schema* right_schema) {
  LogicalPlanPtr plan;
  if (query.join.has_value()) {
    if (right_schema == nullptr) {
      throw PlanningError("unreachable: a JOIN query requires a right_schema");
    }
    auto left_scan = std::make_shared<LogicalScan>(query.join->left_source_paths, source_schema);
    auto right_scan = std::make_shared<LogicalScan>(query.join->right_source_paths, *right_schema);
    plan = std::make_shared<LogicalJoin>(std::move(left_scan), std::move(right_scan),
                                         query.join->left_key_index, query.join->right_key_index);
  } else {
    plan = std::make_shared<LogicalScan>(query.source_paths, source_schema);
  }

  if (query.where != nullptr) {
    plan = std::make_shared<LogicalFilter>(plan, query.where);
  }

  if (!query.order_by.empty() && !query.is_aggregate_query) {
    // Placed before the (aggregate-free) projection: projection only
    // reshapes columns and never changes row count or order, so sorting
    // here is equivalent to sorting the final output, while still allowing
    // ORDER BY to reference a column that is not in the SELECT list.
    plan = std::make_shared<LogicalSort>(plan, to_sort_keys(query.order_by));
  }

  if (query.is_aggregate_query) {
    std::vector<NamedExpression> group_by;
    std::unordered_map<std::string, std::size_t> group_by_positions;
    for (const ExpressionPtr& expr : query.group_by) {
      group_by_positions[expr->to_string()] = group_by.size();
      group_by.push_back(NamedExpression{expr, expr->to_string()});
    }

    // Rewriting every SELECT item up front (before constructing
    // LogicalAggregate) is what lets `aggregates` end up populated with
    // every distinct aggregate the SELECT list references, however deeply
    // nested inside arithmetic -- not just the items that are bare
    // aggregate calls -- see rewrite_aggregate_refs's own comment.
    std::vector<NamedExpression> aggregates;
    std::unordered_map<std::string, std::size_t> aggregate_positions;
    std::vector<ExpressionPtr> rewritten_items;
    rewritten_items.reserve(query.select_list.size());
    for (const BoundSelectItem& item : query.select_list) {
      // A SELECT item that *is* (at the top level) exactly a bare
      // aggregate call registers its LogicalAggregate slot under the
      // query's own alias (item.output_name) rather than a synthetic
      // name, matching this project's existing convention (and what
      // several tests already assert) -- special-cased here rather than
      // inside rewrite_aggregate_refs, which has no access to a
      // caller-supplied name for the generic (possibly deeply-nested)
      // case.
      if (dynamic_cast<const AggregateExpression*>(item.expr.get()) != nullptr) {
        const std::size_t position =
            register_aggregate(item.expr, item.output_name, aggregates, aggregate_positions);
        rewritten_items.push_back(std::make_shared<ColumnExpression>(
            aggregates[position].name, group_by.size() + position, item.expr->result_type()));
        continue;
      }
      rewritten_items.push_back(rewrite_aggregate_refs(item.expr, aggregates, aggregate_positions,
                                                       group_by_positions, group_by.size()));
    }

    plan = std::make_shared<LogicalAggregate>(plan, std::move(group_by), std::move(aggregates));

    // Re-project to the exact column order/names the query requested: the
    // LogicalAggregate above always emits [group_by..., aggregates...]
    // regardless of how the user interleaved them in the SELECT list.
    std::vector<NamedExpression> projection_items;
    projection_items.reserve(query.select_list.size());
    for (std::size_t i = 0; i < query.select_list.size(); ++i) {
      projection_items.push_back(NamedExpression{rewritten_items[i], query.select_list[i].output_name});
    }
    plan = std::make_shared<LogicalProjection>(plan, std::move(projection_items));

    if (!query.order_by.empty()) {
      // Bound against the final output schema (see binder.cpp), which this
      // projection now exactly matches column-for-column.
      plan = std::make_shared<LogicalSort>(plan, to_sort_keys(query.order_by));
    }
  } else {
    std::vector<NamedExpression> projection_items;
    projection_items.reserve(query.select_list.size());
    for (const BoundSelectItem& item : query.select_list) {
      projection_items.push_back(NamedExpression{item.expr, item.output_name});
    }
    plan = std::make_shared<LogicalProjection>(plan, std::move(projection_items));
  }

  if (query.limit.has_value()) {
    plan = std::make_shared<LogicalLimit>(plan, *query.limit);
  }

  return plan;
}

}  // namespace kernellake
