#include "kernellake/planner/logical_planner.hpp"

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

}  // namespace

LogicalPlanPtr build_logical_plan(const BoundQuery& query, const Schema& source_schema) {
  LogicalPlanPtr plan = std::make_shared<LogicalScan>(query.source_paths, source_schema);

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

    std::vector<NamedExpression> aggregates;
    std::unordered_map<std::size_t, std::size_t> select_index_to_aggregate_position;
    for (std::size_t i = 0; i < query.select_list.size(); ++i) {
      const BoundSelectItem& item = query.select_list[i];
      if (dynamic_cast<const AggregateExpression*>(item.expr.get()) != nullptr) {
        select_index_to_aggregate_position[i] = aggregates.size();
        aggregates.push_back(NamedExpression{item.expr, item.output_name});
      }
    }

    plan = std::make_shared<LogicalAggregate>(plan, std::move(group_by), std::move(aggregates));
    const Schema& aggregate_schema = plan->output_schema();

    // Re-project to the exact column order/names the query requested: the
    // LogicalAggregate above always emits [group_by..., aggregates...]
    // regardless of how the user interleaved them in the SELECT list.
    std::vector<NamedExpression> projection_items;
    projection_items.reserve(query.select_list.size());
    for (std::size_t i = 0; i < query.select_list.size(); ++i) {
      const BoundSelectItem& item = query.select_list[i];
      std::size_t absolute_index = 0;
      if (const auto it = select_index_to_aggregate_position.find(i);
          it != select_index_to_aggregate_position.end()) {
        absolute_index = group_by_positions.size() + it->second;
      } else {
        const auto pos_it = group_by_positions.find(item.expr->to_string());
        if (pos_it == group_by_positions.end()) {
          throw PlanningError("SELECT item '" + item.output_name +
                              "' is neither an aggregate nor a GROUP BY column");
        }
        absolute_index = pos_it->second;
      }
      const Field& field = aggregate_schema.field(absolute_index);
      auto column = std::make_shared<ColumnExpression>(field.name, absolute_index, field.type);
      projection_items.push_back(NamedExpression{column, item.output_name});
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
