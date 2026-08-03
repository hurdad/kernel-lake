#include "kernellake/execution_gpu/sort_operator.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/transform.hpp>

#include "kernellake/expression/expression.hpp"

namespace kernellake {

SortOperator::SortOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                           std::vector<LogicalSort::Key> keys)
    : id_(id), child_(std::move(child)), keys_(std::move(keys)) {}

SortOperator::CompiledKey SortOperator::compile_key(const LogicalSort::Key& key) {
  if (const auto* column_ref = dynamic_cast<const ColumnExpression*>(key.expr.get())) {
    return CompiledKey{static_cast<cudf::size_type>(column_ref->column_index()), nullptr, key.ascending};
  }
  return CompiledKey{std::nullopt, &compiler_.compile(*key.expr), key.ascending};
}

void SortOperator::open(ExecutionContext& context) {
  child_->open(context);
  compiled_keys_.reserve(keys_.size());
  for (const LogicalSort::Key& key : keys_) compiled_keys_.push_back(compile_key(key));
}

std::optional<DeviceBatch> SortOperator::next(ExecutionContext& context) {
  if (produced_) return std::nullopt;
  produced_ = true;

  std::vector<DeviceBatch> batches;
  while (std::optional<DeviceBatch> batch = child_->next(context)) {
    batches.push_back(std::move(*batch));
  }
  if (batches.empty()) return std::nullopt;

  const std::shared_ptr<const Schema> schema = batches.front().schema_ptr();
  std::vector<cudf::table_view> views;
  views.reserve(batches.size());
  for (const DeviceBatch& batch : batches) views.push_back(batch.view());
  const std::unique_ptr<cudf::table> combined =
      cudf::concatenate(views, context.stream, context.memory_resource);
  const cudf::table_view combined_view = combined->view();

  // Owns any computed (non-plain-column) key columns so their views stay
  // valid through the sorted_order/gather calls below.
  std::vector<std::unique_ptr<cudf::column>> owned_key_columns;
  std::vector<cudf::column_view> key_views;
  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> null_orders;
  key_views.reserve(compiled_keys_.size());
  orders.reserve(compiled_keys_.size());
  null_orders.reserve(compiled_keys_.size());
  for (const CompiledKey& key : compiled_keys_) {
    if (key.source_column_index.has_value()) {
      key_views.push_back(combined_view.column(*key.source_column_index));
    } else {
      owned_key_columns.push_back(
          cudf::compute_column(combined_view, *key.expr, context.stream, context.memory_resource));
      key_views.push_back(owned_key_columns.back()->view());
    }
    orders.push_back(key.ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    // Matches the common SQL convention (e.g. PostgreSQL's default):
    // NULLs sort last in ASC order, first in DESC order.
    null_orders.push_back(key.ascending ? cudf::null_order::AFTER : cudf::null_order::BEFORE);
  }

  const cudf::table_view key_table_view(key_views);
  const std::unique_ptr<cudf::column> sorted_indices =
      cudf::stable_sorted_order(key_table_view, orders, null_orders, context.stream, context.memory_resource);
  std::unique_ptr<cudf::table> sorted =
      cudf::gather(combined_view, sorted_indices->view(), cudf::out_of_bounds_policy::DONT_CHECK,
                   context.stream, context.memory_resource);
  return DeviceBatch(std::move(sorted), schema);
}

void SortOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
