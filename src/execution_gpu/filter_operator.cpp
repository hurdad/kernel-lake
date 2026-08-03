#include "kernellake/execution_gpu/filter_operator.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

namespace kernellake {

namespace {

// Splits `expr` into its top-level AND-connected conjuncts (recursing
// through nested ANDs), so a LIKE anywhere among them can be pulled out and
// evaluated separately from the rest -- see the class comment.
void flatten_and(const ExpressionPtr& expr, std::vector<ExpressionPtr>& out) {
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(expr.get());
      binary != nullptr && binary->op() == BinaryOperator::And) {
    flatten_and(binary->left(), out);
    flatten_and(binary->right(), out);
    return;
  }
  out.push_back(expr);
}

ExpressionPtr and_together(std::vector<ExpressionPtr> conjuncts) {
  ExpressionPtr result = std::move(conjuncts.front());
  for (std::size_t i = 1; i < conjuncts.size(); ++i) {
    result = std::make_shared<BinaryExpression>(BinaryOperator::And, std::move(result),
                                                std::move(conjuncts[i]), boolean_type(false));
  }
  return result;
}

}  // namespace

FilterOperator::FilterOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child,
                               ExpressionPtr predicate)
    : id_(id), child_(std::move(child)), predicate_(std::move(predicate)) {}

void FilterOperator::open(ExecutionContext& context) {
  child_->open(context);

  std::vector<ExpressionPtr> conjuncts;
  flatten_and(predicate_, conjuncts);

  std::vector<ExpressionPtr> ast_conjuncts;
  for (const ExpressionPtr& conjunct : conjuncts) {
    const auto* like = dynamic_cast<const LikeExpression*>(conjunct.get());
    if (like == nullptr) {
      ast_conjuncts.push_back(conjunct);
      continue;
    }
    CompiledLikeConjunct compiled;
    compiled.pattern = like->pattern();
    compiled.negated = like->negated();
    if (const auto* column = dynamic_cast<const ColumnExpression*>(like->value().get())) {
      compiled.value_column_index = static_cast<cudf::size_type>(column->column_index());
    } else {
      compiled.value_expr = &compiler_.compile(*like->value());
    }
    compiled_like_conjuncts_.push_back(std::move(compiled));
  }
  if (!ast_conjuncts.empty()) {
    compiled_ast_conjuncts_ = &compiler_.compile(*and_together(std::move(ast_conjuncts)));
  }
}

std::unique_ptr<cudf::column> FilterOperator::evaluate_like(const CompiledLikeConjunct& like,
                                                            const cudf::table_view& batch,
                                                            ExecutionContext& context) {
  std::unique_ptr<cudf::column> owned_value_column;
  const cudf::column_view value_view =
      like.value_column_index.has_value()
          ? batch.column(*like.value_column_index)
          : (owned_value_column =
                 cudf::compute_column(batch, *like.value_expr, context.stream, context.memory_resource))
                ->view();

  std::unique_ptr<cudf::column> mask = cudf::strings::like(
      cudf::strings_column_view(value_view), like.pattern, "", context.stream, context.memory_resource);
  if (!like.negated) return mask;
  return cudf::unary_operation(mask->view(), cudf::unary_operator::NOT, context.stream,
                               context.memory_resource);
}

std::optional<DeviceBatch> FilterOperator::next(ExecutionContext& context) {
  while (std::optional<DeviceBatch> batch = child_->next(context)) {
    const cudf::table_view view = batch->view();
    std::unique_ptr<cudf::column> mask =
        compiled_ast_conjuncts_ != nullptr
            ? cudf::compute_column(view, *compiled_ast_conjuncts_, context.stream, context.memory_resource)
            : nullptr;
    for (const CompiledLikeConjunct& like : compiled_like_conjuncts_) {
      std::unique_ptr<cudf::column> like_mask = evaluate_like(like, view, context);
      mask = mask == nullptr
                 ? std::move(like_mask)
                 : cudf::binary_operation(mask->view(), like_mask->view(), cudf::binary_operator::LOGICAL_AND,
                                          cudf::data_type{cudf::type_id::BOOL8}, context.stream,
                                          context.memory_resource);
    }

    std::shared_ptr<const Schema> schema = batch->schema_ptr();
    std::unique_ptr<cudf::table> filtered =
        cudf::apply_boolean_mask(view, mask->view(), context.stream, context.memory_resource);
    if (filtered->num_rows() == 0) continue;
    return DeviceBatch(std::move(filtered), std::move(schema));
  }
  return std::nullopt;
}

void FilterOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
