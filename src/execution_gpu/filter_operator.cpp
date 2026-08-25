#include "kernellake/execution_gpu/filter_operator.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include "kernellake/execution_gpu/cudf_adapter.hpp"

namespace kernellake {

namespace {

// Splits `expr` into its top-level OR-connected leaves (recursing through
// nested ORs) -- the IN-desugared mirror of flatten_and() below, since
// bind_node(AstIn&, bool) builds a left-associated OR-chain of equality
// comparisons (see binder.cpp).
void flatten_or(const ExpressionPtr& expr, std::vector<ExpressionPtr>& out) {
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(expr.get());
      binary != nullptr && binary->op() == BinaryOperator::Or) {
    flatten_or(binary->left(), out);
    flatten_or(binary->right(), out);
    return;
  }
  out.push_back(expr);
}

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

std::optional<FilterOperator::CompiledSubstringInConjunct> FilterOperator::try_compile_substring_in(
    const ExpressionPtr& conjunct, ExecutionContext& context) {
  bool negated = false;
  ExpressionPtr or_root = conjunct;
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(conjunct.get());
      unary != nullptr && unary->op() == UnaryOperator::Not) {
    negated = true;
    or_root = unary->operand();
  }

  std::vector<ExpressionPtr> leaves;
  flatten_or(or_root, leaves);
  // A single-element "OR-chain" (IN with exactly one value) is still this
  // shape -- flatten_or() on a bare non-OR expr just returns it as the one
  // leaf, same as flatten_and() does for a lone conjunct.

  const SubstringExpression* shared_substring = nullptr;
  std::vector<std::string> literals;
  literals.reserve(leaves.size());
  for (const ExpressionPtr& leaf : leaves) {
    const auto* eq = dynamic_cast<const BinaryExpression*>(leaf.get());
    if (eq == nullptr || eq->op() != BinaryOperator::Equal) {
      return std::nullopt;
    }
    const auto* substring = dynamic_cast<const SubstringExpression*>(eq->left().get());
    const auto* literal = dynamic_cast<const LiteralExpression*>(eq->right().get());
    if (substring == nullptr || literal == nullptr || literal->is_null() ||
        literal->result_type().id != TypeId::String) {
      return std::nullopt;
    }
    if (shared_substring == nullptr) {
      shared_substring = substring;
    } else if (substring->structural_key() != shared_substring->structural_key()) {
      // A different underlying SUBSTRING call (or a different start/
      // length) -- not the single-shared-substring shape this fast path
      // handles; fall back to the ordinary AST path (which will itself
      // fail to compile the SUBSTRING leaf, the correct "not yet
      // supported here" outcome for a shape this codebase doesn't claim).
      return std::nullopt;
    }
    literals.push_back(std::get<std::string>(literal->value()));
  }
  if (shared_substring == nullptr) {
    return std::nullopt;
  }

  CompiledSubstringInConjunct compiled;
  compiled.start_zero_based = shared_substring->start_zero_based();
  compiled.length = shared_substring->length();
  compiled.literals = std::move(literals);
  compiled.negated = negated;
  if (const auto* column = dynamic_cast<const ColumnExpression*>(shared_substring->operand().get())) {
    compiled.operand_column_index = static_cast<cudf::size_type>(column->column_index());
  } else {
    compiled.operand_expr = &compiler_.compile(*shared_substring->operand(), context);
  }
  return compiled;
}

void FilterOperator::open(ExecutionContext& context) {
  child_->open(context);

  std::vector<ExpressionPtr> conjuncts;
  flatten_and(predicate_, conjuncts);

  std::vector<ExpressionPtr> ast_conjuncts;
  for (const ExpressionPtr& conjunct : conjuncts) {
    if (const auto* like = dynamic_cast<const LikeExpression*>(conjunct.get())) {
      CompiledLikeConjunct compiled;
      compiled.pattern = like->pattern();
      compiled.negated = like->negated();
      if (const auto* column = dynamic_cast<const ColumnExpression*>(like->value().get())) {
        compiled.value_column_index = static_cast<cudf::size_type>(column->column_index());
      } else {
        compiled.value_expr = &compiler_.compile(*like->value(), context);
      }
      compiled_like_conjuncts_.push_back(std::move(compiled));
      continue;
    }
    if (std::optional<CompiledSubstringInConjunct> substring_in =
            try_compile_substring_in(conjunct, context)) {
      compiled_substring_in_conjuncts_.push_back(std::move(*substring_in));
      continue;
    }
    ast_conjuncts.push_back(conjunct);
  }
  if (!ast_conjuncts.empty()) {
    compiled_ast_conjuncts_ = &compiler_.compile(*and_together(std::move(ast_conjuncts)), context);
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

// Mirrors evaluate_like()'s own algorithm: materialize the shared
// SUBSTRING computation once, then compare it against each literal and
// NULL_LOGICAL_OR-fold the results (Kleene semantics -- a NULL operand
// must propagate as NULL through every leaf identically, then OR
// together, the same reasoning expression_compiler.cpp's own AND/OR
// documents; see also compile_between()'s NULL_LOGICAL_AND fix), rather
// than plain LOGICAL_OR.
std::unique_ptr<cudf::column> FilterOperator::evaluate_substring_in(
    const CompiledSubstringInConjunct& substring_in, const cudf::table_view& batch,
    ExecutionContext& context) {
  std::unique_ptr<cudf::column> owned_operand_column;
  const cudf::column_view operand_view =
      substring_in.operand_column_index.has_value()
          ? batch.column(*substring_in.operand_column_index)
          : (owned_operand_column = cudf::compute_column(batch, *substring_in.operand_expr, context.stream,
                                                         context.memory_resource))
                ->view();

  const cudf::numeric_scalar<cudf::size_type> start(
      static_cast<cudf::size_type>(substring_in.start_zero_based), true, context.stream,
      context.memory_resource);
  const cudf::numeric_scalar<cudf::size_type> stop(
      static_cast<cudf::size_type>(substring_in.start_zero_based + substring_in.length), true, context.stream,
      context.memory_resource);
  const std::unique_ptr<cudf::column> substring_column = cudf::strings::slice_strings(
      cudf::strings_column_view(operand_view), start, stop,
      cudf::numeric_scalar<cudf::size_type>(1, true, context.stream, context.memory_resource), context.stream,
      context.memory_resource);

  std::unique_ptr<cudf::column> mask;
  for (const std::string& literal : substring_in.literals) {
    const cudf::string_scalar literal_scalar(literal, true, context.stream, context.memory_resource);
    std::unique_ptr<cudf::column> equal = cudf::binary_operation(
        substring_column->view(), literal_scalar, cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8}, context.stream, context.memory_resource);
    mask = mask == nullptr
               ? std::move(equal)
               : cudf::binary_operation(mask->view(), equal->view(), cudf::binary_operator::NULL_LOGICAL_OR,
                                        cudf::data_type{cudf::type_id::BOOL8}, context.stream,
                                        context.memory_resource);
  }
  if (!substring_in.negated) return mask;
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
    for (const CompiledSubstringInConjunct& substring_in : compiled_substring_in_conjuncts_) {
      std::unique_ptr<cudf::column> substring_mask = evaluate_substring_in(substring_in, view, context);
      mask = mask == nullptr
                 ? std::move(substring_mask)
                 : cudf::binary_operation(
                       mask->view(), substring_mask->view(), cudf::binary_operator::LOGICAL_AND,
                       cudf::data_type{cudf::type_id::BOOL8}, context.stream, context.memory_resource);
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
