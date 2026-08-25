#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kernellake/execution_gpu/expression_compiler.hpp"
#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/expression/expression.hpp"

namespace kernellake {

// Filters each batch from `child` by `predicate`, evaluated via
// cudf::compute_column + cudf::apply_boolean_mask. Batches that filter down
// to zero rows are skipped rather than returned, so downstream operators
// only ever see non-empty batches.
//
// cudf::ast has no LIKE operator, so a `LikeExpression` (or its negation)
// appearing as one of the predicate's top-level AND-connected conjuncts
// (e.g. `WHERE region LIKE 'A%' AND amount > 10`) is evaluated separately
// via cudf::strings::like and combined with the AST-evaluated remainder via
// a column-level AND -- see docs/ARCHITECTURE.md. A LIKE anywhere else
// (inside OR, arithmetic, or the SELECT list) is not supported; the
// ExpressionCompiler's "unrecognized expression type" error is the natural
// failure point for that.
//
// Same idea for `SUBSTRING(x, start, len) IN (lit1, lit2, ...)` (TPC-H
// Q22's shape) -- cudf::ast has no substring operator either, and unlike
// LIKE this one isn't even a single leaf: the binder desugars `IN` into an
// OR-chain of `SUBSTRING(...) = lit` equalities (see bind_node(AstIn&,
// bool)), so the whole top-level conjunct is an OR-tree, not one node.
// Detected as a whole conjunct (mirroring the LIKE case's own
// whole-conjunct-only scope, not a general "SUBSTRING anywhere" rewrite):
// every leaf of the OR-tree (or, for `NOT IN`, the OR-tree directly under
// a top-level `NOT`) must be `SUBSTRING(same operand, same start, same
// length) = <string literal>` -- the substring is materialized once, then
// compared against each literal and OR-folded, mirroring evaluate_like()'s
// own column-level combination.
class FilterOperator final : public PhysicalOperator {
 public:
  FilterOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child, ExpressionPtr predicate);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "Filter"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  struct CompiledLikeConjunct {
    std::optional<cudf::size_type> value_column_index;
    const cudf::ast::expression* value_expr = nullptr;  // set iff value_column_index is not
    std::string pattern;
    bool negated;
  };

  struct CompiledSubstringInConjunct {
    std::optional<cudf::size_type> operand_column_index;
    const cudf::ast::expression* operand_expr = nullptr;  // set iff operand_column_index is not
    std::int64_t start_zero_based;
    std::int64_t length;
    std::vector<std::string> literals;
    bool negated;
  };

  [[nodiscard]] std::unique_ptr<cudf::column> evaluate_like(const CompiledLikeConjunct& like,
                                                            const cudf::table_view& batch,
                                                            ExecutionContext& context);
  [[nodiscard]] std::unique_ptr<cudf::column> evaluate_substring_in(
      const CompiledSubstringInConjunct& substring_in, const cudf::table_view& batch,
      ExecutionContext& context);
  // Returns the compiled conjunct if `conjunct` matches the whole-conjunct
  // SUBSTRING-IN shape described in this class's own doc comment, or
  // std::nullopt otherwise (the caller falls back to the ordinary AST
  // path). Not `const` -- populates operand_expr via compiler_ when the
  // substring's own operand isn't a plain column.
  [[nodiscard]] std::optional<CompiledSubstringInConjunct> try_compile_substring_in(
      const ExpressionPtr& conjunct, ExecutionContext& context);

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  ExpressionPtr predicate_;
  ExpressionCompiler compiler_;
  const cudf::ast::expression* compiled_ast_conjuncts_ = nullptr;  // null if every conjunct is a LIKE
  std::vector<CompiledLikeConjunct> compiled_like_conjuncts_;
  std::vector<CompiledSubstringInConjunct> compiled_substring_in_conjuncts_;
};

}  // namespace kernellake
