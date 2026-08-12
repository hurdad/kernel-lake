#include "kernellake/sql/subquery_resolver.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace kernellake::sql {

namespace {

template <typename Node>
AstExprPtr make_expr(Node node, std::optional<std::string> alias) {
  auto result = std::make_shared<AstExpr>();
  result->node = std::move(node);
  result->alias = std::move(alias);
  return result;
}

}  // namespace

AstExprPtr resolve_subqueries(const AstExprPtr& expr,
                              const std::function<AstLiteral(const AstSelectStatement&)>& evaluate) {
  return std::visit(
      [&](const auto& node) -> AstExprPtr {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, AstSubquery>) {
          return make_expr(evaluate(*node.statement), expr->alias);
        } else if constexpr (std::is_same_v<T, AstBinary>) {
          return make_expr(AstBinary{node.op, resolve_subqueries(node.left, evaluate),
                                     resolve_subqueries(node.right, evaluate)},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstUnary>) {
          return make_expr(AstUnary{node.op, resolve_subqueries(node.operand, evaluate)}, expr->alias);
        } else if constexpr (std::is_same_v<T, AstCast>) {
          return make_expr(AstCast{resolve_subqueries(node.operand, evaluate), node.type_name,
                                   node.decimal_precision, node.decimal_scale},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstExtract>) {
          return make_expr(AstExtract{node.field, resolve_subqueries(node.operand, evaluate)}, expr->alias);
        } else if constexpr (std::is_same_v<T, AstBetween>) {
          return make_expr(
              AstBetween{resolve_subqueries(node.value, evaluate), resolve_subqueries(node.lower, evaluate),
                         resolve_subqueries(node.upper, evaluate)},
              expr->alias);
        } else if constexpr (std::is_same_v<T, AstLike>) {
          return make_expr(AstLike{resolve_subqueries(node.value, evaluate),
                                   resolve_subqueries(node.pattern, evaluate), node.negated},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstIn>) {
          // `node.subquery` (the `IN (SELECT ...)` source, see ast.hpp's
          // own comment on AstIn) is deliberately left untouched here --
          // this function only ever resolves a HAVING scalar subquery;
          // resolving an IN-subquery is resolve_in_subqueries()'s job,
          // run separately over WHERE. If one somehow reaches the binder
          // still set (e.g. it appeared inside HAVING, which never runs
          // resolve_in_subqueries()), bind_node(const AstIn&, bool)
          // rejects it with a clear, specific error.
          AstIn result;
          result.value = resolve_subqueries(node.value, evaluate);
          result.list.reserve(node.list.size());
          for (const AstExprPtr& item : node.list) {
            result.list.push_back(resolve_subqueries(item, evaluate));
          }
          result.subquery = node.subquery;
          result.negated = node.negated;
          return make_expr(std::move(result), expr->alias);
        } else if constexpr (std::is_same_v<T, AstCase>) {
          AstCase result;
          result.when_then.reserve(node.when_then.size());
          for (const auto& [condition, branch_result] : node.when_then) {
            result.when_then.emplace_back(resolve_subqueries(condition, evaluate),
                                          resolve_subqueries(branch_result, evaluate));
          }
          result.else_branch =
              node.else_branch != nullptr ? resolve_subqueries(node.else_branch, evaluate) : nullptr;
          return make_expr(std::move(result), expr->alias);
        } else if constexpr (std::is_same_v<T, AstAggregate>) {
          // `argument` is null only for CountStar -- nothing to recurse
          // into there. A subquery inside an aggregate's own argument
          // (e.g. `SUM(x + (SELECT ...))`) is an unusual thing to write
          // but not structurally different from any other nested
          // expression, so it's still resolved correctly here.
          return make_expr(
              AstAggregate{node.function,
                           node.argument != nullptr ? resolve_subqueries(node.argument, evaluate) : nullptr},
              expr->alias);
        } else {
          // AstColumnRef, AstStar, AstLiteral: leaves, no expression
          // operand to recurse into, so no subquery could be nested
          // inside one -- returned unchanged.
          return expr;
        }
      },
      expr->node);
}

AstExprPtr resolve_in_subqueries(
    const AstExprPtr& expr,
    const std::function<std::vector<AstLiteral>(const AstSelectStatement&)>& evaluate) {
  return std::visit(
      [&](const auto& node) -> AstExprPtr {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, AstIn>) {
          AstIn result;
          result.value = resolve_in_subqueries(node.value, evaluate);
          result.negated = node.negated;
          if (node.subquery != nullptr) {
            std::vector<AstLiteral> values = evaluate(*node.subquery);
            if (values.empty()) {
              // `x IN ()` is always false (`x NOT IN ()` is always true),
              // regardless of `x` -- standard SQL semantics for an empty
              // set. No need to keep `node.value` around at all here.
              return make_expr(AstLiteral{AstLiteralKind::Boolean, 0, 0.0, {}, node.negated}, expr->alias);
            }
            for (const AstLiteral& literal : values) {
              result.list.push_back(make_expr(literal, std::nullopt));
            }
            return make_expr(std::move(result), expr->alias);
          }
          result.list.reserve(node.list.size());
          for (const AstExprPtr& item : node.list) {
            result.list.push_back(resolve_in_subqueries(item, evaluate));
          }
          return make_expr(std::move(result), expr->alias);
        } else if constexpr (std::is_same_v<T, AstBinary>) {
          return make_expr(AstBinary{node.op, resolve_in_subqueries(node.left, evaluate),
                                     resolve_in_subqueries(node.right, evaluate)},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstUnary>) {
          return make_expr(AstUnary{node.op, resolve_in_subqueries(node.operand, evaluate)}, expr->alias);
        } else if constexpr (std::is_same_v<T, AstCast>) {
          return make_expr(AstCast{resolve_in_subqueries(node.operand, evaluate), node.type_name,
                                   node.decimal_precision, node.decimal_scale},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstExtract>) {
          return make_expr(AstExtract{node.field, resolve_in_subqueries(node.operand, evaluate)},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstBetween>) {
          return make_expr(AstBetween{resolve_in_subqueries(node.value, evaluate),
                                      resolve_in_subqueries(node.lower, evaluate),
                                      resolve_in_subqueries(node.upper, evaluate)},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstLike>) {
          return make_expr(AstLike{resolve_in_subqueries(node.value, evaluate),
                                   resolve_in_subqueries(node.pattern, evaluate), node.negated},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstCase>) {
          AstCase result;
          result.when_then.reserve(node.when_then.size());
          for (const auto& [condition, branch_result] : node.when_then) {
            result.when_then.emplace_back(resolve_in_subqueries(condition, evaluate),
                                          resolve_in_subqueries(branch_result, evaluate));
          }
          result.else_branch =
              node.else_branch != nullptr ? resolve_in_subqueries(node.else_branch, evaluate) : nullptr;
          return make_expr(std::move(result), expr->alias);
        } else if constexpr (std::is_same_v<T, AstAggregate>) {
          return make_expr(AstAggregate{node.function, node.argument != nullptr
                                                           ? resolve_in_subqueries(node.argument, evaluate)
                                                           : nullptr},
                           expr->alias);
        } else {
          // AstColumnRef, AstStar, AstLiteral, AstSubquery: leaves for
          // this walker's purposes -- an AstSubquery here is a HAVING-
          // scalar-subquery concern (resolve_subqueries()'s own job, not
          // this function's), left untouched.
          return expr;
        }
      },
      expr->node);
}

}  // namespace kernellake::sql
