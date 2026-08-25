#include "kernellake/sql/subquery_resolver.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace kernellake::sql {

namespace {

template <typename Node>
AstExprPtr make_expr(Node node, std::optional<std::string> alias) {
  auto result = std::make_shared<AstExpr>();
  result->node = std::move(node);
  result->alias = std::move(alias);
  return result;
}

// Flattens top-level AND conjuncts of `expr` into `out`, mirroring
// optimizer.cpp's/binder.cpp's own identically-shaped conjunct-collecting
// helpers (each duplicated locally rather than shared -- see those files'
// own comments -- since they operate on different tree types: this one on
// the untyped sql::AstExpr, the others on the typed kernellake::Expression
// post-bind).
void collect_and_conjuncts(const AstExprPtr& expr, std::vector<AstExprPtr>& out) {
  if (const auto* binary = std::get_if<AstBinary>(&expr->node);
      binary != nullptr && binary->op == AstBinaryOp::And) {
    collect_and_conjuncts(binary->left, out);
    collect_and_conjuncts(binary->right, out);
    return;
  }
  out.push_back(expr);
}

// Inverse of collect_and_conjuncts() above: rebuilds a single AND-tree
// from `conjuncts` (left-associated, in order) -- nullptr if `conjuncts`
// is empty (i.e. every original WHERE conjunct was rewritten away).
AstExprPtr rebuild_and(std::vector<AstExprPtr> conjuncts) {
  if (conjuncts.empty()) {
    return nullptr;
  }
  AstExprPtr result = std::move(conjuncts.front());
  for (std::size_t i = 1; i < conjuncts.size(); ++i) {
    result = make_expr(AstBinary{AstBinaryOp::And, result, std::move(conjuncts[i])}, std::nullopt);
  }
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
        } else if constexpr (std::is_same_v<T, AstSubstring>) {
          return make_expr(AstSubstring{resolve_subqueries(node.operand, evaluate), node.start, node.length},
                           expr->alias);
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
        } else if constexpr (std::is_same_v<T, AstSubstring>) {
          return make_expr(
              AstSubstring{resolve_in_subqueries(node.operand, evaluate), node.start, node.length},
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

AstSelectStatement rewrite_exists_subqueries(AstSelectStatement stmt) {
  // Neither shape has a real "outer alias" to correlate against yet: a
  // derived table's own alias is never threaded anywhere (see
  // AstSelectStatement::from_subquery_alias's own comment), and a
  // single-table FROM with no alias at all has no name a correlation
  // predicate inside EXISTS could reference. Left completely untouched --
  // any AstExists conjunct here reaches the binder unresolved, which
  // rejects it with a clear error.
  if (stmt.where == nullptr || stmt.from_subquery != nullptr) {
    return stmt;
  }
  if (!stmt.join.has_value() && !stmt.from.alias.has_value()) {
    return stmt;
  }

  std::vector<AstExprPtr> conjuncts;
  collect_and_conjuncts(stmt.where, conjuncts);

  std::vector<AstExprPtr> remaining;
  std::vector<AstJoinStep> new_steps;
  remaining.reserve(conjuncts.size());
  for (AstExprPtr& conjunct : conjuncts) {
    const auto* exists = std::get_if<AstExists>(&conjunct->node);
    const AstSelectStatement* sub = exists != nullptr ? exists->subquery.get() : nullptr;
    const bool rewritable = sub != nullptr && sub->where != nullptr && !sub->join.has_value() &&
                            sub->from_subquery == nullptr && sub->from.alias.has_value() &&
                            sub->group_by.empty() && sub->having == nullptr && sub->order_by.empty() &&
                            !sub->limit.has_value();
    if (!rewritable) {
      remaining.push_back(std::move(conjunct));
      continue;
    }
    new_steps.push_back(
        AstJoinStep{sub->from, sub->where,
                    exists->negated ? kernellake::JoinType::LeftAnti : kernellake::JoinType::LeftSemi});
  }
  if (new_steps.empty()) {
    // Nothing rewritable found -- return stmt exactly as received (not a
    // reconstructed-but-equivalent WHERE tree), so an unrelated AND
    // conjunct's own subtree identity/structure is never disturbed by a
    // pass that found nothing to do.
    return stmt;
  }

  if (!stmt.join.has_value()) {
    stmt.join = AstJoinClause{stmt.from, {}};
  }
  for (AstJoinStep& step : new_steps) {
    stmt.join->steps.push_back(std::move(step));
  }
  stmt.where = rebuild_and(std::move(remaining));
  return stmt;
}

namespace {

// A shallow copy of `expr` with `alias` replacing its own -- used to give
// a cloned subtree a new `AS` output name without disturbing the
// original (still referenced elsewhere, e.g. as a `GROUP BY` key).
AstExprPtr with_alias(const AstExprPtr& expr, std::string alias) {
  auto result = std::make_shared<AstExpr>(*expr);
  result->alias = std::move(alias);
  return result;
}

AstExprPtr column_ref(std::string table, std::string name) {
  return make_expr(AstColumnRef{std::move(name), std::move(table)}, std::nullopt);
}

// Removes every `AstColumnRef::table` qualifier from `expr`'s tree --
// needed when a decorrelated subquery's own inner FROM is a *single*,
// non-joined source (TPC-H Q17's shape: `FROM lineitem AS l2 WHERE
// l2.l_partkey = p.p_partkey`): the correlation conjunct's own inner-side
// column (`l2.l_partkey`) and the original SELECT-list expression
// (`0.2 * AVG(l2.l_quantity)`) both carry the subquery's own alias
// qualifier, valid there (a real JOIN query), but the synthesized derived
// table built from them keeps that single source *unjoined* (see
// try_decorrelate() below) -- and the single-table Binder rejects any
// qualified reference outright (`bind_node(const AstColumnRef&, bool)`),
// even an unambiguous one, since there's no second source to disambiguate
// from. `AstSubquery`'s own inner statement is deliberately left alone --
// a nested independent subquery has its own, unrelated alias scope.
AstExprPtr strip_table_qualifier(const AstExprPtr& expr) {
  return std::visit(
      [&](const auto& node) -> AstExprPtr {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, AstColumnRef>) {
          return make_expr(AstColumnRef{node.name, std::nullopt}, expr->alias);
        } else if constexpr (std::is_same_v<T, AstBinary>) {
          return make_expr(
              AstBinary{node.op, strip_table_qualifier(node.left), strip_table_qualifier(node.right)},
              expr->alias);
        } else if constexpr (std::is_same_v<T, AstUnary>) {
          return make_expr(AstUnary{node.op, strip_table_qualifier(node.operand)}, expr->alias);
        } else if constexpr (std::is_same_v<T, AstCast>) {
          return make_expr(AstCast{strip_table_qualifier(node.operand), node.type_name,
                                   node.decimal_precision, node.decimal_scale},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstExtract>) {
          return make_expr(AstExtract{node.field, strip_table_qualifier(node.operand)}, expr->alias);
        } else if constexpr (std::is_same_v<T, AstSubstring>) {
          return make_expr(AstSubstring{strip_table_qualifier(node.operand), node.start, node.length},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstBetween>) {
          return make_expr(AstBetween{strip_table_qualifier(node.value), strip_table_qualifier(node.lower),
                                      strip_table_qualifier(node.upper)},
                           expr->alias);
        } else if constexpr (std::is_same_v<T, AstLike>) {
          return make_expr(
              AstLike{strip_table_qualifier(node.value), strip_table_qualifier(node.pattern), node.negated},
              expr->alias);
        } else if constexpr (std::is_same_v<T, AstIn>) {
          AstIn result;
          result.value = strip_table_qualifier(node.value);
          result.negated = node.negated;
          result.subquery = node.subquery;
          for (const AstExprPtr& item : node.list) {
            result.list.push_back(strip_table_qualifier(item));
          }
          return make_expr(std::move(result), expr->alias);
        } else if constexpr (std::is_same_v<T, AstCase>) {
          AstCase result;
          for (const auto& [condition, branch_result] : node.when_then) {
            result.when_then.emplace_back(strip_table_qualifier(condition),
                                          strip_table_qualifier(branch_result));
          }
          result.else_branch =
              node.else_branch != nullptr ? strip_table_qualifier(node.else_branch) : nullptr;
          return make_expr(std::move(result), expr->alias);
        } else if constexpr (std::is_same_v<T, AstAggregate>) {
          return make_expr(
              AstAggregate{node.function,
                           node.argument != nullptr ? strip_table_qualifier(node.argument) : nullptr},
              expr->alias);
        } else {
          // AstStar, AstLiteral, AstSubquery, AstExists: nothing to strip
          // (AstSubquery/AstExists carry their own, unrelated alias
          // scope -- left untouched, see this function's own comment).
          return expr;
        }
      },
      expr->node);
}

// Every alias `stmt` itself exposes (its own FROM/JOIN sources) -- the
// "outer" scope a correlated subquery's own WHERE-clause conjunct is
// checked against, see try_decorrelate() below.
std::vector<std::string> outer_aliases(const AstSelectStatement& stmt) {
  std::vector<std::string> aliases;
  if (stmt.join.has_value()) {
    if (stmt.join->first.alias.has_value()) {
      aliases.push_back(*stmt.join->first.alias);
    }
    for (const AstJoinStep& step : stmt.join->steps) {
      if (step.source.alias.has_value()) {
        aliases.push_back(*step.source.alias);
      }
    }
  } else if (stmt.from.alias.has_value()) {
    aliases.push_back(*stmt.from.alias);
  }
  return aliases;
}

bool is_outer_alias(const std::optional<std::string>& table, const std::vector<std::string>& aliases) {
  return table.has_value() && std::find(aliases.begin(), aliases.end(), *table) != aliases.end();
}

constexpr bool is_comparison_op(AstBinaryOp op) {
  return op == AstBinaryOp::Eq || op == AstBinaryOp::NotEq || op == AstBinaryOp::Lt ||
         op == AstBinaryOp::LtEq || op == AstBinaryOp::Gt || op == AstBinaryOp::GtEq;
}

// A top-level WHERE conjunct of shape `<expr> <comparison> (SELECT ...)`
// or `(SELECT ...) <comparison> <expr>` -- the shape
// resolve_subqueries()/rewrite_exists_subqueries() don't already handle
// (a bare scalar subquery comparison, not HAVING, not EXISTS).
struct ScalarSubqueryComparison {
  AstExprPtr outer_expr;
  bool subquery_on_right;
  AstBinaryOp op;
  const AstSelectStatement* inner;
};

std::optional<ScalarSubqueryComparison> match_scalar_subquery_comparison(const AstExprPtr& expr) {
  const auto* binary = std::get_if<AstBinary>(&expr->node);
  if (binary == nullptr || !is_comparison_op(binary->op)) {
    return std::nullopt;
  }
  const auto* right_sub = std::get_if<AstSubquery>(&binary->right->node);
  const auto* left_sub = std::get_if<AstSubquery>(&binary->left->node);
  if (right_sub != nullptr && left_sub == nullptr) {
    return ScalarSubqueryComparison{binary->left, true, binary->op, right_sub->statement.get()};
  }
  if (left_sub != nullptr && right_sub == nullptr) {
    return ScalarSubqueryComparison{binary->right, false, binary->op, left_sub->statement.get()};
  }
  return std::nullopt;
}

// The result of successfully decorrelating one correlated scalar
// subquery: a synthesized derived table plus the pieces needed to wire
// it into the outer query -- see rewrite_correlated_scalar_subqueries()'s
// own doc comment (subquery_resolver.hpp) for the full scope.
struct DecorrelatedSubquery {
  AstSelectStatement derived_select;
  std::string derived_alias;
  AstExprPtr join_condition;                      // <outer key 0> = <alias>.__corr_key_0
  std::vector<AstExprPtr> extra_where_conjuncts;  // remaining correlation keys, if any
  std::string value_column_name;
};

std::optional<DecorrelatedSubquery> try_decorrelate(const AstSelectStatement& inner,
                                                    const std::vector<std::string>& outer_scope,
                                                    int synthetic_id) {
  if (inner.select_list.size() != 1 || inner.where == nullptr || inner.from_subquery != nullptr) {
    return std::nullopt;
  }
  if (!inner.group_by.empty() || inner.having != nullptr || !inner.order_by.empty() ||
      inner.limit.has_value()) {
    return std::nullopt;
  }
  if (!inner.join.has_value() && !inner.from.alias.has_value()) {
    return std::nullopt;
  }

  std::vector<AstExprPtr> conjuncts;
  collect_and_conjuncts(inner.where, conjuncts);

  std::vector<AstExprPtr> correlation_outer_exprs;
  std::vector<AstExprPtr> correlation_inner_exprs;
  std::vector<AstExprPtr> remaining_inner_conjuncts;
  for (const AstExprPtr& conjunct : conjuncts) {
    const auto* binary = std::get_if<AstBinary>(&conjunct->node);
    const auto* left_col = binary != nullptr ? std::get_if<AstColumnRef>(&binary->left->node) : nullptr;
    const auto* right_col = binary != nullptr ? std::get_if<AstColumnRef>(&binary->right->node) : nullptr;
    const bool eq =
        binary != nullptr && binary->op == AstBinaryOp::Eq && left_col != nullptr && right_col != nullptr;
    const bool left_is_outer = eq && is_outer_alias(left_col->table, outer_scope);
    const bool right_is_outer = eq && is_outer_alias(right_col->table, outer_scope);
    if (eq && left_is_outer && !right_is_outer) {
      correlation_outer_exprs.push_back(binary->left);
      correlation_inner_exprs.push_back(binary->right);
    } else if (eq && right_is_outer && !left_is_outer) {
      correlation_outer_exprs.push_back(binary->right);
      correlation_inner_exprs.push_back(binary->left);
    } else {
      remaining_inner_conjuncts.push_back(conjunct);
    }
  }
  if (correlation_outer_exprs.empty()) {
    // No correlation found -- not this function's shape (a non-correlated
    // scalar subquery, resolve_subqueries()'s own job).
    return std::nullopt;
  }

  // Only meaningful when `inner`'s own FROM is a single, non-joined
  // source (Q17's shape): its qualified column references (`l2.l_partkey`)
  // are valid there, but the derived table built below keeps that single
  // source unjoined (`derived.join` stays unset), and the single-table
  // Binder rejects *any* qualified reference -- see
  // strip_table_qualifier()'s own comment. A no-op (returns its argument
  // unchanged) when `inner.join.has_value()` (Q2's multi-way-join shape),
  // where every reference genuinely needs its qualifier.
  const bool inner_is_single_source = !inner.join.has_value();
  auto adapt = [&](const AstExprPtr& expr) {
    return inner_is_single_source ? strip_table_qualifier(expr) : expr;
  };

  const std::string alias = fmt::format("__corr_subq_{}", synthetic_id);
  AstSelectStatement derived;
  derived.from = inner.from;
  derived.join = inner.join;
  for (AstExprPtr& conjunct : remaining_inner_conjuncts) {
    conjunct = adapt(conjunct);
  }
  derived.where = rebuild_and(std::move(remaining_inner_conjuncts));
  for (std::size_t i = 0; i < correlation_inner_exprs.size(); ++i) {
    const AstExprPtr adapted = adapt(correlation_inner_exprs[i]);
    derived.select_list.push_back(with_alias(adapted, fmt::format("__corr_key_{}", i)));
    derived.group_by.push_back(adapted);
  }
  constexpr const char* kValueColumn = "__corr_val_0";
  derived.select_list.push_back(with_alias(adapt(inner.select_list[0]), kValueColumn));

  DecorrelatedSubquery result;
  result.join_condition =
      make_expr(AstBinary{AstBinaryOp::Eq, correlation_outer_exprs[0], column_ref(alias, "__corr_key_0")},
                std::nullopt);
  for (std::size_t i = 1; i < correlation_outer_exprs.size(); ++i) {
    result.extra_where_conjuncts.push_back(
        make_expr(AstBinary{AstBinaryOp::Eq, correlation_outer_exprs[i],
                            column_ref(alias, fmt::format("__corr_key_{}", i))},
                  std::nullopt));
  }
  result.derived_select = std::move(derived);
  result.derived_alias = alias;
  result.value_column_name = kValueColumn;
  return result;
}

}  // namespace

AstSelectStatement rewrite_correlated_scalar_subqueries(AstSelectStatement stmt) {
  if (stmt.where == nullptr || stmt.from_subquery != nullptr) {
    return stmt;
  }
  if (!stmt.join.has_value() && !stmt.from.alias.has_value()) {
    return stmt;
  }

  const std::vector<std::string> outer_scope = outer_aliases(stmt);
  std::vector<AstExprPtr> conjuncts;
  collect_and_conjuncts(stmt.where, conjuncts);

  std::vector<AstJoinStep> new_steps;
  std::vector<AstExprPtr> extra_conjuncts;
  int synthetic_id = 0;
  for (AstExprPtr& conjunct : conjuncts) {
    const std::optional<ScalarSubqueryComparison> match =  // NOLINT(bugprone-unchecked-optional-access)
        match_scalar_subquery_comparison(conjunct);
    if (!match.has_value()) {
      continue;
    }
    std::optional<DecorrelatedSubquery> decorrelated =  // NOLINT(bugprone-unchecked-optional-access)
        try_decorrelate(*match->inner, outer_scope, synthetic_id);
    if (!decorrelated.has_value()) {
      continue;
    }
    ++synthetic_id;

    AstJoinStep step;
    step.source.alias = decorrelated->derived_alias;
    step.derived_source = std::make_shared<AstSelectStatement>(std::move(decorrelated->derived_select));
    step.condition = decorrelated->join_condition;
    step.join_type = kernellake::JoinType::Inner;
    new_steps.push_back(std::move(step));
    for (const AstExprPtr& extra : decorrelated->extra_where_conjuncts) {
      extra_conjuncts.push_back(extra);
    }

    // decorrelated/match were both just confirmed has_value() above (the
    // early `continue`s), so every access below is safe -- clang-tidy's
    // unchecked-optional-access check can't see across that control flow,
    // hence the NOLINTs.
    const std::string derived_alias = decorrelated->derived_alias;          // NOLINT(*-optional-access)
    const std::string value_column_name = decorrelated->value_column_name;  // NOLINT(*-optional-access)
    const AstBinaryOp op = match->op;                                       // NOLINT(*-optional-access)
    const bool subquery_on_right = match->subquery_on_right;                // NOLINT(*-optional-access)
    const AstExprPtr outer_expr = match->outer_expr;                        // NOLINT(*-optional-access)

    AstExprPtr value_ref = column_ref(derived_alias, value_column_name);
    conjunct = subquery_on_right ? make_expr(AstBinary{op, outer_expr, value_ref}, std::nullopt)
                                 : make_expr(AstBinary{op, value_ref, outer_expr}, std::nullopt);
  }
  if (new_steps.empty()) {
    // Nothing rewritable found -- return stmt exactly as received, same
    // reasoning as rewrite_exists_subqueries()'s own identical guard.
    return stmt;
  }

  if (!stmt.join.has_value()) {
    stmt.join = AstJoinClause{stmt.from, {}};
  }
  for (AstJoinStep& step : new_steps) {
    stmt.join->steps.push_back(std::move(step));
  }
  for (AstExprPtr& extra : extra_conjuncts) {
    conjuncts.push_back(std::move(extra));
  }
  stmt.where = rebuild_and(std::move(conjuncts));
  return stmt;
}

}  // namespace kernellake::sql
