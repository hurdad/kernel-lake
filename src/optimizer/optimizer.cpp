#include "kernellake/optimizer/optimizer.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

// ---------------------------------------------------------------------------
// Expression-level rules: constant folding, boolean simplification, and
// BETWEEN simplification. Rebuilds bottom-up, reusing the original pointer
// wherever nothing changed.
// ---------------------------------------------------------------------------

std::optional<double> literal_as_double(const LiteralExpression& literal) {
  if (literal.is_null()) {
    return std::nullopt;
  }
  if (std::holds_alternative<std::int64_t>(literal.value())) {
    return static_cast<double>(std::get<std::int64_t>(literal.value()));
  }
  if (std::holds_alternative<double>(literal.value())) {
    return std::get<double>(literal.value());
  }
  return std::nullopt;
}

std::optional<double> fold_arithmetic(BinaryOperator op, double left, double right) {
  switch (op) {
    case BinaryOperator::Add:
      return left + right;
    case BinaryOperator::Subtract:
      return left - right;
    case BinaryOperator::Multiply:
      return left * right;
    case BinaryOperator::Divide:
      return right != 0.0 ? std::optional<double>(left / right) : std::nullopt;
    default:
      return std::nullopt;
  }
}

std::optional<bool> fold_numeric_comparison(BinaryOperator op, double left, double right) {
  switch (op) {
    case BinaryOperator::Equal:
      return left == right;
    case BinaryOperator::NotEqual:
      return left != right;
    case BinaryOperator::Less:
      return left < right;
    case BinaryOperator::LessEqual:
      return left <= right;
    case BinaryOperator::Greater:
      return left > right;
    case BinaryOperator::GreaterEqual:
      return left >= right;
    default:
      return std::nullopt;
  }
}

// Exact-integer counterparts of fold_arithmetic/fold_numeric_comparison
// above, used only when both literals are int64 (see simplify_expression):
// round-tripping an int64 through double loses precision past 2^53 and,
// worse, static_cast<int64_t>(double) is undefined behavior once the
// double's magnitude exceeds INT64_MAX/MIN -- both are real risks for
// int64, whose whole range is far wider than a double's 53-bit mantissa.
// Returns nullopt on overflow (checked via __builtin_*_overflow, supported
// by both GCC and Clang, this project's only two compilers) rather than
// folding to a wrapped/UB result -- the expression is simply left
// unfolded and evaluated at runtime instead, same as the existing
// divide-by-zero behavior above.
std::optional<std::int64_t> fold_arithmetic_int64(BinaryOperator op, std::int64_t left, std::int64_t right) {
  std::int64_t result = 0;
  switch (op) {
    case BinaryOperator::Add:
      return __builtin_add_overflow(left, right, &result) ? std::nullopt
                                                          : std::optional<std::int64_t>(result);
    case BinaryOperator::Subtract:
      return __builtin_sub_overflow(left, right, &result) ? std::nullopt
                                                          : std::optional<std::int64_t>(result);
    case BinaryOperator::Multiply:
      return __builtin_mul_overflow(left, right, &result) ? std::nullopt
                                                          : std::optional<std::int64_t>(result);
    case BinaryOperator::Divide:
      // INT64_MIN / -1 overflows (magnitude exceeds INT64_MAX) and is UB
      // for the raw '/' operator -- skip folding that case too, same as
      // divide-by-zero.
      if (right == 0 || (left == std::numeric_limits<std::int64_t>::min() && right == -1)) {
        return std::nullopt;
      }
      return left / right;
    default:
      return std::nullopt;
  }
}

std::optional<bool> fold_integer_comparison(BinaryOperator op, std::int64_t left, std::int64_t right) {
  switch (op) {
    case BinaryOperator::Equal:
      return left == right;
    case BinaryOperator::NotEqual:
      return left != right;
    case BinaryOperator::Less:
      return left < right;
    case BinaryOperator::LessEqual:
      return left <= right;
    case BinaryOperator::Greater:
      return left > right;
    case BinaryOperator::GreaterEqual:
      return left >= right;
    default:
      return std::nullopt;
  }
}

ExpressionPtr make_bool_literal(bool value) {
  return std::make_shared<LiteralExpression>(LiteralExpression::make_bool(value));
}

ExpressionPtr simplify_expression(const ExpressionPtr& expr) {
  if (const auto* between = dynamic_cast<const BetweenExpression*>(expr.get())) {
    ExpressionPtr value = simplify_expression(between->value());
    ExpressionPtr lower = simplify_expression(between->lower());
    ExpressionPtr upper = simplify_expression(between->upper());
    const bool nullable =
        value->result_type().nullable || lower->result_type().nullable || upper->result_type().nullable;
    ExpressionPtr ge = std::make_shared<BinaryExpression>(BinaryOperator::GreaterEqual, value, lower,
                                                          boolean_type(nullable));
    ExpressionPtr le =
        std::make_shared<BinaryExpression>(BinaryOperator::LessEqual, value, upper, boolean_type(nullable));
    ExpressionPtr conjunction =
        std::make_shared<BinaryExpression>(BinaryOperator::And, ge, le, boolean_type(nullable));
    return simplify_expression(conjunction);
  }
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(expr.get())) {
    ExpressionPtr operand = simplify_expression(unary->operand());
    if (unary->op() == UnaryOperator::Not) {
      if (const auto* lit = dynamic_cast<const LiteralExpression*>(operand.get())) {
        if (!lit->is_null() && std::holds_alternative<bool>(lit->value())) {
          return make_bool_literal(!std::get<bool>(lit->value()));
        }
      }
      if (const auto* inner = dynamic_cast<const UnaryExpression*>(operand.get())) {
        if (inner->op() == UnaryOperator::Not) {
          return inner->operand();
        }
      }
    }
    if (operand.get() == unary->operand().get()) {
      return expr;
    }
    return std::make_shared<UnaryExpression>(unary->op(), std::move(operand), unary->result_type());
  }
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(expr.get())) {
    ExpressionPtr left = simplify_expression(binary->left());
    ExpressionPtr right = simplify_expression(binary->right());
    const auto* left_lit = dynamic_cast<const LiteralExpression*>(left.get());
    const auto* right_lit = dynamic_cast<const LiteralExpression*>(right.get());
    const BinaryOperator op = binary->op();

    if (is_logical(op)) {
      if (left_lit != nullptr && !left_lit->is_null() && std::holds_alternative<bool>(left_lit->value())) {
        const bool lv = std::get<bool>(left_lit->value());
        return op == BinaryOperator::And ? (lv ? right : make_bool_literal(false))
                                         : (lv ? make_bool_literal(true) : right);
      }
      if (right_lit != nullptr && !right_lit->is_null() && std::holds_alternative<bool>(right_lit->value())) {
        const bool rv = std::get<bool>(right_lit->value());
        return op == BinaryOperator::And ? (rv ? left : make_bool_literal(false))
                                         : (rv ? make_bool_literal(true) : left);
      }
    } else if (left_lit != nullptr && right_lit != nullptr && !left_lit->is_null() && !right_lit->is_null() &&
               std::holds_alternative<std::int64_t>(left_lit->value()) &&
               std::holds_alternative<std::int64_t>(right_lit->value())) {
      // Both operands are int64: fold with exact integer arithmetic rather
      // than the double round-trip below (see fold_arithmetic_int64's own
      // comment for why -- precision loss past 2^53, UB past INT64_MAX/MIN).
      const std::int64_t lv = std::get<std::int64_t>(left_lit->value());
      const std::int64_t rv = std::get<std::int64_t>(right_lit->value());
      if (is_arithmetic(op)) {
        if (const std::optional<std::int64_t> result = fold_arithmetic_int64(op, lv, rv)) {
          const bool as_float =
              binary->result_type().id == TypeId::Float64 || binary->result_type().id == TypeId::Float32;
          return as_float ? std::make_shared<LiteralExpression>(
                                LiteralExpression::make_float64(static_cast<double>(*result)))
                          : std::make_shared<LiteralExpression>(LiteralExpression::make_int64(*result));
        }
      } else if (is_comparison(op)) {
        if (const std::optional<bool> result = fold_integer_comparison(op, lv, rv)) {
          return make_bool_literal(*result);
        }
      }
    } else if (left_lit != nullptr && right_lit != nullptr) {
      const std::optional<double> lv = literal_as_double(*left_lit);
      const std::optional<double> rv = literal_as_double(*right_lit);
      if (lv.has_value() && rv.has_value()) {
        if (is_arithmetic(op)) {
          if (const std::optional<double> result = fold_arithmetic(op, *lv, *rv)) {
            const bool as_float =
                binary->result_type().id == TypeId::Float64 || binary->result_type().id == TypeId::Float32;
            return as_float ? std::make_shared<LiteralExpression>(LiteralExpression::make_float64(*result))
                            : std::make_shared<LiteralExpression>(
                                  LiteralExpression::make_int64(static_cast<std::int64_t>(*result)));
          }
        } else if (is_comparison(op)) {
          if (const std::optional<bool> result = fold_numeric_comparison(op, *lv, *rv)) {
            return make_bool_literal(*result);
          }
        }
      }
    }

    if (left.get() == binary->left().get() && right.get() == binary->right().get()) {
      return expr;
    }
    return std::make_shared<BinaryExpression>(op, std::move(left), std::move(right), binary->result_type());
  }
  if (const auto* cast = dynamic_cast<const CastExpression*>(expr.get())) {
    ExpressionPtr operand = simplify_expression(cast->operand());
    if (operand.get() == cast->operand().get()) {
      return expr;
    }
    return std::make_shared<CastExpression>(std::move(operand), cast->result_type());
  }
  if (const auto* aggregate = dynamic_cast<const AggregateExpression*>(expr.get())) {
    if (aggregate->argument() == nullptr) {
      return expr;
    }
    ExpressionPtr argument = simplify_expression(aggregate->argument());
    if (argument.get() == aggregate->argument().get()) {
      return expr;
    }
    return std::make_shared<AggregateExpression>(aggregate->function(), std::move(argument),
                                                 aggregate->result_type());
  }
  if (const auto* like = dynamic_cast<const LikeExpression*>(expr.get())) {
    ExpressionPtr value = simplify_expression(like->value());
    if (value.get() == like->value().get()) {
      return expr;
    }
    return std::make_shared<LikeExpression>(std::move(value), like->pattern(), like->negated());
  }
  if (const auto* case_expr = dynamic_cast<const CaseExpression*>(expr.get())) {
    bool changed = false;
    std::vector<CaseExpression::WhenThen> when_then;
    when_then.reserve(case_expr->when_then().size());
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      ExpressionPtr condition = simplify_expression(branch.condition);
      ExpressionPtr result = simplify_expression(branch.result);
      changed = changed || condition.get() != branch.condition.get() || result.get() != branch.result.get();
      when_then.push_back(CaseExpression::WhenThen{std::move(condition), std::move(result)});
    }
    ExpressionPtr else_branch =
        case_expr->else_branch() != nullptr ? simplify_expression(case_expr->else_branch()) : nullptr;
    changed = changed || else_branch.get() != case_expr->else_branch().get();
    if (!changed) {
      return expr;
    }
    return std::make_shared<CaseExpression>(std::move(when_then), std::move(else_branch),
                                            case_expr->result_type());
  }
  return expr;  // ColumnExpression, LiteralExpression: nothing to simplify.
}

// ---------------------------------------------------------------------------
// Expression-tree column collection, used for projection pushdown.
// ---------------------------------------------------------------------------

// Collects each referenced column's *index* (not name): required for
// LogicalJoin, where a combined post-join row can have two same-named
// columns from opposite sides, and only the index unambiguously says which
// side a given ColumnExpression actually came from (see the LogicalJoin
// branch in annotate_scan() below).
void collect_columns(const ExpressionPtr& expr, std::unordered_set<std::size_t>& out) {
  if (expr == nullptr) {
    return;
  }
  if (const auto* column = dynamic_cast<const ColumnExpression*>(expr.get())) {
    out.insert(column->column_index());
  } else if (const auto* binary = dynamic_cast<const BinaryExpression*>(expr.get())) {
    collect_columns(binary->left(), out);
    collect_columns(binary->right(), out);
  } else if (const auto* unary = dynamic_cast<const UnaryExpression*>(expr.get())) {
    collect_columns(unary->operand(), out);
  } else if (const auto* between = dynamic_cast<const BetweenExpression*>(expr.get())) {
    collect_columns(between->value(), out);
    collect_columns(between->lower(), out);
    collect_columns(between->upper(), out);
  } else if (const auto* cast = dynamic_cast<const CastExpression*>(expr.get())) {
    collect_columns(cast->operand(), out);
  } else if (const auto* aggregate = dynamic_cast<const AggregateExpression*>(expr.get())) {
    collect_columns(aggregate->argument(), out);
  } else if (const auto* like = dynamic_cast<const LikeExpression*>(expr.get())) {
    collect_columns(like->value(), out);
  } else if (const auto* case_expr = dynamic_cast<const CaseExpression*>(expr.get())) {
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      collect_columns(branch.condition, out);
      collect_columns(branch.result, out);
    }
    collect_columns(case_expr->else_branch(), out);
  }
  // LiteralExpression: nothing to collect. AstIn desugars into
  // BinaryExpression at bind time (see binder.cpp), so it needs no case
  // here.
}

// Unwraps CastExpression to find the underlying column/literal, for
// predicate-pushdown pattern matching (`CAST(col AS T) OP literal` still
// counts as a pushable comparison on `col`).
const Expression* unwrap_cast(const Expression* expr) {
  while (const auto* cast = dynamic_cast<const CastExpression*>(expr)) {
    expr = cast->operand().get();
  }
  return expr;
}

BinaryOperator flip(BinaryOperator op) {
  switch (op) {
    case BinaryOperator::Less:
      return BinaryOperator::Greater;
    case BinaryOperator::LessEqual:
      return BinaryOperator::GreaterEqual;
    case BinaryOperator::Greater:
      return BinaryOperator::Less;
    case BinaryOperator::GreaterEqual:
      return BinaryOperator::LessEqual;
    default:
      return op;
  }
}

void collect_pushable_predicates(const ExpressionPtr& predicate, std::vector<PushablePredicate>& out) {
  const auto* binary = dynamic_cast<const BinaryExpression*>(predicate.get());
  if (binary == nullptr) {
    return;
  }

  if (binary->op() == BinaryOperator::And) {
    collect_pushable_predicates(binary->left(), out);
    collect_pushable_predicates(binary->right(), out);
    return;
  }
  if (!is_comparison(binary->op())) {
    return;
  }

  const Expression* left = unwrap_cast(binary->left().get());
  const Expression* right = unwrap_cast(binary->right().get());
  const auto* left_col = dynamic_cast<const ColumnExpression*>(left);
  const auto* right_col = dynamic_cast<const ColumnExpression*>(right);
  const auto* left_lit = dynamic_cast<const LiteralExpression*>(left);
  const auto* right_lit = dynamic_cast<const LiteralExpression*>(right);

  if (left_col != nullptr && right_lit != nullptr) {
    out.push_back(PushablePredicate{left_col->name(), binary->op(), binary->right()});
  } else if (right_col != nullptr && left_lit != nullptr) {
    out.push_back(PushablePredicate{right_col->name(), flip(binary->op()), binary->left()});
  }
}

// ---------------------------------------------------------------------------
// Plan-level structural rules.
// ---------------------------------------------------------------------------

LogicalPlanPtr rewrite_plan(const LogicalPlanPtr& node);

// Pushes a LIMIT down through any chain of pass-through LogicalProjection
// nodes (which never change row count or order) so it ends up sitting
// directly above the first node that can actually benefit from it
// (Filter/Sort/Aggregate/Scan).
LogicalPlanPtr insert_limit(LogicalPlanPtr node, std::int64_t limit) {
  if (const auto* projection = dynamic_cast<const LogicalProjection*>(node.get())) {
    LogicalPlanPtr new_child = insert_limit(projection->child(), limit);
    return std::make_shared<LogicalProjection>(std::move(new_child), projection->items());
  }
  return std::make_shared<LogicalLimit>(std::move(node), limit);
}

bool is_identity_projection(const LogicalProjection& projection, const Schema& child_schema) {
  const std::vector<NamedExpression>& items = projection.items();
  if (items.size() != child_schema.field_count()) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto* column = dynamic_cast<const ColumnExpression*>(items[i].expr.get());
    if (column == nullptr) {
      return false;
    }
    if (column->column_index() != i) {
      return false;
    }
    if (items[i].name != child_schema.field(i).name) {
      return false;
    }
  }
  return true;
}

LogicalPlanPtr rewrite_plan(const LogicalPlanPtr& node) {
  if (dynamic_cast<const LogicalScan*>(node.get()) != nullptr) {
    return node;
  }

  if (const auto* join = dynamic_cast<const LogicalJoin*>(node.get())) {
    // Nothing to simplify about the join itself (its "expression" is just
    // two column indices, not an Expression tree) -- only its two subtrees
    // need rewriting. Each is either a bare LogicalScan (a plain
    // `read_parquet(...)` source) or, for an N-way join chain, itself
    // another LogicalJoin (the left-deep shape build_logical_plan()
    // builds -- see docs/ARCHITECTURE.md's "Hash joins" section) -- the
    // recursive rewrite_plan() call below handles either shape identically,
    // dispatching on the child's actual type rather than assuming one.
    LogicalPlanPtr left = rewrite_plan(join->left());
    LogicalPlanPtr right = rewrite_plan(join->right());
    return std::make_shared<LogicalJoin>(std::move(left), std::move(right), join->left_key_index(),
                                         join->right_key_index());
  }

  if (const auto* filter = dynamic_cast<const LogicalFilter*>(node.get())) {
    LogicalPlanPtr child = rewrite_plan(filter->child());
    ExpressionPtr predicate = simplify_expression(filter->predicate());

    if (const auto* child_filter = dynamic_cast<const LogicalFilter*>(child.get())) {
      const bool nullable =
          predicate->result_type().nullable || child_filter->predicate()->result_type().nullable;
      ExpressionPtr combined = simplify_expression(std::make_shared<BinaryExpression>(
          BinaryOperator::And, predicate, child_filter->predicate(), boolean_type(nullable)));
      return rewrite_plan(std::make_shared<LogicalFilter>(child_filter->child(), combined));
    }

    if (const auto* literal = dynamic_cast<const LiteralExpression*>(predicate.get());
        literal != nullptr && !literal->is_null() && std::holds_alternative<bool>(literal->value())) {
      if (std::get<bool>(literal->value())) {
        return child;  // Filter that always passes: remove it.
      }
      auto always_false = std::make_shared<LogicalFilter>(child, predicate);
      always_false->estimated_rows = 0;
      return always_false;
    }

    return std::make_shared<LogicalFilter>(std::move(child), std::move(predicate));
  }

  if (const auto* projection = dynamic_cast<const LogicalProjection*>(node.get())) {
    LogicalPlanPtr child = rewrite_plan(projection->child());
    std::vector<NamedExpression> items;
    items.reserve(projection->items().size());
    for (const NamedExpression& item : projection->items()) {
      items.push_back(NamedExpression{simplify_expression(item.expr), item.name});
    }
    auto rebuilt = std::make_shared<LogicalProjection>(child, std::move(items));
    if (is_identity_projection(*rebuilt, child->output_schema())) {
      return child;
    }
    return rebuilt;
  }

  if (const auto* aggregate = dynamic_cast<const LogicalAggregate*>(node.get())) {
    LogicalPlanPtr child = rewrite_plan(aggregate->child());
    std::vector<NamedExpression> group_by;
    for (const NamedExpression& item : aggregate->group_by()) {
      group_by.push_back(NamedExpression{simplify_expression(item.expr), item.name});
    }
    std::vector<NamedExpression> aggregates;
    for (const NamedExpression& item : aggregate->aggregates()) {
      aggregates.push_back(NamedExpression{simplify_expression(item.expr), item.name});
    }
    return std::make_shared<LogicalAggregate>(std::move(child), std::move(group_by), std::move(aggregates));
  }

  if (const auto* sort = dynamic_cast<const LogicalSort*>(node.get())) {
    LogicalPlanPtr child = rewrite_plan(sort->child());
    std::vector<LogicalSort::Key> keys;
    for (const LogicalSort::Key& key : sort->keys()) {
      keys.push_back(LogicalSort::Key{simplify_expression(key.expr), key.ascending});
    }
    return std::make_shared<LogicalSort>(std::move(child), std::move(keys));
  }

  if (const auto* limit = dynamic_cast<const LogicalLimit*>(node.get())) {
    LogicalPlanPtr child = rewrite_plan(limit->child());
    return insert_limit(std::move(child), limit->limit());
  }

  throw PlanningError("optimizer encountered an unrecognized logical plan node");
}

// ---------------------------------------------------------------------------
// Projection and predicate pushdown: a final top-down pass that annotates
// the single LogicalScan leaf. Column references made by a LogicalProjection
// sitting directly on a LogicalAggregate describe the *aggregate's* output
// schema, not the scan's, and are correctly excluded.
// ---------------------------------------------------------------------------

void annotate_scan(const LogicalPlanPtr& node, std::unordered_set<std::size_t>& required_columns,
                   std::vector<PushablePredicate>& pushable_predicates) {
  if (const auto* filter = dynamic_cast<const LogicalFilter*>(node.get())) {
    collect_columns(filter->predicate(), required_columns);
    collect_pushable_predicates(filter->predicate(), pushable_predicates);
    annotate_scan(filter->child(), required_columns, pushable_predicates);
  } else if (const auto* sort = dynamic_cast<const LogicalSort*>(node.get())) {
    for (const LogicalSort::Key& key : sort->keys()) {
      collect_columns(key.expr, required_columns);
    }
    annotate_scan(sort->child(), required_columns, pushable_predicates);
  } else if (const auto* aggregate = dynamic_cast<const LogicalAggregate*>(node.get())) {
    for (const NamedExpression& item : aggregate->group_by()) {
      collect_columns(item.expr, required_columns);
    }
    for (const NamedExpression& item : aggregate->aggregates()) {
      collect_columns(item.expr, required_columns);
    }
    annotate_scan(aggregate->child(), required_columns, pushable_predicates);
  } else if (const auto* projection = dynamic_cast<const LogicalProjection*>(node.get())) {
    const bool reads_aggregate_output =
        dynamic_cast<const LogicalAggregate*>(projection->child().get()) != nullptr;
    if (!reads_aggregate_output) {
      for (const NamedExpression& item : projection->items()) {
        collect_columns(item.expr, required_columns);
      }
    }
    annotate_scan(projection->child(), required_columns, pushable_predicates);
  } else if (const auto* limit = dynamic_cast<const LogicalLimit*>(node.get())) {
    annotate_scan(limit->child(), required_columns, pushable_predicates);
  } else if (const auto* join = dynamic_cast<const LogicalJoin*>(node.get())) {
    // Splits the combined-index required-columns set collected so far by
    // which side of the join each index actually belongs to -- unlike every
    // other node above, a join has two independent scan subtrees below it,
    // each needing its own local (offset-corrected) index space. The join
    // key itself is required on both sides regardless of what's referenced
    // above (HashJoinOperator needs it to build/probe), even if the query
    // never otherwise selects it.
    //
    // Predicate pushdown does not cross a join in this version: any
    // pushable_predicates collected above (from a WHERE clause that also
    // mixes join-key columns) are deliberately dropped here rather than
    // routed to one side, since PushablePredicate's bare column_name has no
    // way to say which side it came from once two schemas are in play. Scan
    // pruning still runs per-side in the physical planner; it just always
    // sees an empty predicate list for a JOIN query. See docs/ARCHITECTURE.md.
    const std::size_t left_count = join->left()->output_schema().field_count();
    std::unordered_set<std::size_t> left_required;
    std::unordered_set<std::size_t> right_required;
    for (const std::size_t index : required_columns) {
      if (index < left_count) {
        left_required.insert(index);
      } else {
        right_required.insert(index - left_count);
      }
    }
    left_required.insert(join->left_key_index());
    right_required.insert(join->right_key_index());

    std::vector<PushablePredicate> no_left_pushdown;
    annotate_scan(join->left(), left_required, no_left_pushdown);
    std::vector<PushablePredicate> no_right_pushdown;
    annotate_scan(join->right(), right_required, no_right_pushdown);
  } else if (auto* scan = dynamic_cast<LogicalScan*>(node.get())) {
    std::vector<std::string> columns;
    columns.reserve(required_columns.size());
    for (const std::size_t index : required_columns) {
      columns.push_back(scan->output_schema().field(index).name);
    }
    std::sort(columns.begin(), columns.end());
    scan->set_required_columns(std::move(columns));
    scan->set_pushable_predicates(std::move(pushable_predicates));
  }
}

}  // namespace

LogicalPlanPtr optimize(const LogicalPlanPtr& plan) {
  LogicalPlanPtr rewritten = rewrite_plan(plan);
  std::unordered_set<std::size_t> required_columns;
  std::vector<PushablePredicate> pushable_predicates;
  annotate_scan(rewritten, required_columns, pushable_predicates);
  return rewritten;
}

}  // namespace kernellake
