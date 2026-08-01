#include "kernellake/planner/binder.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

using sql::AstAggregate;
using sql::AstAggregateFunc;
using sql::AstBetween;
using sql::AstBinary;
using sql::AstBinaryOp;
using sql::AstCase;
using sql::AstCast;
using sql::AstColumnRef;
using sql::AstExpr;
using sql::AstExprPtr;
using sql::AstIn;
using sql::AstLike;
using sql::AstLiteral;
using sql::AstLiteralKind;
using sql::AstStar;
using sql::AstUnary;
using sql::AstUnaryOp;

bool is_numeric(TypeId id) {
  switch (id) {
    case TypeId::Int32:
    case TypeId::Int64:
    case TypeId::UInt32:
    case TypeId::UInt64:
    case TypeId::Float32:
    case TypeId::Float64:
    case TypeId::Decimal:
      return true;
    default:
      return false;
  }
}

bool is_floating(TypeId id) {
  return id == TypeId::Float32 || id == TypeId::Float64;
}

// Picks the smallest KernelLake numeric type that can represent both inputs
// without loss for the common cases KernelLake cares about (int/int and
// int/float promotion). Two DECIMALs with different precision/scale are
// rejected rather than guessed at, since choosing a safe widened
// precision/scale automatically is not yet implemented. Exactly one side
// being DECIMAL returns that side's exact type unchanged -- cast_if_needed
// below is what actually enforces that the *other* side can only be a
// numeric literal (retyped in place, no precision loss) rather than a
// column or computed expression (which would need a genuine runtime CAST to
// DECIMAL, and cudf::ast has no such operator -- see docs/ARCHITECTURE.md).
DataType promote_numeric(const DataType& a, const DataType& b) {
  if (a.id == TypeId::Decimal || b.id == TypeId::Decimal) {
    if (a.id == TypeId::Decimal && b.id == TypeId::Decimal) {
      if (a.precision == b.precision && a.scale == b.scale) {
        return DataType{a.id, a.nullable || b.nullable, a.precision, a.scale};
      }
      throw BindingError("mixing two DECIMAL types with different precision/scale is not yet supported (" +
                         a.to_string() + " vs. " + b.to_string() + ")");
    }
    const DataType& decimal_side = a.id == TypeId::Decimal ? a : b;
    return DataType{decimal_side.id, a.nullable || b.nullable, decimal_side.precision, decimal_side.scale};
  }
  const bool nullable = a.nullable || b.nullable;
  if (is_floating(a.id) || is_floating(b.id)) return float64_type(nullable);
  if (a.id == TypeId::UInt64 || b.id == TypeId::UInt64) return uint64_type(nullable);
  if (a.id == TypeId::Int64 || b.id == TypeId::Int64) return int64_type(nullable);
  if (a.id == TypeId::UInt32 || b.id == TypeId::UInt32) return uint32_type(nullable);
  return int32_type(nullable);
}

// Names match parser.cpp's to_cast_type_name() (SQL keyword spelling, not
// KernelLake's own TypeId names). GPU support for a given target type is
// enforced later, at GPU-compile time (expression_compiler.cpp only
// supports widening CASTs to INT64/UINT64/FLOAT64) -- accepting the syntax
// here and rejecting unsupported targets downstream, with a clear error
// either way, matches how the rest of the grammar handles "parsed but not
// executable yet" constructs.
DataType resolve_cast_type_name(const AstCast& node, bool nullable) {
  const std::string& name = node.type_name;
  if (name == "BIGINT") return int64_type(nullable);
  if (name == "INT") return int32_type(nullable);
  if (name == "BOOLEAN") return boolean_type(nullable);
  if (name == "DOUBLE") return float64_type(nullable);
  if (name == "FLOAT") return float32_type(nullable);
  if (name == "DATE") return date32_type(nullable);
  if (name == "DATETIME") return timestamp_type(nullable);
  if (name == "VARCHAR") return string_type(nullable);
  if (name == "DECIMAL") {
    // hsql parses a bare `DECIMAL` (no parens) as precision=scale=0; require
    // an explicit precision/scale rather than guessing a default, matching
    // how CAST(... AS VARCHAR) requires an explicit length elsewhere in this
    // grammar.
    if (node.decimal_precision <= 0) {
      throw BindingError("CAST to DECIMAL requires explicit precision and scale, e.g. DECIMAL(10, 2)");
    }
    if (node.decimal_precision > 38) {
      throw BindingError("DECIMAL precision must be between 1 and 38, got " +
                         std::to_string(node.decimal_precision));
    }
    if (node.decimal_scale < 0 || node.decimal_scale > node.decimal_precision) {
      throw BindingError("DECIMAL scale must be between 0 and precision (" +
                         std::to_string(node.decimal_precision) + "), got " +
                         std::to_string(node.decimal_scale));
    }
    return decimal_type(static_cast<std::int32_t>(node.decimal_precision),
                        static_cast<std::int32_t>(node.decimal_scale), nullable);
  }
  throw BindingError("CAST to unknown type '" + name + "'");
}

ExpressionPtr cast_if_needed(ExpressionPtr expr, const DataType& target) {
  const DataType& current = expr->result_type();
  if (current.id == target.id && current.precision == target.precision && current.scale == target.scale) {
    return expr;
  }
  if (target.id == TypeId::Decimal && current.id != TypeId::Decimal) {
    // A numeric literal can be retyped to DECIMAL in place with no
    // precision loss (it's a compile-time constant, not a runtime cast) --
    // this is what lets `WHERE price > 19.99` bind against a DECIMAL
    // `price` column. A column or computed expression on the other side
    // would need a genuine runtime CAST to DECIMAL, which cudf::ast has no
    // operator for (no CAST_TO_DECIMAL*) -- fails clearly instead of being
    // silently misevaluated. See docs/ARCHITECTURE.md.
    if (const auto* literal = dynamic_cast<const LiteralExpression*>(expr.get())) {
      return std::make_shared<LiteralExpression>(literal->value(), target);
    }
    throw BindingError("mixing DECIMAL with a non-literal " + current.to_string() +
                       " is not yet supported -- wrap it in an explicit CAST(... AS DECIMAL(p, s))");
  }
  return std::make_shared<CastExpression>(std::move(expr), target);
}

bool contains_aggregate(const AstExprPtr& expr) {
  return std::visit(
      [](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, AstAggregate>) {
          return true;
        } else if constexpr (std::is_same_v<T, AstBinary>) {
          return contains_aggregate(node.left) || contains_aggregate(node.right);
        } else if constexpr (std::is_same_v<T, AstUnary>) {
          return contains_aggregate(node.operand);
        } else if constexpr (std::is_same_v<T, AstBetween>) {
          return contains_aggregate(node.value) || contains_aggregate(node.lower) ||
                 contains_aggregate(node.upper);
        } else if constexpr (std::is_same_v<T, AstLike>) {
          return contains_aggregate(node.value) || contains_aggregate(node.pattern);
        } else if constexpr (std::is_same_v<T, AstIn>) {
          if (contains_aggregate(node.value)) return true;
          return std::any_of(node.list.begin(), node.list.end(), contains_aggregate);
        } else if constexpr (std::is_same_v<T, AstCase>) {
          for (const auto& [condition, result] : node.when_then) {
            if (contains_aggregate(condition) || contains_aggregate(result)) return true;
          }
          return node.else_branch != nullptr && contains_aggregate(node.else_branch);
        } else if constexpr (std::is_same_v<T, AstCast>) {
          return contains_aggregate(node.operand);
        } else {
          return false;
        }
      },
      expr->node);
}

// Returns true if `expr` (a SELECT-list item in an aggregate query) refers
// to a source column that is neither wrapped in an aggregate nor listed in
// GROUP BY.
bool references_ungrouped_column(const AstExprPtr& expr, const std::vector<std::string>& group_by,
                                 bool inside_aggregate) {
  return std::visit(
      [&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, AstColumnRef>) {
          if (inside_aggregate) return false;
          return std::find(group_by.begin(), group_by.end(), node.name) == group_by.end();
        } else if constexpr (std::is_same_v<T, AstBinary>) {
          return references_ungrouped_column(node.left, group_by, inside_aggregate) ||
                 references_ungrouped_column(node.right, group_by, inside_aggregate);
        } else if constexpr (std::is_same_v<T, AstUnary>) {
          return references_ungrouped_column(node.operand, group_by, inside_aggregate);
        } else if constexpr (std::is_same_v<T, AstBetween>) {
          return references_ungrouped_column(node.value, group_by, inside_aggregate) ||
                 references_ungrouped_column(node.lower, group_by, inside_aggregate) ||
                 references_ungrouped_column(node.upper, group_by, inside_aggregate);
        } else if constexpr (std::is_same_v<T, AstAggregate>) {
          if (node.argument == nullptr) return false;
          return references_ungrouped_column(node.argument, group_by, true);
        } else if constexpr (std::is_same_v<T, AstLike>) {
          return references_ungrouped_column(node.value, group_by, inside_aggregate) ||
                 references_ungrouped_column(node.pattern, group_by, inside_aggregate);
        } else if constexpr (std::is_same_v<T, AstIn>) {
          if (references_ungrouped_column(node.value, group_by, inside_aggregate)) return true;
          return std::any_of(node.list.begin(), node.list.end(), [&](const AstExprPtr& item) {
            return references_ungrouped_column(item, group_by, inside_aggregate);
          });
        } else if constexpr (std::is_same_v<T, AstCase>) {
          for (const auto& [condition, result] : node.when_then) {
            if (references_ungrouped_column(condition, group_by, inside_aggregate) ||
                references_ungrouped_column(result, group_by, inside_aggregate)) {
              return true;
            }
          }
          return node.else_branch != nullptr &&
                 references_ungrouped_column(node.else_branch, group_by, inside_aggregate);
        } else if constexpr (std::is_same_v<T, AstCast>) {
          return references_ungrouped_column(node.operand, group_by, inside_aggregate);
        } else {
          return false;
        }
      },
      expr->node);
}

class Binder {
 public:
  explicit Binder(const Schema& input_schema) : input_schema_(input_schema) {}

  // `allow_aggregates` is true only while binding SELECT-list / ORDER BY
  // expressions; WHERE and GROUP BY must not contain aggregate functions.
  ExpressionPtr bind(const AstExprPtr& expr, bool allow_aggregates) {
    return std::visit([&](const auto& node) -> ExpressionPtr { return bind_node(node, allow_aggregates); },
                      expr->node);
  }

 private:
  ExpressionPtr bind_node(const AstColumnRef& node, bool) {
    const auto index = input_schema_.find_field(node.name);
    if (!index) {
      throw BindingError("unknown column '" + node.name + "'");
    }
    const Field& field = input_schema_.field(*index);
    return std::make_shared<ColumnExpression>(field.name, *index, field.type);
  }

  ExpressionPtr bind_node(const AstStar&, bool) {
    throw BindingError("'*' is only valid as a whole SELECT-list item");
  }

  ExpressionPtr bind_node(const AstLiteral& node, bool) {
    switch (node.kind) {
      case AstLiteralKind::Integer:
        return std::make_shared<LiteralExpression>(LiteralExpression::make_int64(node.int_value));
      case AstLiteralKind::Float:
        return std::make_shared<LiteralExpression>(LiteralExpression::make_float64(node.float_value));
      case AstLiteralKind::String:
        return std::make_shared<LiteralExpression>(LiteralExpression::make_string(node.string_value));
      case AstLiteralKind::Boolean:
        return std::make_shared<LiteralExpression>(LiteralExpression::make_bool(node.bool_value));
      case AstLiteralKind::Date:
        return std::make_shared<LiteralExpression>(
            LiteralExpression::make_date32(static_cast<std::int32_t>(node.int_value)));
      case AstLiteralKind::Null:
        // Untyped NULL: default to a nullable Int64 placeholder. Binary/
        // BETWEEN binding re-types NULL literals to match their sibling
        // before this default would ever surface in a comparison.
        return std::make_shared<LiteralExpression>(LiteralExpression::make_null(int64_type(true)));
    }
    throw BindingError("unreachable literal kind");
  }

  bool is_untyped_null(const AstExprPtr& expr) const {
    const auto* literal = std::get_if<AstLiteral>(&expr->node);
    return literal != nullptr && literal->kind == AstLiteralKind::Null;
  }

  ExpressionPtr bind_node(const AstBinary& node, bool allow_aggregates) {
    // Re-type a bare NULL literal to match its sibling so `x = NULL` and
    // `x > NULL` bind cleanly instead of hitting the Int64 default.
    if (is_untyped_null(node.left) && !is_untyped_null(node.right)) {
      ExpressionPtr right = bind(node.right, allow_aggregates);
      ExpressionPtr left =
          std::make_shared<LiteralExpression>(LiteralExpression::make_null(right->result_type()));
      return combine_binary(node.op, std::move(left), std::move(right));
    }
    if (is_untyped_null(node.right) && !is_untyped_null(node.left)) {
      ExpressionPtr left = bind(node.left, allow_aggregates);
      ExpressionPtr right =
          std::make_shared<LiteralExpression>(LiteralExpression::make_null(left->result_type()));
      return combine_binary(node.op, std::move(left), std::move(right));
    }
    ExpressionPtr left = bind(node.left, allow_aggregates);
    ExpressionPtr right = bind(node.right, allow_aggregates);
    return combine_binary(node.op, std::move(left), std::move(right));
  }

  static BinaryOperator to_binary_operator(AstBinaryOp op) {
    switch (op) {
      case AstBinaryOp::Add:
        return BinaryOperator::Add;
      case AstBinaryOp::Subtract:
        return BinaryOperator::Subtract;
      case AstBinaryOp::Multiply:
        return BinaryOperator::Multiply;
      case AstBinaryOp::Divide:
        return BinaryOperator::Divide;
      case AstBinaryOp::Eq:
        return BinaryOperator::Equal;
      case AstBinaryOp::NotEq:
        return BinaryOperator::NotEqual;
      case AstBinaryOp::Lt:
        return BinaryOperator::Less;
      case AstBinaryOp::LtEq:
        return BinaryOperator::LessEqual;
      case AstBinaryOp::Gt:
        return BinaryOperator::Greater;
      case AstBinaryOp::GtEq:
        return BinaryOperator::GreaterEqual;
      case AstBinaryOp::And:
        return BinaryOperator::And;
      case AstBinaryOp::Or:
        return BinaryOperator::Or;
    }
    throw BindingError("unreachable binary operator");
  }

  ExpressionPtr combine_binary(AstBinaryOp ast_op, ExpressionPtr left, ExpressionPtr right) {
    const BinaryOperator op = to_binary_operator(ast_op);
    const DataType& lt = left->result_type();
    const DataType& rt = right->result_type();
    const bool result_nullable = lt.nullable || rt.nullable;

    if (is_arithmetic(op)) {
      if (!is_numeric(lt.id) || !is_numeric(rt.id)) {
        throw BindingError("arithmetic operator '" + std::string(kernellake::to_string(op)) +
                           "' requires numeric operands, got " + lt.to_string() + " and " + rt.to_string());
      }
      const DataType result_type = promote_numeric(lt, rt);
      return std::make_shared<BinaryExpression>(op, cast_if_needed(std::move(left), result_type),
                                                cast_if_needed(std::move(right), result_type), result_type);
    }

    if (is_logical(op)) {
      if (lt.id != TypeId::Boolean || rt.id != TypeId::Boolean) {
        throw BindingError("AND/OR require boolean operands, got " + lt.to_string() + " and " +
                           rt.to_string());
      }
      return std::make_shared<BinaryExpression>(op, std::move(left), std::move(right),
                                                boolean_type(result_nullable));
    }

    // Comparison. DECIMAL additionally needs precision/scale to match, not
    // just `id` -- two DECIMALs with different precision/scale otherwise
    // slip past this early-return path without ever reaching
    // promote_numeric()'s mismatched-DECIMAL rejection below.
    if (lt.id == rt.id && lt.precision == rt.precision && lt.scale == rt.scale) {
      return std::make_shared<BinaryExpression>(op, std::move(left), std::move(right),
                                                boolean_type(result_nullable));
    }
    if (is_numeric(lt.id) && is_numeric(rt.id)) {
      const DataType common = promote_numeric(lt, rt);
      return std::make_shared<BinaryExpression>(op, cast_if_needed(std::move(left), common),
                                                cast_if_needed(std::move(right), common),
                                                boolean_type(result_nullable));
    }
    throw BindingError("incompatible comparison between " + lt.to_string() + " and " + rt.to_string());
  }

  ExpressionPtr bind_node(const AstUnary& node, bool allow_aggregates) {
    ExpressionPtr operand = bind(node.operand, allow_aggregates);
    const DataType& operand_type = operand->result_type();
    switch (node.op) {
      case AstUnaryOp::Not:
        if (operand_type.id != TypeId::Boolean) {
          throw BindingError("NOT requires a boolean operand, got " + operand_type.to_string());
        }
        return std::make_shared<UnaryExpression>(UnaryOperator::Not, std::move(operand), operand_type);
      case AstUnaryOp::Negate:
        if (!is_numeric(operand_type.id)) {
          throw BindingError("unary '-' requires a numeric operand, got " + operand_type.to_string());
        }
        return std::make_shared<UnaryExpression>(UnaryOperator::Negate, std::move(operand), operand_type);
      case AstUnaryOp::IsNull:
        return std::make_shared<UnaryExpression>(UnaryOperator::IsNull, std::move(operand),
                                                 boolean_type(false));
      case AstUnaryOp::IsNotNull:
        return std::make_shared<UnaryExpression>(UnaryOperator::IsNotNull, std::move(operand),
                                                 boolean_type(false));
    }
    throw BindingError("unreachable unary operator");
  }

  ExpressionPtr bind_node(const AstBetween& node, bool allow_aggregates) {
    ExpressionPtr value = bind(node.value, allow_aggregates);
    ExpressionPtr lower = bind(node.lower, allow_aggregates);
    ExpressionPtr upper = bind(node.upper, allow_aggregates);

    auto unify = [&](ExpressionPtr bound, const char* side) {
      const DataType& vt = value->result_type();
      const DataType& bt = bound->result_type();
      if (vt.id == bt.id && vt.precision == bt.precision && vt.scale == bt.scale) return bound;
      if (is_numeric(vt.id) && is_numeric(bt.id)) {
        return cast_if_needed(std::move(bound), promote_numeric(vt, bt));
      }
      throw BindingError(std::string("BETWEEN ") + side + " bound type " + bt.to_string() +
                         " is incompatible with value type " + vt.to_string());
    };
    lower = unify(std::move(lower), "lower");
    upper = unify(std::move(upper), "upper");
    if (is_numeric(value->result_type().id)) {
      const DataType common =
          promote_numeric(promote_numeric(value->result_type(), lower->result_type()), upper->result_type());
      value = cast_if_needed(std::move(value), common);
      lower = cast_if_needed(std::move(lower), common);
      upper = cast_if_needed(std::move(upper), common);
    }
    return std::make_shared<BetweenExpression>(std::move(value), std::move(lower), std::move(upper));
  }

  ExpressionPtr bind_node(const AstAggregate& node, bool allow_aggregates) {
    if (!allow_aggregates) {
      throw BindingError("aggregate functions are not allowed here (WHERE/GROUP BY)");
    }
    if (node.function == AstAggregateFunc::CountStar) {
      return std::make_shared<AggregateExpression>(AggregateFunction::CountStar, nullptr, int64_type(false));
    }
    if (node.argument != nullptr && contains_aggregate(node.argument)) {
      throw BindingError("aggregate functions cannot be nested");
    }
    // Aggregate arguments reference raw source columns, never other
    // aggregates, so binding continues with allow_aggregates=false.
    ExpressionPtr argument = bind(node.argument, false);
    const DataType& arg_type = argument->result_type();

    switch (node.function) {
      case AstAggregateFunc::Count:
        return std::make_shared<AggregateExpression>(AggregateFunction::Count, std::move(argument),
                                                     int64_type(false));
      case AstAggregateFunc::Sum: {
        if (!is_numeric(arg_type.id)) {
          throw BindingError("SUM requires a numeric argument, got " + arg_type.to_string());
        }
        // DECIMAL keeps its own precision/scale unchanged (like MIN/MAX
        // below) rather than being cast to a wider type first: cudf's own
        // SUM aggregation evaluates DECIMAL columns natively and preserves
        // scale, and casting *to* DECIMAL inside a compiled expression isn't
        // possible anyway (no CAST_TO_DECIMAL* in cudf::ast).
        if (arg_type.id == TypeId::Decimal) {
          const DataType result_type{TypeId::Decimal, true, arg_type.precision, arg_type.scale};
          return std::make_shared<AggregateExpression>(AggregateFunction::Sum, std::move(argument),
                                                       result_type);
        }
        const DataType result_type =
            is_floating(arg_type.id) ? float64_type(true)
                                     : (arg_type.id == TypeId::UInt64 ? uint64_type(true) : int64_type(true));
        argument = cast_if_needed(std::move(argument), result_type);
        return std::make_shared<AggregateExpression>(AggregateFunction::Sum, std::move(argument),
                                                     result_type);
      }
      case AstAggregateFunc::Avg: {
        if (!is_numeric(arg_type.id)) {
          throw BindingError("AVG requires a numeric argument, got " + arg_type.to_string());
        }
        if (arg_type.id == TypeId::Decimal) {
          throw BindingError("AVG over DECIMAL is not yet supported -- CAST the argument to DOUBLE first");
        }
        return std::make_shared<AggregateExpression>(AggregateFunction::Avg, std::move(argument),
                                                     float64_type(true));
      }
      case AstAggregateFunc::Min:
        return std::make_shared<AggregateExpression>(
            AggregateFunction::Min, std::move(argument),
            DataType{arg_type.id, true, arg_type.precision, arg_type.scale});
      case AstAggregateFunc::Max:
        return std::make_shared<AggregateExpression>(
            AggregateFunction::Max, std::move(argument),
            DataType{arg_type.id, true, arg_type.precision, arg_type.scale});
      case AstAggregateFunc::CountStar:
        break;  // handled above
    }
    throw BindingError("unreachable aggregate function");
  }

  ExpressionPtr bind_node(const AstLike& node, bool allow_aggregates) {
    ExpressionPtr value = bind(node.value, allow_aggregates);
    if (value->result_type().id != TypeId::String) {
      throw BindingError("LIKE requires a STRING operand, got " + value->result_type().to_string());
    }
    // The pattern must be a compile-time string constant -- a per-row
    // pattern column would need cudf::strings::like's column-of-patterns
    // overload, which is not wired up here (see kernellake/execution's
    // LIKE handling).
    const auto* pattern_literal = std::get_if<AstLiteral>(&node.pattern->node);
    if (pattern_literal == nullptr || pattern_literal->kind != AstLiteralKind::String) {
      throw BindingError("LIKE pattern must be a string literal");
    }
    return std::make_shared<LikeExpression>(std::move(value), pattern_literal->string_value, node.negated);
  }

  ExpressionPtr bind_node(const AstIn& node, bool allow_aggregates) {
    if (node.list.empty()) throw BindingError("IN requires at least one value");
    ExpressionPtr value = bind(node.value, allow_aggregates);
    ExpressionPtr result;
    for (const AstExprPtr& item : node.list) {
      ExpressionPtr comparison = combine_binary(AstBinaryOp::Eq, value, bind(item, allow_aggregates));
      result = result == nullptr ? std::move(comparison)
                                 : combine_binary(AstBinaryOp::Or, std::move(result), std::move(comparison));
    }
    if (!node.negated) return result;
    return std::make_shared<UnaryExpression>(UnaryOperator::Not, std::move(result), boolean_type(false));
  }

  ExpressionPtr bind_node(const AstCase& node, bool allow_aggregates) {
    std::vector<std::pair<ExpressionPtr, ExpressionPtr>> branches;
    branches.reserve(node.when_then.size());
    for (const auto& [condition, then_result] : node.when_then) {
      ExpressionPtr bound_condition = bind(condition, allow_aggregates);
      if (bound_condition->result_type().id != TypeId::Boolean) {
        throw BindingError("CASE WHEN condition must be boolean, got " +
                           bound_condition->result_type().to_string());
      }
      branches.emplace_back(std::move(bound_condition), bind(then_result, allow_aggregates));
    }
    ExpressionPtr bound_else =
        node.else_branch != nullptr ? bind(node.else_branch, allow_aggregates) : nullptr;

    // A CASE with no ELSE evaluates to NULL when no WHEN matches, so the
    // result is nullable even if every branch itself is not.
    DataType common = branches.front().second->result_type();
    bool any_nullable = common.nullable || bound_else == nullptr;
    auto unify = [&](const DataType& branch_type) {
      any_nullable = any_nullable || branch_type.nullable;
      if (branch_type.id == common.id) return;
      if (is_numeric(common.id) && is_numeric(branch_type.id)) {
        common = promote_numeric(common, branch_type);
        return;
      }
      throw BindingError("CASE branches have incompatible result types: " + common.to_string() + " and " +
                         branch_type.to_string());
    };
    for (std::size_t i = 1; i < branches.size(); ++i) unify(branches[i].second->result_type());
    if (bound_else != nullptr) unify(bound_else->result_type());
    common.nullable = any_nullable;

    std::vector<CaseExpression::WhenThen> final_branches;
    final_branches.reserve(branches.size());
    for (auto& [condition, then_result] : branches) {
      final_branches.push_back(
          CaseExpression::WhenThen{std::move(condition), cast_if_needed(std::move(then_result), common)});
    }
    ExpressionPtr final_else =
        bound_else != nullptr ? cast_if_needed(std::move(bound_else), common) : nullptr;
    return std::make_shared<CaseExpression>(std::move(final_branches), std::move(final_else), common);
  }

  ExpressionPtr bind_node(const AstCast& node, bool allow_aggregates) {
    ExpressionPtr operand = bind(node.operand, allow_aggregates);
    const DataType target = resolve_cast_type_name(node, operand->result_type().nullable);
    return std::make_shared<CastExpression>(std::move(operand), target);
  }

  const Schema& input_schema_;
};

}  // namespace

BoundQuery bind_query(const sql::AstSelectStatement& stmt, const Schema& input_schema) {
  for (const AstExprPtr& item : stmt.group_by) {
    if (!std::holds_alternative<AstColumnRef>(item->node)) {
      throw BindingError("GROUP BY expressions must be plain column references in this version");
    }
  }

  const bool is_aggregate_query =
      !stmt.group_by.empty() ||
      std::any_of(stmt.select_list.begin(), stmt.select_list.end(), contains_aggregate);

  Binder binder(input_schema);

  BoundQuery result;
  result.source_paths = stmt.from.paths;
  result.is_aggregate_query = is_aggregate_query;

  std::vector<Field> output_fields;
  std::unordered_set<std::string> seen_names;
  // Maps a SELECT-list output name to its position, so GROUP BY can
  // reference a computed expression by its alias (e.g. `SELECT CASE ... END
  // AS bucket ... GROUP BY bucket`) -- there is no other way to name a
  // computed, non-column GROUP BY key. Resolved below, once every SELECT
  // item is bound.
  std::unordered_map<std::string, std::size_t> alias_to_select_index;

  auto add_select_item = [&](ExpressionPtr expr, std::string name) {
    if (!seen_names.insert(name).second) {
      throw BindingError("duplicate output column name '" + name + "'");
    }
    alias_to_select_index[name] = result.select_list.size();
    output_fields.push_back(Field{name, expr->result_type()});
    result.select_list.push_back(BoundSelectItem{expr, std::move(name)});
  };

  // The "referenced in SELECT list must appear in GROUP BY" check is
  // deferred to a second pass below (once GROUP BY's alias references are
  // resolved against this SELECT list) rather than interleaved here, since
  // it needs the *final* set of grouped names, not just the raw column
  // references written directly in the GROUP BY clause.
  for (const AstExprPtr& item : stmt.select_list) {
    if (std::holds_alternative<AstStar>(item->node)) {
      if (stmt.select_list.size() != 1) {
        throw BindingError("'*' cannot be combined with other SELECT-list items");
      }
      for (const Field& field : input_schema.fields()) {
        add_select_item(
            std::make_shared<ColumnExpression>(field.name, *input_schema.find_field(field.name), field.type),
            field.name);
      }
      continue;
    }

    ExpressionPtr bound = binder.bind(item, /*allow_aggregates=*/true);
    std::string name = item->alias.value_or(std::holds_alternative<AstColumnRef>(item->node)
                                                ? std::get<AstColumnRef>(item->node).name
                                                : bound->to_string());
    add_select_item(std::move(bound), std::move(name));
  }

  result.output_schema = Schema(std::move(output_fields));

  if (stmt.where != nullptr) {
    if (contains_aggregate(stmt.where)) {
      throw BindingError("aggregate functions are not allowed in WHERE");
    }
    ExpressionPtr where = binder.bind(stmt.where, /*allow_aggregates=*/false);
    if (where->result_type().id != TypeId::Boolean) {
      throw BindingError("WHERE clause must be a boolean expression, got " +
                         where->result_type().to_string());
    }
    result.where = std::move(where);
  }

  // A GROUP BY name is resolved against the base table first (matching a
  // real column takes priority over a same-named alias, consistent with
  // how most SQL engines resolve this ambiguity) and falls back to a
  // SELECT-list alias -- the only way to GROUP BY a computed expression
  // like a CASE, since it has no column name of its own to repeat.
  std::vector<std::string> group_by_names;
  // The SELECT item that *defines* an alias-referenced GROUP BY key (e.g.
  // the CASE itself in `... AS bucket ... GROUP BY bucket`) is exempt from
  // the ungrouped-column check below: it likely references raw columns
  // (e.g. `amount` inside the CASE) that aren't individually listed in
  // group_by_names, but the expression *as a whole* is exactly what's
  // being grouped on, which is what actually matters.
  std::unordered_set<std::size_t> select_indices_used_as_group_by_alias;
  for (const AstExprPtr& item : stmt.group_by) {
    const std::string& name = std::get<AstColumnRef>(item->node).name;
    if (input_schema.find_field(name)) {
      result.group_by.push_back(binder.bind(item, /*allow_aggregates=*/false));
    } else if (const auto alias = alias_to_select_index.find(name); alias != alias_to_select_index.end()) {
      result.group_by.push_back(result.select_list[alias->second].expr);
      select_indices_used_as_group_by_alias.insert(alias->second);
    } else {
      throw BindingError("unknown column '" + name + "' in GROUP BY");
    }
    group_by_names.push_back(name);
  }

  if (is_aggregate_query) {
    for (std::size_t i = 0; i < stmt.select_list.size(); ++i) {
      const AstExprPtr& item = stmt.select_list[i];
      if (std::holds_alternative<AstStar>(item->node)) continue;  // handled during expansion above
      if (select_indices_used_as_group_by_alias.count(i) != 0) continue;
      if (references_ungrouped_column(item, group_by_names, /*inside_aggregate=*/false)) {
        throw BindingError(
            "column referenced in SELECT list must appear in GROUP BY or be used inside an "
            "aggregate function");
      }
    }
  }

  for (const sql::AstOrderByItem& item : stmt.order_by) {
    if (is_aggregate_query) {
      // Binding against the base-table schema (like the non-aggregate case
      // below) can't work here: GROUP BY/aggregate queries can ORDER BY an
      // aggregate alias ("ORDER BY total") that doesn't exist as a column
      // until after aggregation. Scoped deliberately to plain references to
      // an existing SELECT-list output name (by far the common case, and
      // exactly what the physical Sort operator receives, since it runs
      // after the final projection) rather than arbitrary re-derived
      // expressions -- see docs/ARCHITECTURE.md.
      const auto* column = std::get_if<AstColumnRef>(&item.expr->node);
      if (column == nullptr) {
        throw BindingError(
            "ORDER BY after GROUP BY only supports a plain SELECT-list output name in this "
            "version of KernelLake");
      }
      const std::optional<std::size_t> index = result.output_schema.find_field(column->name);
      if (!index) {
        throw BindingError("ORDER BY after GROUP BY: '" + column->name +
                           "' is not one of this query's output columns");
      }
      const Field& field = result.output_schema.field(*index);
      result.order_by.push_back(BoundOrderByItem{
          std::make_shared<ColumnExpression>(field.name, *index, field.type), item.ascending});
    } else {
      result.order_by.push_back(
          BoundOrderByItem{binder.bind(item.expr, /*allow_aggregates=*/true), item.ascending});
    }
  }

  result.limit = stmt.limit;
  return result;
}

}  // namespace kernellake
