#include "kernellake/planner/binder.hpp"

#include <algorithm>
#include <unordered_set>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

using sql::AstAggregate;
using sql::AstAggregateFunc;
using sql::AstBetween;
using sql::AstBinary;
using sql::AstBinaryOp;
using sql::AstColumnRef;
using sql::AstExpr;
using sql::AstExprPtr;
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

bool is_floating(TypeId id) { return id == TypeId::Float32 || id == TypeId::Float64; }

// Picks the smallest KernelLake numeric type that can represent both inputs
// without loss for the common cases KernelLake cares about (int/int and
// int/float promotion). Decimal-with-decimal and decimal-with-float mixes
// are rejected rather than guessed at, since choosing a safe precision/scale
// automatically is not yet implemented.
DataType promote_numeric(const DataType& a, const DataType& b) {
  if (a.id == TypeId::Decimal || b.id == TypeId::Decimal) {
    if (a.id == b.id && a.precision == b.precision && a.scale == b.scale) return a;
    throw BindingError("mixing DECIMAL with other numeric types is not yet supported");
  }
  const bool nullable = a.nullable || b.nullable;
  if (is_floating(a.id) || is_floating(b.id)) return float64_type(nullable);
  if (a.id == TypeId::UInt64 || b.id == TypeId::UInt64) return uint64_type(nullable);
  if (a.id == TypeId::Int64 || b.id == TypeId::Int64) return int64_type(nullable);
  if (a.id == TypeId::UInt32 || b.id == TypeId::UInt32) return uint32_type(nullable);
  return int32_type(nullable);
}

ExpressionPtr cast_if_needed(ExpressionPtr expr, const DataType& target) {
  if (expr->result_type().id == target.id) return expr;
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
    return std::visit(
        [&](const auto& node) -> ExpressionPtr { return bind_node(node, allow_aggregates); },
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
        return std::make_shared<LiteralExpression>(
            LiteralExpression::make_float64(node.float_value));
      case AstLiteralKind::String:
        return std::make_shared<LiteralExpression>(
            LiteralExpression::make_string(node.string_value));
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
                            "' requires numeric operands, got " + lt.to_string() + " and " +
                            rt.to_string());
      }
      const DataType result_type = promote_numeric(lt, rt);
      return std::make_shared<BinaryExpression>(op, cast_if_needed(std::move(left), result_type),
                                                 cast_if_needed(std::move(right), result_type),
                                                 result_type);
    }

    if (is_logical(op)) {
      if (lt.id != TypeId::Boolean || rt.id != TypeId::Boolean) {
        throw BindingError("AND/OR require boolean operands, got " + lt.to_string() + " and " +
                            rt.to_string());
      }
      return std::make_shared<BinaryExpression>(op, std::move(left), std::move(right),
                                                 boolean_type(result_nullable));
    }

    // Comparison.
    if (lt.id == rt.id) {
      return std::make_shared<BinaryExpression>(op, std::move(left), std::move(right),
                                                 boolean_type(result_nullable));
    }
    if (is_numeric(lt.id) && is_numeric(rt.id)) {
      const DataType common = promote_numeric(lt, rt);
      return std::make_shared<BinaryExpression>(op, cast_if_needed(std::move(left), common),
                                                 cast_if_needed(std::move(right), common),
                                                 boolean_type(result_nullable));
    }
    throw BindingError("incompatible comparison between " + lt.to_string() + " and " +
                        rt.to_string());
  }

  ExpressionPtr bind_node(const AstUnary& node, bool allow_aggregates) {
    ExpressionPtr operand = bind(node.operand, allow_aggregates);
    const DataType& operand_type = operand->result_type();
    switch (node.op) {
      case AstUnaryOp::Not:
        if (operand_type.id != TypeId::Boolean) {
          throw BindingError("NOT requires a boolean operand, got " + operand_type.to_string());
        }
        return std::make_shared<UnaryExpression>(UnaryOperator::Not, std::move(operand),
                                                  operand_type);
      case AstUnaryOp::Negate:
        if (!is_numeric(operand_type.id)) {
          throw BindingError("unary '-' requires a numeric operand, got " +
                              operand_type.to_string());
        }
        return std::make_shared<UnaryExpression>(UnaryOperator::Negate, std::move(operand),
                                                  operand_type);
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
      if (vt.id == bt.id) return bound;
      if (is_numeric(vt.id) && is_numeric(bt.id)) {
        return cast_if_needed(std::move(bound), promote_numeric(vt, bt));
      }
      throw BindingError(std::string("BETWEEN ") + side + " bound type " + bt.to_string() +
                          " is incompatible with value type " + vt.to_string());
    };
    lower = unify(std::move(lower), "lower");
    upper = unify(std::move(upper), "upper");
    if (is_numeric(value->result_type().id)) {
      const DataType common = promote_numeric(promote_numeric(value->result_type(),
                                                                lower->result_type()),
                                               upper->result_type());
      value = cast_if_needed(std::move(value), common);
      lower = cast_if_needed(std::move(lower), common);
      upper = cast_if_needed(std::move(upper), common);
    }
    return std::make_shared<BetweenExpression>(std::move(value), std::move(lower),
                                                std::move(upper));
  }

  ExpressionPtr bind_node(const AstAggregate& node, bool allow_aggregates) {
    if (!allow_aggregates) {
      throw BindingError("aggregate functions are not allowed here (WHERE/GROUP BY)");
    }
    if (node.function == AstAggregateFunc::CountStar) {
      return std::make_shared<AggregateExpression>(AggregateFunction::CountStar, nullptr,
                                                     int64_type(false));
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
        return std::make_shared<AggregateExpression>(AggregateFunction::Count,
                                                       std::move(argument), int64_type(false));
      case AstAggregateFunc::Sum: {
        if (!is_numeric(arg_type.id)) {
          throw BindingError("SUM requires a numeric argument, got " + arg_type.to_string());
        }
        const DataType result_type = is_floating(arg_type.id) ? float64_type(true)
                                      : (arg_type.id == TypeId::UInt64 ? uint64_type(true)
                                                                        : int64_type(true));
        argument = cast_if_needed(std::move(argument), result_type);
        return std::make_shared<AggregateExpression>(AggregateFunction::Sum, std::move(argument),
                                                       result_type);
      }
      case AstAggregateFunc::Avg: {
        if (!is_numeric(arg_type.id)) {
          throw BindingError("AVG requires a numeric argument, got " + arg_type.to_string());
        }
        return std::make_shared<AggregateExpression>(AggregateFunction::Avg, std::move(argument),
                                                       float64_type(true));
      }
      case AstAggregateFunc::Min:
        return std::make_shared<AggregateExpression>(AggregateFunction::Min, std::move(argument),
                                                       DataType{arg_type.id, true,
                                                                arg_type.precision,
                                                                arg_type.scale});
      case AstAggregateFunc::Max:
        return std::make_shared<AggregateExpression>(AggregateFunction::Max, std::move(argument),
                                                       DataType{arg_type.id, true,
                                                                arg_type.precision,
                                                                arg_type.scale});
      case AstAggregateFunc::CountStar:
        break;  // handled above
    }
    throw BindingError("unreachable aggregate function");
  }

  const Schema& input_schema_;
};

}  // namespace

BoundQuery bind_query(const sql::AstSelectStatement& stmt, const Schema& input_schema) {
  std::vector<std::string> group_by_names;
  for (const AstExprPtr& item : stmt.group_by) {
    if (const auto* column = std::get_if<AstColumnRef>(&item->node)) {
      group_by_names.push_back(column->name);
    } else {
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

  auto add_select_item = [&](ExpressionPtr expr, std::string name) {
    if (!seen_names.insert(name).second) {
      throw BindingError("duplicate output column name '" + name + "'");
    }
    output_fields.push_back(Field{name, expr->result_type()});
    result.select_list.push_back(BoundSelectItem{expr, std::move(name)});
  };

  for (const AstExprPtr& item : stmt.select_list) {
    if (std::holds_alternative<AstStar>(item->node)) {
      if (stmt.select_list.size() != 1) {
        throw BindingError("'*' cannot be combined with other SELECT-list items");
      }
      for (const Field& field : input_schema.fields()) {
        add_select_item(
            std::make_shared<ColumnExpression>(field.name, *input_schema.find_field(field.name),
                                                field.type),
            field.name);
      }
      continue;
    }

    if (is_aggregate_query &&
        references_ungrouped_column(item, group_by_names, /*inside_aggregate=*/false)) {
      throw BindingError(
          "column referenced in SELECT list must appear in GROUP BY or be used inside an "
          "aggregate function");
    }

    ExpressionPtr bound = binder.bind(item, /*allow_aggregates=*/true);
    std::string name = item->alias.value_or(
        std::holds_alternative<AstColumnRef>(item->node) ? std::get<AstColumnRef>(item->node).name
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

  for (const AstExprPtr& item : stmt.group_by) {
    result.group_by.push_back(binder.bind(item, /*allow_aggregates=*/false));
  }

  for (const sql::AstOrderByItem& item : stmt.order_by) {
    result.order_by.push_back(
        BoundOrderByItem{binder.bind(item.expr, /*allow_aggregates=*/true), item.ascending});
  }

  result.limit = stmt.limit;
  return result;
}

}  // namespace kernellake
