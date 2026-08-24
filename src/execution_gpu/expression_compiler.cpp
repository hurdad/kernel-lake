#include "kernellake/execution_gpu/expression_compiler.hpp"

#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <fmt/format.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cudf_adapter.hpp"

namespace kernellake {

namespace {

double as_double(const LiteralStorage& value) {
  if (std::holds_alternative<std::int64_t>(value)) return static_cast<double>(std::get<std::int64_t>(value));
  if (std::holds_alternative<double>(value)) return std::get<double>(value);
  if (std::holds_alternative<bool>(value)) return std::get<bool>(value) ? 1.0 : 0.0;
  return 0.0;
}

std::int64_t as_int64(const LiteralStorage& value) {
  if (std::holds_alternative<std::int64_t>(value)) return std::get<std::int64_t>(value);
  if (std::holds_alternative<double>(value)) return static_cast<std::int64_t>(std::get<double>(value));
  if (std::holds_alternative<bool>(value)) return std::get<bool>(value) ? 1 : 0;
  return 0;
}

}  // namespace

const cudf::ast::expression& ExpressionCompiler::compile(const Expression& expr, ExecutionContext& context) {
  if (const auto* column = dynamic_cast<const ColumnExpression*>(&expr)) return compile_column(*column);
  if (const auto* literal = dynamic_cast<const LiteralExpression*>(&expr)) {
    return compile_literal(*literal, context);
  }
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(&expr))
    return compile_binary(*binary, context);
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(&expr)) return compile_unary(*unary, context);
  if (const auto* between = dynamic_cast<const BetweenExpression*>(&expr)) {
    return compile_between(*between, context);
  }
  if (const auto* cast = dynamic_cast<const CastExpression*>(&expr)) return compile_cast(*cast, context);
  if (dynamic_cast<const AggregateExpression*>(&expr) != nullptr) {
    throw ExecutionError(
        "aggregate expressions cannot be compiled as a row-wise GPU expression; they are "
        "evaluated separately by the aggregate operators");
  }
  throw ExecutionError("unrecognized expression type in GPU expression compiler");
}

const cudf::ast::expression& ExpressionCompiler::compile_column(const ColumnExpression& expr) {
  return tree_.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(expr.column_index()));
}

const cudf::ast::expression& ExpressionCompiler::make_literal(const DataType& type,
                                                              const LiteralStorage& value, bool is_valid,
                                                              ExecutionContext& context) {
  switch (type.id) {
    case TypeId::Boolean: {
      auto scalar = std::make_unique<cudf::numeric_scalar<bool>>(
          std::holds_alternative<bool>(value) && std::get<bool>(value), is_valid, context.stream,
          context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::Int32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<std::int32_t>>(
          static_cast<std::int32_t>(as_int64(value)), is_valid, context.stream, context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::Int64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<std::int64_t>>(
          as_int64(value), is_valid, context.stream, context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::UInt32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<std::uint32_t>>(
          static_cast<std::uint32_t>(as_int64(value)), is_valid, context.stream, context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::UInt64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<std::uint64_t>>(
          static_cast<std::uint64_t>(as_int64(value)), is_valid, context.stream, context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::Float32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<float>>(
          static_cast<float>(as_double(value)), is_valid, context.stream, context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::Float64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<double>>(as_double(value), is_valid, context.stream,
                                                                   context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::String: {
      const std::string text = std::holds_alternative<std::string>(value) ? std::get<std::string>(value) : "";
      auto scalar =
          std::make_unique<cudf::string_scalar>(text, is_valid, context.stream, context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::Date32: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
          cudf::duration_D{static_cast<std::int32_t>(as_int64(value))}, is_valid, context.stream,
          context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::Timestamp: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
          cudf::duration_us{as_int64(value)}, is_valid, context.stream, context.memory_resource);
      auto& ref = *scalar;
      scalars_.push_back(std::move(scalar));
      return tree_.emplace<cudf::ast::literal>(ref);
    }
    case TypeId::Decimal: {
      // cudf::ast::literal's constructor is templated on the *concrete*
      // fixed_point_scalar<T> (no type-erased overload exists), unlike the
      // rest of this switch's cudf::scalar-based cases -- can't reuse
      // cudf_adapter's make_decimal_scalar() here since it type-erases too
      // early; decimal_raw_value() gives the same raw/scale computation
      // without erasing the concrete type first.
      const DecimalRawValue raw_value = decimal_raw_value(type, value);
      const numeric::scale_type scale{raw_value.cudf_scale};
      switch (raw_value.type_id) {
        case cudf::type_id::DECIMAL32: {
          auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
              static_cast<std::int32_t>(raw_value.raw), scale, is_valid, context.stream,
              context.memory_resource);
          auto& ref = *scalar;
          scalars_.push_back(std::move(scalar));
          return tree_.emplace<cudf::ast::literal>(ref);
        }
        case cudf::type_id::DECIMAL64: {
          auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
              static_cast<std::int64_t>(raw_value.raw), scale, is_valid, context.stream,
              context.memory_resource);
          auto& ref = *scalar;
          scalars_.push_back(std::move(scalar));
          return tree_.emplace<cudf::ast::literal>(ref);
        }
        default: {
          auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
              raw_value.raw, scale, is_valid, context.stream, context.memory_resource);
          auto& ref = *scalar;
          scalars_.push_back(std::move(scalar));
          return tree_.emplace<cudf::ast::literal>(ref);
        }
      }
    }
  }
  throw ExecutionError("unreachable: unknown KernelLake TypeId in expression compiler");
}

const cudf::ast::expression& ExpressionCompiler::compile_literal(const LiteralExpression& expr,
                                                                 ExecutionContext& context) {
  return make_literal(expr.result_type(), expr.value(), !expr.is_null(), context);
}

namespace {
cudf::ast::ast_operator to_ast_operator(BinaryOperator op) {
  switch (op) {
    case BinaryOperator::Add:
      return cudf::ast::ast_operator::ADD;
    case BinaryOperator::Subtract:
      return cudf::ast::ast_operator::SUB;
    case BinaryOperator::Multiply:
      return cudf::ast::ast_operator::MUL;
    case BinaryOperator::Divide:
      return cudf::ast::ast_operator::DIV;
    case BinaryOperator::Equal:
      return cudf::ast::ast_operator::EQUAL;
    case BinaryOperator::NotEqual:
      return cudf::ast::ast_operator::NOT_EQUAL;
    case BinaryOperator::Less:
      return cudf::ast::ast_operator::LESS;
    case BinaryOperator::LessEqual:
      return cudf::ast::ast_operator::LESS_EQUAL;
    case BinaryOperator::Greater:
      return cudf::ast::ast_operator::GREATER;
    case BinaryOperator::GreaterEqual:
      return cudf::ast::ast_operator::GREATER_EQUAL;
    case BinaryOperator::And:
      // NULL_LOGICAL_AND, not LOGICAL_AND: SQL's three-valued (Kleene) logic
      // requires FALSE AND NULL = FALSE (not NULL) so that e.g. `WHERE x > 5
      // AND y IS NULL` still excludes rows where x is NULL. Plain
      // LOGICAL_AND/LOGICAL_OR propagate NULL whenever *either* operand is
      // null, with no special-casing of a definitively-FALSE/TRUE operand --
      // see cudf/ast/detail/operator_functor.cuh's own NULL_LOGICAL_AND/
      // NULL_LOGICAL_OR doc comments for the exact truth tables.
      return cudf::ast::ast_operator::NULL_LOGICAL_AND;
    case BinaryOperator::Or:
      // NULL_LOGICAL_OR, not LOGICAL_OR: same Kleene-logic reasoning as AND
      // above, mirrored -- TRUE OR NULL = TRUE (not NULL), which is what
      // makes `WHERE x IS NULL OR x = 'Bob'` correctly include NULL rows.
      return cudf::ast::ast_operator::NULL_LOGICAL_OR;
  }
  throw ExecutionError("unreachable: unknown BinaryOperator in expression compiler");
}
}  // namespace

const cudf::ast::expression& ExpressionCompiler::compile_binary(const BinaryExpression& expr,
                                                                ExecutionContext& context) {
  const cudf::ast::expression& left = compile(*expr.left(), context);
  const cudf::ast::expression& right = compile(*expr.right(), context);
  return tree_.emplace<cudf::ast::operation>(to_ast_operator(expr.op()), left, right);
}

const cudf::ast::expression& ExpressionCompiler::compile_unary(const UnaryExpression& expr,
                                                               ExecutionContext& context) {
  const cudf::ast::expression& operand = compile(*expr.operand(), context);
  switch (expr.op()) {
    case UnaryOperator::Not:
      return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, operand);
    case UnaryOperator::IsNull:
      return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, operand);
    case UnaryOperator::IsNotNull: {
      const cudf::ast::expression& is_null =
          tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, operand);
      return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, is_null);
    }
    case UnaryOperator::Negate: {
      // cudf::ast has no dedicated unary negation operator; synthesize it
      // as (0 - x), matching the operand's own type.
      const DataType& type = expr.operand()->result_type();
      const cudf::ast::expression& zero = make_literal(type, LiteralStorage{std::int64_t{0}}, true, context);
      return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::SUB, zero, operand);
    }
  }
  throw ExecutionError("unreachable: unknown UnaryOperator in expression compiler");
}

const cudf::ast::expression& ExpressionCompiler::compile_between(const BetweenExpression& expr,
                                                                 ExecutionContext& context) {
  const cudf::ast::expression& value = compile(*expr.value(), context);
  const cudf::ast::expression& lower = compile(*expr.lower(), context);
  const cudf::ast::expression& upper = compile(*expr.upper(), context);
  const cudf::ast::expression& ge =
      tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::GREATER_EQUAL, value, lower);
  const cudf::ast::expression& le =
      tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LESS_EQUAL, value, upper);
  return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LOGICAL_AND, ge, le);
}

const cudf::ast::expression& ExpressionCompiler::compile_cast(const CastExpression& expr,
                                                              ExecutionContext& context) {
  const cudf::ast::expression& operand = compile(*expr.operand(), context);
  switch (expr.result_type().id) {
    case TypeId::Int64:
      return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_INT64, operand);
    case TypeId::UInt64:
      return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_UINT64, operand);
    case TypeId::Float64:
      return tree_.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_FLOAT64, operand);
    default:
      throw ExecutionError(
          fmt::format("CAST to {} is not supported for GPU row-wise expressions (cudf::ast only "
                      "supports widening casts to INT64/UINT64/FLOAT64)",
                      expr.result_type().to_string()));
  }
}

}  // namespace kernellake
