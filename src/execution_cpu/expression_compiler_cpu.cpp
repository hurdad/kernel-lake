#include "kernellake/execution_cpu/expression_compiler_cpu.hpp"

#include <arrow/compute/api_scalar.h>
#include <arrow/compute/cast.h>
#include <arrow/scalar.h>
#include <fmt/format.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/types/arrow_adapter.hpp"

namespace kernellake {

namespace {

double as_double(const LiteralStorage& value) {
  if (std::holds_alternative<std::int64_t>(value)) {
    return static_cast<double>(std::get<std::int64_t>(value));
  }
  if (std::holds_alternative<double>(value)) {
    return std::get<double>(value);
  }
  if (std::holds_alternative<bool>(value)) {
    return std::get<bool>(value) ? 1.0 : 0.0;
  }
  return 0.0;
}

std::int64_t as_int64(const LiteralStorage& value) {
  if (std::holds_alternative<std::int64_t>(value)) {
    return std::get<std::int64_t>(value);
  }
  if (std::holds_alternative<double>(value)) {
    return static_cast<std::int64_t>(std::get<double>(value));
  }
  if (std::holds_alternative<bool>(value)) {
    return std::get<bool>(value) ? 1 : 0;
  }
  return 0;
}

arrow::Datum literal_datum(const LiteralExpression& literal) {
  const DataType& type = literal.result_type();
  const std::shared_ptr<arrow::DataType> arrow_type = to_arrow_type(type);
  if (literal.is_null()) {
    return arrow::Datum(arrow::MakeNullScalar(arrow_type));
  }
  const LiteralStorage& value = literal.value();
  switch (type.id) {
    case TypeId::Boolean:
      return arrow::Datum(std::make_shared<arrow::BooleanScalar>(std::holds_alternative<bool>(value) &&
                                                                 std::get<bool>(value)));
    case TypeId::Int32:
      return arrow::Datum(std::make_shared<arrow::Int32Scalar>(static_cast<std::int32_t>(as_int64(value))));
    case TypeId::Int64:
      return arrow::Datum(std::make_shared<arrow::Int64Scalar>(as_int64(value)));
    case TypeId::UInt32:
      return arrow::Datum(std::make_shared<arrow::UInt32Scalar>(static_cast<std::uint32_t>(as_int64(value))));
    case TypeId::UInt64:
      return arrow::Datum(std::make_shared<arrow::UInt64Scalar>(static_cast<std::uint64_t>(as_int64(value))));
    case TypeId::Float32:
      return arrow::Datum(std::make_shared<arrow::FloatScalar>(static_cast<float>(as_double(value))));
    case TypeId::Float64:
      return arrow::Datum(std::make_shared<arrow::DoubleScalar>(as_double(value)));
    case TypeId::String:
      return arrow::Datum(std::make_shared<arrow::StringScalar>(
          std::holds_alternative<std::string>(value) ? std::get<std::string>(value) : ""));
    case TypeId::Date32:
      return arrow::Datum(std::make_shared<arrow::Date32Scalar>(static_cast<std::int32_t>(as_int64(value))));
    case TypeId::Timestamp:
      return arrow::Datum(std::make_shared<arrow::TimestampScalar>(as_int64(value), arrow::TimeUnit::MICRO));
    case TypeId::Decimal:
      throw ExecutionError("DECIMAL literals are not yet supported by the CPU execution backend");
  }
  throw ExecutionError("unreachable: unknown KernelLake TypeId in CPU expression compiler");
}

std::string binary_function_name(BinaryOperator op) {
  switch (op) {
    case BinaryOperator::Add:
      return "add";
    case BinaryOperator::Subtract:
      return "subtract";
    case BinaryOperator::Multiply:
      return "multiply";
    case BinaryOperator::Divide:
      return "divide";
    case BinaryOperator::Equal:
      return "equal";
    case BinaryOperator::NotEqual:
      return "not_equal";
    case BinaryOperator::Less:
      return "less";
    case BinaryOperator::LessEqual:
      return "less_equal";
    case BinaryOperator::Greater:
      return "greater";
    case BinaryOperator::GreaterEqual:
      return "greater_equal";
    case BinaryOperator::And:
      // Kleene (3-valued) logic: matches SQL's NULL-propagating AND/OR
      // semantics, not C++'s two-valued boolean logic.
      return "and_kleene";
    case BinaryOperator::Or:
      return "or_kleene";
  }
  throw ExecutionError("unreachable: unknown BinaryOperator in CPU expression compiler");
}

}  // namespace

arrow::compute::Expression compile_expression_cpu(const Expression& expr) {
  if (const auto* column = dynamic_cast<const ColumnExpression*>(&expr)) {
    return arrow::compute::field_ref(column->name());
  }
  if (const auto* literal = dynamic_cast<const LiteralExpression*>(&expr)) {
    return arrow::compute::literal(literal_datum(*literal));
  }
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(&expr)) {
    return arrow::compute::call(
        binary_function_name(binary->op()),
        {compile_expression_cpu(*binary->left()), compile_expression_cpu(*binary->right())});
  }
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(&expr)) {
    const arrow::compute::Expression operand = compile_expression_cpu(*unary->operand());
    switch (unary->op()) {
      case UnaryOperator::Not:
        return arrow::compute::call("invert", {operand});
      case UnaryOperator::Negate:
        return arrow::compute::call("negate", {operand});
      case UnaryOperator::IsNull:
        return arrow::compute::call("is_null", {operand});
      case UnaryOperator::IsNotNull:
        return arrow::compute::call("is_valid", {operand});
    }
    throw ExecutionError("unreachable: unknown UnaryOperator in CPU expression compiler");
  }
  if (const auto* between = dynamic_cast<const BetweenExpression*>(&expr)) {
    const arrow::compute::Expression value = compile_expression_cpu(*between->value());
    const arrow::compute::Expression ge =
        arrow::compute::call("greater_equal", {value, compile_expression_cpu(*between->lower())});
    const arrow::compute::Expression le =
        arrow::compute::call("less_equal", {value, compile_expression_cpu(*between->upper())});
    return arrow::compute::call("and_kleene", {ge, le});
  }
  if (const auto* cast = dynamic_cast<const CastExpression*>(&expr)) {
    if (cast->result_type().id == TypeId::Decimal || cast->result_type().id == TypeId::String) {
      throw ExecutionError(fmt::format("CAST to {} is not yet supported by the CPU execution backend",
                                       cast->result_type().to_string()));
    }
    arrow::compute::CastOptions options;
    options.to_type = to_arrow_type(cast->result_type());
    return arrow::compute::call("cast", {compile_expression_cpu(*cast->operand())}, options);
  }
  throw ExecutionError(
      "unrecognized expression type in CPU expression compiler (LIKE/IN/CASE/DECIMAL are not yet supported "
      "by "
      "the CPU execution backend -- see docs/ARCHITECTURE.md)");
}

}  // namespace kernellake
