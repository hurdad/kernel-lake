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

}  // namespace

arrow::Datum literal_to_arrow_datum(const LiteralExpression& literal) {
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
  // to_arrow_type() (arrow_adapter.cpp) already rejects an invalid TypeId
  // before literal_to_arrow_datum() ever reaches this switch, so this is
  // truly dead code, not a coverage gap -- see
  // LiteralToArrowDatum.UnknownTypeIdFailsInToArrowTypeBeforeReachingThisFunctionsOwnSwitch
  // in expression_compiler_cpu_test.cpp.
  throw ExecutionError(
      "unreachable: unknown KernelLake TypeId in CPU expression compiler");  // GCOVR_EXCL_LINE
}

namespace {

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

std::string extract_function_name(DatePart part) {
  switch (part) {
    case DatePart::Year:
      return "year";
    case DatePart::Month:
      return "month";
    case DatePart::Day:
      return "day";
  }
  throw ExecutionError("unreachable: unknown DatePart in CPU expression compiler");
}

}  // namespace

arrow::compute::Expression compile_expression_cpu(const Expression& expr) {
  if (const auto* column = dynamic_cast<const ColumnExpression*>(&expr)) {
    // By position, not by name: a JOIN's combined physical schema can have
    // two columns with the same bare name from opposite sides (see
    // physical_planner.cpp's remap_columns(), which is what already
    // guarantees column_index() here is the correct position into
    // whatever Arrow schema this expression is actually evaluated
    // against) -- arrow::compute::field_ref(name) would throw "Multiple
    // matches" for exactly that case instead of silently resolving to the
    // wrong one, but it's still wrong to rely on names being unique here
    // at all.
    return arrow::compute::field_ref(static_cast<int>(column->column_index()));
  }
  if (const auto* literal = dynamic_cast<const LiteralExpression*>(&expr)) {
    return arrow::compute::literal(literal_to_arrow_datum(*literal));
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
  // Arrow Compute's "case_when" kernel takes a struct of per-branch boolean
  // conditions (built via "make_struct") as its first argument, followed by
  // one value expression per condition in the same order, with an optional
  // trailing value for CaseExpression::else_branch() (a row matching no
  // condition and no ELSE emits null -- exactly CaseExpression's own
  // documented semantics, so no extra handling is needed here for that
  // case). MakeStructOptions's field names are never observed by
  // "case_when" itself (positional, not named, lookup) -- placeholder
  // names only exist because MakeStructOptions requires one per field.
  if (const auto* case_expr = dynamic_cast<const CaseExpression*>(&expr)) {
    std::vector<arrow::compute::Expression> conditions;
    std::vector<std::string> condition_names;
    std::vector<arrow::compute::Expression> args;
    conditions.reserve(case_expr->when_then().size());
    condition_names.reserve(case_expr->when_then().size());
    args.reserve(case_expr->when_then().size() + 2);
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      conditions.push_back(compile_expression_cpu(*branch.condition));
      condition_names.push_back(fmt::format("cond_{}", condition_names.size()));
    }
    args.push_back(
        arrow::compute::call("make_struct", conditions, arrow::compute::MakeStructOptions(condition_names)));
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      args.push_back(compile_expression_cpu(*branch.result));
    }
    if (case_expr->else_branch() != nullptr) {
      args.push_back(compile_expression_cpu(*case_expr->else_branch()));
    }
    return arrow::compute::call("case_when", std::move(args));
  }
  // Arrow Compute's "match_like" kernel takes a SQL-style pattern (the same
  // '%'/'_' wildcard syntax LikeExpression::pattern() already stores, no
  // conversion needed) via MatchSubstringOptions -- SQL LIKE is
  // case-sensitive by default, matching MatchSubstringOptions's own
  // ignore_case=false default.
  if (const auto* like = dynamic_cast<const LikeExpression*>(&expr)) {
    const arrow::compute::Expression matched =
        arrow::compute::call("match_like", {compile_expression_cpu(*like->value())},
                             arrow::compute::MatchSubstringOptions(like->pattern()));
    return like->negated() ? arrow::compute::call("invert", {matched}) : matched;
  }
  if (const auto* extract = dynamic_cast<const ExtractExpression*>(&expr)) {
    // Arrow Compute's year()/month()/day() kernels all return INT64,
    // matching ExtractExpression::result_type() exactly -- no extra cast
    // needed, unlike the GPU backend's cudf::datetime::extract_datetime_component
    // (see expression_compiler.cpp), which returns INT16.
    return arrow::compute::call(extract_function_name(extract->part()),
                                {compile_expression_cpu(*extract->operand())});
  }
  if (const auto* substring = dynamic_cast<const SubstringExpression*>(&expr)) {
    // Arrow's own utf8_slice_codeunits is 0-based [start, stop) --
    // SubstringExpression::start_zero_based() does the one -1 conversion
    // from SQL's 1-based convention, at this exact boundary, so both
    // execution backends share the same already-converted value (see
    // expression_compiler.cpp's identical GPU-side use).
    const std::int64_t start = substring->start_zero_based();
    return arrow::compute::call(
        "utf8_slice_codeunits", {compile_expression_cpu(*substring->operand())},
        arrow::compute::SliceOptions(static_cast<std::int64_t>(start), start + substring->length()));
  }
  throw ExecutionError(
      "unrecognized expression type in CPU expression compiler (IN/DECIMAL are not yet supported "
      "by the CPU execution backend -- see docs/ARCHITECTURE.md)");
}

}  // namespace kernellake
