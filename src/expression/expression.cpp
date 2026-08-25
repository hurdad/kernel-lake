#include "kernellake/expression/expression.hpp"

#include <sstream>

namespace kernellake {

std::string LiteralExpression::to_string() const {
  if (is_null()) {
    return "NULL";
  }
  return std::visit(
      [](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return "NULL";
        } else if constexpr (std::is_same_v<T, bool>) {
          return value ? "TRUE" : "FALSE";
        } else if constexpr (std::is_same_v<T, std::string>) {
          return "'" + value + "'";
        } else {
          return std::to_string(value);
        }
      },
      value_);
}

std::string_view to_string(BinaryOperator op) noexcept {
  switch (op) {
    case BinaryOperator::Add:
      return "+";
    case BinaryOperator::Subtract:
      return "-";
    case BinaryOperator::Multiply:
      return "*";
    case BinaryOperator::Divide:
      return "/";
    case BinaryOperator::Equal:
      return "=";
    case BinaryOperator::NotEqual:
      return "!=";
    case BinaryOperator::Less:
      return "<";
    case BinaryOperator::LessEqual:
      return "<=";
    case BinaryOperator::Greater:
      return ">";
    case BinaryOperator::GreaterEqual:
      return ">=";
    case BinaryOperator::And:
      return "AND";
    case BinaryOperator::Or:
      return "OR";
  }
  return "?";
}

bool is_arithmetic(BinaryOperator op) noexcept {
  switch (op) {
    case BinaryOperator::Add:
    case BinaryOperator::Subtract:
    case BinaryOperator::Multiply:
    case BinaryOperator::Divide:
      return true;
    default:
      return false;
  }
}

bool is_comparison(BinaryOperator op) noexcept {
  switch (op) {
    case BinaryOperator::Equal:
    case BinaryOperator::NotEqual:
    case BinaryOperator::Less:
    case BinaryOperator::LessEqual:
    case BinaryOperator::Greater:
    case BinaryOperator::GreaterEqual:
      return true;
    default:
      return false;
  }
}

bool is_logical(BinaryOperator op) noexcept {
  return op == BinaryOperator::And || op == BinaryOperator::Or;
}

std::string BinaryExpression::to_string() const {
  return "(" + left_->to_string() + " " + std::string(kernellake::to_string(op_)) + " " +
         right_->to_string() + ")";
}

std::string BinaryExpression::structural_key() const {
  return "(" + left_->structural_key() + " " + std::string(kernellake::to_string(op_)) + " " +
         right_->structural_key() + ")";
}

std::string_view to_string(DatePart part) noexcept {
  switch (part) {
    case DatePart::Year:
      return "YEAR";
    case DatePart::Month:
      return "MONTH";
    case DatePart::Day:
      return "DAY";
  }
  return "?";
}

std::string_view to_string(UnaryOperator op) noexcept {
  switch (op) {
    case UnaryOperator::Not:
      return "NOT";
    case UnaryOperator::Negate:
      return "-";
    case UnaryOperator::IsNull:
      return "IS NULL";
    case UnaryOperator::IsNotNull:
      return "IS NOT NULL";
  }
  return "?";
}

std::string UnaryExpression::to_string() const {
  switch (op_) {
    case UnaryOperator::Not:
      return "NOT (" + operand_->to_string() + ")";
    case UnaryOperator::Negate:
      return "-" + operand_->to_string();
    case UnaryOperator::IsNull:
      return operand_->to_string() + " IS NULL";
    case UnaryOperator::IsNotNull:
      return operand_->to_string() + " IS NOT NULL";
  }
  return "?";
}

std::string UnaryExpression::structural_key() const {
  switch (op_) {
    case UnaryOperator::Not:
      return "NOT (" + operand_->structural_key() + ")";
    case UnaryOperator::Negate:
      return "-" + operand_->structural_key();
    case UnaryOperator::IsNull:
      return operand_->structural_key() + " IS NULL";
    case UnaryOperator::IsNotNull:
      return operand_->structural_key() + " IS NOT NULL";
  }
  return "?";
}

std::string_view to_string(AggregateFunction function) noexcept {
  switch (function) {
    case AggregateFunction::Sum:
      return "SUM";
    case AggregateFunction::Count:
    case AggregateFunction::CountStar:
    case AggregateFunction::CountDistinct:
      return "COUNT";
    case AggregateFunction::Min:
      return "MIN";
    case AggregateFunction::Max:
      return "MAX";
    case AggregateFunction::Avg:
      return "AVG";
  }
  return "?";
}

std::string AggregateExpression::to_string() const {
  std::ostringstream out;
  out << kernellake::to_string(function_) << "(";
  if (function_ == AggregateFunction::CountStar) {
    out << "*";
  } else {
    // CountDistinct's own "DISTINCT " prefix is the only textual difference
    // from a plain Count over the same argument -- both to_string() and
    // structural_key() below need it, so a query with COUNT(x) and
    // COUNT(DISTINCT x) in the same SELECT list (structurally different
    // aggregates) isn't mistaken for a duplicate/shared subexpression by
    // any structural_key()-based comparison elsewhere in the planner.
    if (function_ == AggregateFunction::CountDistinct) {
      out << "DISTINCT ";
    }
    out << argument_->to_string();
  }
  out << ")";
  return out.str();
}

std::string AggregateExpression::structural_key() const {
  std::ostringstream out;
  out << kernellake::to_string(function_) << "(";
  if (function_ == AggregateFunction::CountStar) {
    out << "*";
  } else {
    if (function_ == AggregateFunction::CountDistinct) {
      out << "DISTINCT ";
    }
    out << argument_->structural_key();
  }
  out << ")";
  return out.str();
}

}  // namespace kernellake
