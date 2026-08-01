#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "kernellake/types/schema.hpp"

namespace kernellake {

// Expressions are always fully typed: they are produced by the binder
// (see kernellake::sql binder), never directly from raw SQL text. The SQL
// parser's own AST (kernellake::sql) is a separate, untyped representation
// that the binder consumes to build these nodes.
class Expression;
using ExpressionPtr = std::shared_ptr<const Expression>;

class Expression {
public:
  virtual ~Expression() = default;

  [[nodiscard]] virtual const DataType& result_type() const = 0;
  [[nodiscard]] virtual std::string to_string() const = 0;
};

// ---------------------------------------------------------------------------
// Column reference
// ---------------------------------------------------------------------------

class ColumnExpression final : public Expression {
public:
  ColumnExpression(std::string name, std::size_t column_index, DataType type)
      : name_(std::move(name)), column_index_(column_index), type_(std::move(type)) {}

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] std::size_t column_index() const noexcept { return column_index_; }
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override { return name_; }

private:
  std::string name_;
  std::size_t column_index_;
  DataType type_;
};

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

// Dates are stored as days since the Unix epoch (matching Arrow date32),
// timestamps as microseconds since the Unix epoch (matching Arrow's
// microsecond timestamp unit used throughout KernelLake).
using LiteralStorage = std::variant<std::monostate, bool, std::int64_t, double, std::string>;

class LiteralExpression final : public Expression {
public:
  LiteralExpression(LiteralStorage value, DataType type)
      : value_(std::move(value)), type_(std::move(type)) {}

  [[nodiscard]] static LiteralExpression make_null(DataType type) {
    return LiteralExpression(std::monostate{}, std::move(type));
  }
  [[nodiscard]] static LiteralExpression make_bool(bool value) {
    return LiteralExpression(value, boolean_type(false));
  }
  [[nodiscard]] static LiteralExpression make_int64(std::int64_t value) {
    return LiteralExpression(value, int64_type(false));
  }
  [[nodiscard]] static LiteralExpression make_float64(double value) {
    return LiteralExpression(value, float64_type(false));
  }
  [[nodiscard]] static LiteralExpression make_string(std::string value) {
    return LiteralExpression(std::move(value), string_type(false));
  }
  [[nodiscard]] static LiteralExpression make_date32(std::int32_t days_since_epoch) {
    return LiteralExpression(static_cast<std::int64_t>(days_since_epoch), date32_type(false));
  }
  [[nodiscard]] static LiteralExpression make_timestamp(std::int64_t micros_since_epoch) {
    return LiteralExpression(micros_since_epoch, timestamp_type(false));
  }

  [[nodiscard]] bool is_null() const noexcept { return std::holds_alternative<std::monostate>(value_); }
  [[nodiscard]] const LiteralStorage& value() const noexcept { return value_; }
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override;

private:
  LiteralStorage value_;
  DataType type_;
};

// ---------------------------------------------------------------------------
// Binary expressions
// ---------------------------------------------------------------------------

enum class BinaryOperator {
  Add,
  Subtract,
  Multiply,
  Divide,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  And,
  Or,
};

[[nodiscard]] std::string_view to_string(BinaryOperator op) noexcept;
[[nodiscard]] bool is_arithmetic(BinaryOperator op) noexcept;
[[nodiscard]] bool is_comparison(BinaryOperator op) noexcept;
[[nodiscard]] bool is_logical(BinaryOperator op) noexcept;

class BinaryExpression final : public Expression {
public:
  BinaryExpression(BinaryOperator op, ExpressionPtr left, ExpressionPtr right, DataType result_type)
      : op_(op), left_(std::move(left)), right_(std::move(right)), type_(std::move(result_type)) {}

  [[nodiscard]] BinaryOperator op() const noexcept { return op_; }
  [[nodiscard]] const ExpressionPtr& left() const noexcept { return left_; }
  [[nodiscard]] const ExpressionPtr& right() const noexcept { return right_; }
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override;

private:
  BinaryOperator op_;
  ExpressionPtr left_;
  ExpressionPtr right_;
  DataType type_;
};

// ---------------------------------------------------------------------------
// Unary expressions
// ---------------------------------------------------------------------------

enum class UnaryOperator {
  Not,
  Negate,
  IsNull,
  IsNotNull,
};

[[nodiscard]] std::string_view to_string(UnaryOperator op) noexcept;

class UnaryExpression final : public Expression {
public:
  UnaryExpression(UnaryOperator op, ExpressionPtr operand, DataType result_type)
      : op_(op), operand_(std::move(operand)), type_(std::move(result_type)) {}

  [[nodiscard]] UnaryOperator op() const noexcept { return op_; }
  [[nodiscard]] const ExpressionPtr& operand() const noexcept { return operand_; }
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override;

private:
  UnaryOperator op_;
  ExpressionPtr operand_;
  DataType type_;
};

// ---------------------------------------------------------------------------
// Cast
// ---------------------------------------------------------------------------

class CastExpression final : public Expression {
public:
  CastExpression(ExpressionPtr operand, DataType target_type)
      : operand_(std::move(operand)), type_(std::move(target_type)) {}

  [[nodiscard]] const ExpressionPtr& operand() const noexcept { return operand_; }
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override {
    return "CAST(" + operand_->to_string() + " AS " + type_.to_string() + ")";
  }

private:
  ExpressionPtr operand_;
  DataType type_;
};

// ---------------------------------------------------------------------------
// BETWEEN
// ---------------------------------------------------------------------------

class BetweenExpression final : public Expression {
public:
  BetweenExpression(ExpressionPtr value, ExpressionPtr lower, ExpressionPtr upper)
      : value_(std::move(value)),
        lower_(std::move(lower)),
        upper_(std::move(upper)),
        type_(boolean_type(false)) {}

  [[nodiscard]] const ExpressionPtr& value() const noexcept { return value_; }
  [[nodiscard]] const ExpressionPtr& lower() const noexcept { return lower_; }
  [[nodiscard]] const ExpressionPtr& upper() const noexcept { return upper_; }
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override {
    return value_->to_string() + " BETWEEN " + lower_->to_string() + " AND " + upper_->to_string();
  }

private:
  ExpressionPtr value_;
  ExpressionPtr lower_;
  ExpressionPtr upper_;
  DataType type_;
};

// ---------------------------------------------------------------------------
// Aggregates
// ---------------------------------------------------------------------------

enum class AggregateFunction {
  Sum,
  Count,
  CountStar,
  Min,
  Max,
  Avg,
};

[[nodiscard]] std::string_view to_string(AggregateFunction function) noexcept;

class AggregateExpression final : public Expression {
public:
  // `argument` is null only for CountStar (COUNT(*) has no operand).
  AggregateExpression(AggregateFunction function, ExpressionPtr argument, DataType result_type)
      : function_(function), argument_(std::move(argument)), type_(std::move(result_type)) {}

  [[nodiscard]] AggregateFunction function() const noexcept { return function_; }
  [[nodiscard]] const ExpressionPtr& argument() const noexcept { return argument_; }
  [[nodiscard]] const DataType& result_type() const override { return type_; }
  [[nodiscard]] std::string to_string() const override;

private:
  AggregateFunction function_;
  ExpressionPtr argument_;
  DataType type_;
};

}  // namespace kernellake
