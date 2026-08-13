// resolve_subqueries()/resolve_in_subqueries() (subquery_resolver.cpp) are
// pure AST-to-AST tree walkers with zero I/O/storage dependency of their
// own (see subquery_resolver.hpp's own doc comment), so this drives them
// directly with hand-built AstExprPtr trees and a stub `evaluate` callback
// rather than going through a real QueryEngine. Before this file, these
// functions were only ever exercised indirectly through Q11/Q18's own
// query_engine_execute_cpu_test.cpp end-to-end tests, which never nest a
// subquery inside a CAST/EXTRACT/BETWEEN/LIKE/CASE/aggregate argument --
// this file exercises every branch of the if-constexpr tree walk in both
// functions directly.
#include <gtest/gtest.h>

#include "kernellake/sql/subquery_resolver.hpp"

namespace kernellake::sql {
namespace {

AstExprPtr make_literal_int(std::int64_t value) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstLiteral{AstLiteralKind::Integer, value, 0.0, {}, false};
  return expr;
}

AstExprPtr make_column(std::string name) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstColumnRef{std::move(name), std::nullopt};
  return expr;
}

AstExprPtr make_subquery() {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstSubquery{std::make_shared<AstSelectStatement>()};
  return expr;
}

[[nodiscard]] bool is_subquery(const AstExprPtr& expr) {
  return std::holds_alternative<AstSubquery>(expr->node);
}

[[nodiscard]] const AstLiteral& as_literal(const AstExprPtr& expr) {
  return std::get<AstLiteral>(expr->node);
}

// ---- resolve_subqueries() (HAVING scalar subqueries) ----

TEST(ResolveSubqueries, ReplacesBareSubqueryWithEvaluatedLiteral) {
  const AstExprPtr expr = make_subquery();
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 42, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  EXPECT_EQ(as_literal(result).int_value, 42);
}

TEST(ResolveSubqueries, RecursesIntoBothSidesOfBinary) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstBinary{AstBinaryOp::Gt, make_subquery(), make_subquery()};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 7, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& binary = std::get<AstBinary>(result->node);
  EXPECT_EQ(as_literal(binary.left).int_value, 7);
  EXPECT_EQ(as_literal(binary.right).int_value, 7);
}

TEST(ResolveSubqueries, RecursesIntoUnaryOperand) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstUnary{AstUnaryOp::Not, make_subquery()};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Boolean, 0, 0.0, {}, true};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  EXPECT_TRUE(as_literal(std::get<AstUnary>(result->node).operand).bool_value);
}

TEST(ResolveSubqueries, RecursesIntoCastOperand) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstCast{make_subquery(), "DECIMAL", 10, 2};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 3, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& cast = std::get<AstCast>(result->node);
  EXPECT_EQ(as_literal(cast.operand).int_value, 3);
  EXPECT_EQ(cast.type_name, "DECIMAL");
  EXPECT_EQ(cast.decimal_precision, 10);
  EXPECT_EQ(cast.decimal_scale, 2);
}

TEST(ResolveSubqueries, RecursesIntoExtractOperand) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstExtract{AstExtractField::Year, make_subquery()};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 2024, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& extract = std::get<AstExtract>(result->node);
  EXPECT_EQ(extract.field, AstExtractField::Year);
  EXPECT_EQ(as_literal(extract.operand).int_value, 2024);
}

TEST(ResolveSubqueries, RecursesIntoAllThreeBetweenOperands) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstBetween{make_subquery(), make_subquery(), make_subquery()};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 5, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& between = std::get<AstBetween>(result->node);
  EXPECT_EQ(as_literal(between.value).int_value, 5);
  EXPECT_EQ(as_literal(between.lower).int_value, 5);
  EXPECT_EQ(as_literal(between.upper).int_value, 5);
}

TEST(ResolveSubqueries, RecursesIntoLikeValueAndPatternAndPreservesNegated) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstLike{make_subquery(), make_subquery(), true};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::String, 0, 0.0, "x", false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& like = std::get<AstLike>(result->node);
  EXPECT_EQ(as_literal(like.value).string_value, "x");
  EXPECT_EQ(as_literal(like.pattern).string_value, "x");
  EXPECT_TRUE(like.negated);
}

TEST(ResolveSubqueries, InLeavesSubqueryFieldUntouchedAndResolvesValueAndList) {
  auto expr = std::make_shared<AstExpr>();
  const AstExprPtr in_subquery = make_subquery();
  AstIn in;
  in.value = make_subquery();
  in.list = {make_subquery()};
  in.subquery = std::make_shared<AstSelectStatement>();  // Not touched by resolve_subqueries().
  in.negated = true;
  expr->node = std::move(in);
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& resolved = std::get<AstIn>(result->node);
  EXPECT_EQ(as_literal(resolved.value).int_value, 1);
  ASSERT_EQ(resolved.list.size(), 1u);
  EXPECT_EQ(as_literal(resolved.list[0]).int_value, 1);
  EXPECT_NE(resolved.subquery, nullptr);
  EXPECT_TRUE(resolved.negated);
}

TEST(ResolveSubqueries, RecursesIntoCaseConditionsResultsAndElseBranch) {
  auto expr = std::make_shared<AstExpr>();
  AstCase case_expr;
  case_expr.when_then.emplace_back(make_subquery(), make_subquery());
  case_expr.else_branch = make_subquery();
  expr->node = std::move(case_expr);
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 9, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& resolved = std::get<AstCase>(result->node);
  ASSERT_EQ(resolved.when_then.size(), 1u);
  EXPECT_EQ(as_literal(resolved.when_then[0].first).int_value, 9);
  EXPECT_EQ(as_literal(resolved.when_then[0].second).int_value, 9);
  ASSERT_NE(resolved.else_branch, nullptr);
  EXPECT_EQ(as_literal(resolved.else_branch).int_value, 9);
}

TEST(ResolveSubqueries, CaseWithoutElseBranchStaysNull) {
  auto expr = std::make_shared<AstExpr>();
  AstCase case_expr;
  case_expr.when_then.emplace_back(make_literal_int(1), make_literal_int(2));
  case_expr.else_branch = nullptr;
  expr->node = std::move(case_expr);
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 0, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstCase>(result->node).else_branch, nullptr);
}

TEST(ResolveSubqueries, RecursesIntoAggregateArgumentWhenPresent) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstAggregate{AstAggregateFunc::Sum, make_subquery()};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 100, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  const auto& aggregate = std::get<AstAggregate>(result->node);
  ASSERT_NE(aggregate.argument, nullptr);
  EXPECT_EQ(as_literal(aggregate.argument).int_value, 100);
}

TEST(ResolveSubqueries, CountStarArgumentStaysNull) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstAggregate{AstAggregateFunc::CountStar, nullptr};
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 0, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstAggregate>(result->node).argument, nullptr);
}

TEST(ResolveSubqueries, LeafNodesPassThroughUnchanged) {
  const AstExprPtr expr = make_column("x");
  const auto evaluate = [](const AstSelectStatement&) {
    return AstLiteral{AstLiteralKind::Integer, 0, 0.0, {}, false};
  };
  const AstExprPtr result = resolve_subqueries(expr, evaluate);
  EXPECT_EQ(result, expr);  // Same shared_ptr -- untouched, not just structurally equal.
}

// ---- resolve_in_subqueries() (WHERE `IN (SELECT ...)`, TPC-H Q18) ----

TEST(ResolveInSubqueries, PopulatesListFromNonEmptyEvaluateResultAndClearsSubquery) {
  auto expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  expr->node = std::move(in);
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false},
            AstLiteral{AstLiteralKind::Integer, 2, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  const auto& resolved = std::get<AstIn>(result->node);
  EXPECT_EQ(resolved.subquery, nullptr);
  ASSERT_EQ(resolved.list.size(), 2u);
  EXPECT_EQ(as_literal(resolved.list[0]).int_value, 1);
  EXPECT_EQ(as_literal(resolved.list[1]).int_value, 2);
}

TEST(ResolveInSubqueries, EmptyEvaluateResultBecomesFalseLiteralWhenNotNegated) {
  auto expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in.negated = false;
  expr->node = std::move(in);
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> { return {}; };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  const AstLiteral& literal = as_literal(result);
  EXPECT_EQ(literal.kind, AstLiteralKind::Boolean);
  EXPECT_FALSE(literal.bool_value);
}

TEST(ResolveInSubqueries, EmptyEvaluateResultBecomesTrueLiteralWhenNegated) {
  auto expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in.negated = true;
  expr->node = std::move(in);
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> { return {}; };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  EXPECT_TRUE(as_literal(result).bool_value);
}

TEST(ResolveInSubqueries, NoSubqueryRecursesIntoExistingListItems) {
  auto expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = nullptr;
  in.list = {make_literal_int(1), make_literal_int(2)};
  expr->node = std::move(in);
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    ADD_FAILURE() << "evaluate() must not be called when subquery is null";
    return {};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  const auto& resolved = std::get<AstIn>(result->node);
  ASSERT_EQ(resolved.list.size(), 2u);
  EXPECT_EQ(as_literal(resolved.list[0]).int_value, 1);
  EXPECT_EQ(as_literal(resolved.list[1]).int_value, 2);
}

TEST(ResolveInSubqueries, RecursesIntoBinaryOperandsToFindNestedIn) {
  auto expr = std::make_shared<AstExpr>();
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  expr->node = AstBinary{AstBinaryOp::And, in_expr, make_literal_int(1)};
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 5, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  const auto& binary = std::get<AstBinary>(result->node);
  const auto& resolved_in = std::get<AstIn>(binary.left->node);
  EXPECT_EQ(resolved_in.subquery, nullptr);
  ASSERT_EQ(resolved_in.list.size(), 1u);
}

TEST(ResolveInSubqueries, RecursesIntoUnaryOperand) {
  auto expr = std::make_shared<AstExpr>();
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  expr->node = AstUnary{AstUnaryOp::Not, in_expr};
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstIn>(std::get<AstUnary>(result->node).operand->node).subquery, nullptr);
}

TEST(ResolveInSubqueries, RecursesIntoCastOperand) {
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstCast{in_expr, "INT64", 0, 0};
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstIn>(std::get<AstCast>(result->node).operand->node).subquery, nullptr);
}

TEST(ResolveInSubqueries, RecursesIntoExtractOperand) {
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstExtract{AstExtractField::Month, in_expr};
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstIn>(std::get<AstExtract>(result->node).operand->node).subquery, nullptr);
}

TEST(ResolveInSubqueries, RecursesIntoAllThreeBetweenOperands) {
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstBetween{in_expr, make_literal_int(0), make_literal_int(10)};
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstIn>(std::get<AstBetween>(result->node).value->node).subquery, nullptr);
}

TEST(ResolveInSubqueries, RecursesIntoLikeValueAndPattern) {
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstLike{in_expr, make_literal_int(0), false};
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstIn>(std::get<AstLike>(result->node).value->node).subquery, nullptr);
}

TEST(ResolveInSubqueries, RecursesIntoCaseConditionsResultsAndElseBranch) {
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  auto expr = std::make_shared<AstExpr>();
  AstCase case_expr;
  case_expr.when_then.emplace_back(in_expr, make_literal_int(1));
  case_expr.else_branch = nullptr;
  expr->node = std::move(case_expr);
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  const auto& resolved = std::get<AstCase>(result->node);
  EXPECT_EQ(std::get<AstIn>(resolved.when_then[0].first->node).subquery, nullptr);
  EXPECT_EQ(resolved.else_branch, nullptr);
}

TEST(ResolveInSubqueries, RecursesIntoAggregateArgumentWhenPresent) {
  auto in_expr = std::make_shared<AstExpr>();
  AstIn in;
  in.value = make_column("id");
  in.subquery = std::make_shared<AstSelectStatement>();
  in_expr->node = std::move(in);
  auto expr = std::make_shared<AstExpr>();
  expr->node = AstAggregate{AstAggregateFunc::Sum, in_expr};
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    return {AstLiteral{AstLiteralKind::Integer, 1, 0.0, {}, false}};
  };
  const AstExprPtr result = resolve_in_subqueries(expr, evaluate);
  EXPECT_EQ(std::get<AstIn>(std::get<AstAggregate>(result->node).argument->node).subquery, nullptr);
}

TEST(ResolveInSubqueries, LeafAndSubqueryNodesPassThroughUnchanged) {
  const auto evaluate = [](const AstSelectStatement&) -> std::vector<AstLiteral> {
    ADD_FAILURE() << "evaluate() must not be called for a leaf/HAVING-subquery node";
    return {};
  };

  const AstExprPtr column_expr = make_column("x");
  EXPECT_EQ(resolve_in_subqueries(column_expr, evaluate), column_expr);

  const AstExprPtr subquery_expr = make_subquery();
  const AstExprPtr result = resolve_in_subqueries(subquery_expr, evaluate);
  EXPECT_TRUE(is_subquery(result));
  EXPECT_EQ(result, subquery_expr);
}

}  // namespace
}  // namespace kernellake::sql
