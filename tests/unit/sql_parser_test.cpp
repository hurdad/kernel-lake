#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/sql/parser.hpp"

namespace kernellake::sql {
namespace {

TEST(SqlParser, ParsesGeneralMvpQuery) {
  const auto stmt = parse_sql(
      "SELECT region, SUM(amount) AS total_amount, COUNT(*) AS order_count "
      "FROM read_parquet('/data/sales/*.parquet') "
      "WHERE event_date >= DATE '2026-01-01' AND amount > 0 "
      "GROUP BY region "
      "LIMIT 100");

  ASSERT_EQ(stmt.from.paths.size(), 1u);
  EXPECT_EQ(stmt.from.paths[0], "/data/sales/*.parquet");
  ASSERT_EQ(stmt.select_list.size(), 3u);
  EXPECT_TRUE(std::holds_alternative<AstColumnRef>(stmt.select_list[0]->node));
  ASSERT_TRUE(std::holds_alternative<AstAggregate>(stmt.select_list[1]->node));
  EXPECT_EQ(std::get<AstAggregate>(stmt.select_list[1]->node).function, AstAggregateFunc::Sum);
  EXPECT_EQ(stmt.select_list[1]->alias, "total_amount");
  ASSERT_TRUE(std::holds_alternative<AstAggregate>(stmt.select_list[2]->node));
  EXPECT_EQ(std::get<AstAggregate>(stmt.select_list[2]->node).function, AstAggregateFunc::CountStar);

  ASSERT_NE(stmt.where, nullptr);
  ASSERT_EQ(stmt.group_by.size(), 1u);
  ASSERT_TRUE(stmt.limit.has_value());
  EXPECT_EQ(*stmt.limit, 100);
}

TEST(SqlParser, ParsesTpchQ6Shape) {
  const auto stmt = parse_sql(
      "SELECT SUM(l_extendedprice * l_discount) AS revenue "
      "FROM read_parquet('/data/tpch/lineitem/*.parquet') "
      "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
      "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24");

  ASSERT_EQ(stmt.select_list.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<AstAggregate>(stmt.select_list[0]->node));
  const auto& sum = std::get<AstAggregate>(stmt.select_list[0]->node);
  EXPECT_EQ(sum.function, AstAggregateFunc::Sum);
  ASSERT_TRUE(std::holds_alternative<AstBinary>(sum.argument->node));
  EXPECT_EQ(std::get<AstBinary>(sum.argument->node).op, AstBinaryOp::Multiply);

  ASSERT_NE(stmt.where, nullptr);
  ASSERT_TRUE(std::holds_alternative<AstBinary>(stmt.where->node));
  EXPECT_EQ(std::get<AstBinary>(stmt.where->node).op, AstBinaryOp::And);
}

TEST(SqlParser, ParsesMultiplePathArguments) {
  const auto stmt = parse_sql("SELECT a FROM read_parquet('/data/a.parquet', '/data/b.parquet')");
  ASSERT_EQ(stmt.from.paths.size(), 2u);
  EXPECT_EQ(stmt.from.paths[0], "/data/a.parquet");
  EXPECT_EQ(stmt.from.paths[1], "/data/b.parquet");
}

TEST(SqlParser, ParsesIsNullAndIsNotNull) {
  const auto is_null = parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a IS NULL");
  ASSERT_TRUE(std::holds_alternative<AstUnary>(is_null.where->node));
  EXPECT_EQ(std::get<AstUnary>(is_null.where->node).op, AstUnaryOp::IsNull);

  const auto is_not_null = parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a IS NOT NULL");
  ASSERT_TRUE(std::holds_alternative<AstUnary>(is_not_null.where->node));
  EXPECT_EQ(std::get<AstUnary>(is_not_null.where->node).op, AstUnaryOp::IsNotNull);
}

TEST(SqlParser, ParsesOrderByAndStar) {
  const auto stmt = parse_sql("SELECT * FROM read_parquet('/x.parquet') ORDER BY a DESC, b ASC");
  ASSERT_EQ(stmt.select_list.size(), 1u);
  EXPECT_TRUE(std::holds_alternative<AstStar>(stmt.select_list[0]->node));
  ASSERT_EQ(stmt.order_by.size(), 2u);
  EXPECT_FALSE(stmt.order_by[0].ascending);
  EXPECT_TRUE(stmt.order_by[1].ascending);
}

TEST(SqlParser, ParsesBooleanLiterals) {
  const auto stmt = parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE b = TRUE");
  ASSERT_TRUE(std::holds_alternative<AstBinary>(stmt.where->node));
  const auto& binary = std::get<AstBinary>(stmt.where->node);
  ASSERT_TRUE(std::holds_alternative<AstLiteral>(binary.right->node));
  const auto& literal = std::get<AstLiteral>(binary.right->node);
  EXPECT_EQ(literal.kind, AstLiteralKind::Boolean);
  EXPECT_TRUE(literal.bool_value);
}

TEST(SqlParser, RejectsMissingReadParquetSource) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM sales")), SqlError);
}

TEST(SqlParser, ParsesReadIcebergSource) {
  const auto stmt = parse_sql("SELECT a FROM read_iceberg('prod.db.orders') WHERE a > 0");
  ASSERT_EQ(stmt.from.paths.size(), 1u);
  EXPECT_EQ(stmt.from.paths[0], "iceberg://prod.db.orders");
}

TEST(SqlParser, RejectsReadIcebergWithMultipleArguments) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_iceberg('prod.db.orders', 'extra')")), SqlError);
}

TEST(SqlParser, ParsesReadUnityCatalogSource) {
  const auto stmt = parse_sql("SELECT a FROM read_unity_catalog('prod.main.db.orders') WHERE a > 0");
  ASSERT_EQ(stmt.from.paths.size(), 1u);
  EXPECT_EQ(stmt.from.paths[0], "unitycatalog://prod.main.db.orders");
}

TEST(SqlParser, RejectsReadUnityCatalogWithMultipleArguments) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_unity_catalog('prod.main.db.orders', 'extra')")),
               SqlError);
}

TEST(SqlParser, ParsesJoinBetweenParquetAndUnityCatalogSources) {
  const auto stmt = parse_sql(
      "SELECT a.x, b.y FROM read_parquet('/x.parquet') AS a "
      "JOIN read_unity_catalog('prod.main.db.orders') AS b ON a.order_id = b.order_id");
  ASSERT_TRUE(stmt.join.has_value());
  EXPECT_EQ(stmt.join->first.paths[0], "/x.parquet");
  ASSERT_EQ(stmt.join->steps.size(), 1u);
  EXPECT_EQ(stmt.join->steps[0].source.paths[0], "unitycatalog://prod.main.db.orders");
}

TEST(SqlParser, ParsesJoinBetweenParquetAndIcebergSources) {
  const auto stmt = parse_sql(
      "SELECT a.x, b.y FROM read_parquet('/x.parquet') AS a "
      "JOIN read_iceberg('prod.db.orders') AS b ON a.order_id = b.order_id");
  ASSERT_TRUE(stmt.join.has_value());
  EXPECT_EQ(stmt.join->first.paths[0], "/x.parquet");
  ASSERT_EQ(stmt.join->steps.size(), 1u);
  EXPECT_EQ(stmt.join->steps[0].source.paths[0], "iceberg://prod.db.orders");
}

TEST(SqlParser, RejectsJoinsWithoutAliases) {
  // Both sides of a JOIN must be aliased -- unqualified column references
  // like `a.order_id` would otherwise have no way to pick a side.
  EXPECT_THROW(
      (void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') JOIN read_parquet('/y.parquet') ON true")),
      SqlError);
}

TEST(SqlParser, ParsesTwoTableInnerJoin) {
  const auto stmt = parse_sql(
      "SELECT a.x, b.y FROM read_parquet('/x.parquet') AS a "
      "JOIN read_parquet('/y.parquet') AS b ON a.order_id = b.order_id");
  ASSERT_TRUE(stmt.join.has_value());
  EXPECT_EQ(stmt.join->first.paths, std::vector<std::string>{"/x.parquet"});
  EXPECT_EQ(stmt.join->first.alias, "a");
  ASSERT_EQ(stmt.join->steps.size(), 1u);
  EXPECT_EQ(stmt.join->steps[0].source.paths, std::vector<std::string>{"/y.parquet"});
  EXPECT_EQ(stmt.join->steps[0].source.alias, "b");
  ASSERT_NE(stmt.join->steps[0].condition, nullptr);
  const auto* condition = std::get_if<AstBinary>(&stmt.join->steps[0].condition->node);
  ASSERT_NE(condition, nullptr);
  EXPECT_EQ(condition->op, AstBinaryOp::Eq);
  const auto* left_ref = std::get_if<AstColumnRef>(&condition->left->node);
  ASSERT_NE(left_ref, nullptr);
  EXPECT_EQ(left_ref->name, "order_id");
  ASSERT_TRUE(left_ref->table.has_value());
  EXPECT_EQ(*left_ref->table, "a");

  ASSERT_EQ(stmt.select_list.size(), 2u);
  const auto* select_a = std::get_if<AstColumnRef>(&stmt.select_list[0]->node);
  ASSERT_NE(select_a, nullptr);
  EXPECT_EQ(select_a->name, "x");
  ASSERT_TRUE(select_a->table.has_value());
  EXPECT_EQ(*select_a->table, "a");
}

// Regression test: a 3+-way JOIN chain used to be rejected outright
// ("KernelLake supports at most two read_parquet(...) sources"), even
// though the underlying hsql SQL parser already parses `A JOIN B JOIN C`
// correctly into a left-deep nested TableRef tree ((A JOIN B) JOIN C) --
// this project's own AST conversion was what rejected it, not hsql. Fixed
// by generalizing AstJoinClause to a chain (`first` + `steps`, one step
// per additional source) and flattening the nested TableRef tree via
// flatten_join_chain() in parser.cpp.
TEST(SqlParser, ParsesThreeTableInnerJoinChain) {
  const auto stmt = parse_sql(
      "SELECT a.x FROM read_parquet('/x.parquet') AS a "
      "JOIN read_parquet('/y.parquet') AS b ON a.order_id = b.order_id "
      "JOIN read_parquet('/z.parquet') AS c ON b.order_id = c.order_id");
  ASSERT_TRUE(stmt.join.has_value());
  EXPECT_EQ(stmt.join->first.paths, std::vector<std::string>{"/x.parquet"});
  EXPECT_EQ(stmt.join->first.alias, "a");
  ASSERT_EQ(stmt.join->steps.size(), 2u);
  EXPECT_EQ(stmt.join->steps[0].source.paths, std::vector<std::string>{"/y.parquet"});
  EXPECT_EQ(stmt.join->steps[0].source.alias, "b");
  EXPECT_EQ(stmt.join->steps[1].source.paths, std::vector<std::string>{"/z.parquet"});
  EXPECT_EQ(stmt.join->steps[1].source.alias, "c");
  const auto* second_condition = std::get_if<AstBinary>(&stmt.join->steps[1].condition->node);
  ASSERT_NE(second_condition, nullptr);
  const auto* second_left_ref = std::get_if<AstColumnRef>(&second_condition->left->node);
  ASSERT_NE(second_left_ref, nullptr);
  ASSERT_TRUE(second_left_ref->table.has_value());
  EXPECT_EQ(*second_left_ref->table, "b");
}

TEST(SqlParser, RejectsNonInnerJoin) {
  EXPECT_THROW((void)(parse_sql("SELECT a.x FROM read_parquet('/x.parquet') AS a "
                                "LEFT JOIN read_parquet('/y.parquet') AS b ON a.order_id = b.order_id")),
               SqlError);
}

TEST(SqlParser, RejectsCommaStyleJoin) {
  EXPECT_THROW(
      (void)(parse_sql("SELECT a.x FROM read_parquet('/x.parquet') AS a, read_parquet('/y.parquet') AS b "
                       "WHERE a.order_id = b.order_id")),
      SqlError);
}

// A 3-way JOIN chain is now supported (see ParsesThreeTableInnerJoinChain
// above) -- but an excessive number of chained sources is still rejected,
// as a guard against a pathological number of joined sources driving
// unbounded chain-building work (kMaxJoinSources in parser.cpp), same
// rationale as the paren-depth/SQL-text-length limits elsewhere in this
// file.
TEST(SqlParser, RejectsExcessiveJoinChainLength) {
  std::string sql = "SELECT a0.x FROM read_parquet('/t0.parquet') AS a0";
  for (int i = 1; i <= 13; ++i) {
    const std::string prev = std::to_string(i - 1);
    const std::string cur = std::to_string(i);
    sql += " JOIN read_parquet('/t" + cur + ".parquet') AS a" + cur + " ON a" + prev + ".k = a" + cur + ".k";
  }
  EXPECT_THROW((void)(parse_sql(sql)), SqlError);
}

TEST(SqlParser, RejectsUnsupportedFunction) {
  EXPECT_THROW((void)(parse_sql("SELECT UPPER(a) FROM read_parquet('/x.parquet')")), SqlError);
}

TEST(SqlParser, RejectsMalformedSql) {
  EXPECT_THROW((void)(parse_sql("SELECT FROM read_parquet('/x.parquet') WHERE")), SqlError);
}

TEST(SqlParser, RejectsInvalidDateLiteral) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE d >= DATE '2026-13-40'")),
               SqlError);
}

TEST(SqlParser, RejectsOffset) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') LIMIT 10 OFFSET 5")), SqlError);
}

TEST(SqlParser, ParsesExtractYearMonthDay) {
  AstSelectStatement stmt = parse_sql(
      "SELECT EXTRACT(YEAR FROM d) AS y, EXTRACT(MONTH FROM d) AS m, EXTRACT(DAY FROM d) AS dd FROM "
      "read_parquet('/x.parquet')");
  ASSERT_EQ(stmt.select_list.size(), 3u);
  EXPECT_EQ(std::get<AstExtract>(stmt.select_list[0]->node).field, AstExtractField::Year);
  EXPECT_EQ(std::get<AstExtract>(stmt.select_list[1]->node).field, AstExtractField::Month);
  EXPECT_EQ(std::get<AstExtract>(stmt.select_list[2]->node).field, AstExtractField::Day);
  EXPECT_EQ(std::get<AstColumnRef>(std::get<AstExtract>(stmt.select_list[0]->node).operand->node).name, "d");
}

TEST(SqlParser, RejectsExtractFieldOtherThanYearMonthDay) {
  EXPECT_THROW((void)(parse_sql("SELECT EXTRACT(HOUR FROM d) FROM read_parquet('/x.parquet')")), SqlError);
}

// Regression test: the vendored hyrise/sql-parser (bison/flex, recursive
// descent) recurses once per nesting level with no depth limit of its own
// -- a query with many thousands of nested parens drove it into a C-stack
// overflow (a process crash, not a catchable exception) before parse_sql()
// added its own pre-scan. Both the too-long and too-deep limits must throw
// a clean SqlError instead of ever reaching hsql at all.
TEST(SqlParser, RejectsExcessivelyDeepExpressionNesting) {
  const std::string deep_expr = std::string(600, '(') + "1" + std::string(600, ')');
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a > " + deep_expr)),
               SqlError);
}

TEST(SqlParser, AcceptsReasonablyDeepExpressionNesting) {
  const std::string ok_expr = std::string(100, '(') + "1" + std::string(100, ')');
  EXPECT_NO_THROW((void)(parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a > " + ok_expr)));
}

// Regression test: hsql's own recursive-descent parser builds a chained
// JOIN (`A JOIN B JOIN C JOIN ...`) as a left-deep TableRef tree, and
// recurses once per level while doing so -- with no depth limit of its
// own, same as its parenthesized-expression parsing above. A long enough
// chain (empirically ~40,000 JOINs -- well under this file's 1 MiB
// kMaxSqlBytes cap) drove it into a real C-stack overflow (SIGSEGV)
// *inside hsql::SQLParser::parse() itself*, before KernelLake's own
// semantic kMaxJoinSources check (which only runs after hsql has already
// built its tree, see RejectsExcessiveJoinChainLength above) ever got a
// chance to reject it. Confirmed with a standalone repro before this was
// fixed by adding a cheap pre-scan (kMaxJoinKeywords) alongside the
// existing paren-depth one. If this ever regressed, this test would
// crash the whole test binary, not just fail an assertion.
TEST(SqlParser, RejectsExcessivelyLongJoinChainWithoutCrashing) {
  std::string sql = "SELECT a.x FROM read_parquet('/x.parquet') AS a";
  for (int i = 0; i < 40000; ++i) {
    sql += " JOIN t" + std::to_string(i) + " ON true";
  }
  EXPECT_THROW((void)(parse_sql(sql)), SqlError);
}

TEST(SqlParser, RejectsExcessivelyLongSqlText) {
  const std::string huge_sql =
      "SELECT a FROM read_parquet('/x.parquet') WHERE a > 1 -- " + std::string(1 << 21, 'x');
  EXPECT_THROW((void)(parse_sql(huge_sql)), SqlError);
}

// Regression test: preprocess_from_read_parquet() used to extract
// read_parquet(...)'s comma-separated string-literal arguments with a
// std::regex pattern containing a repeated group
// (`(?:'...'\s*,\s*)*'...'`). libstdc++'s std::regex recurses once per
// repetition of a `(...)*` group, so a single path argument long enough
// (empirically ~35,000 characters -- well under this file's own 1 MiB
// kMaxSqlBytes cap) drove it into a real C-stack overflow: a process
// crash (SIGSEGV), not a catchable SqlError -- confirmed with a
// standalone repro before this was fixed by replacing the regex with a
// linear, non-recursive hand-written scanner. If this ever regressed,
// this test would crash the whole test binary, not just fail an
// assertion.
TEST(SqlParser, AcceptsAVeryLongSinglePathArgumentWithoutCrashing) {
  const std::string long_path(200000, 'a');
  const auto stmt = parse_sql("SELECT a FROM read_parquet('" + long_path + "')");
  ASSERT_EQ(stmt.from.paths.size(), 1u);
  EXPECT_EQ(stmt.from.paths[0], long_path);
}

// Companion regression test for the same underlying bug, exercising many
// repetitions of the comma-separated-argument group instead of one long
// argument.
TEST(SqlParser, AcceptsManyCommaSeparatedPathArgumentsWithoutCrashing) {
  std::string sql = "SELECT a FROM read_parquet(";
  constexpr int kPathCount = 20000;
  for (int i = 0; i < kPathCount; ++i) sql += "'/data/a.parquet', ";
  sql += "'/data/last.parquet')";

  const auto stmt = parse_sql(sql);
  ASSERT_EQ(stmt.from.paths.size(), static_cast<std::size_t>(kPathCount) + 1);
  EXPECT_EQ(stmt.from.paths.front(), "/data/a.parquet");
  EXPECT_EQ(stmt.from.paths.back(), "/data/last.parquet");
}

TEST(SqlParser, ReadParquetIsCaseInsensitive) {
  const auto stmt = parse_sql("SELECT a FROM READ_PARQUET('/x.parquet')");
  ASSERT_EQ(stmt.from.paths.size(), 1u);
  EXPECT_EQ(stmt.from.paths[0], "/x.parquet");
}

TEST(SqlParser, ReadParquetPathArgumentPreservesEscapedQuoteVerbatim) {
  // Matches the pre-existing (unchanged by this fix) behavior of the old
  // regex: an escaped quote inside a path argument doesn't end the string
  // early, but the captured path text keeps the backslash character
  // as-is rather than unescaping it.
  const auto stmt = parse_sql(R"(SELECT a FROM read_parquet('/data/a\'b.parquet'))");
  ASSERT_EQ(stmt.from.paths.size(), 1u);
  EXPECT_EQ(stmt.from.paths[0], R"(/data/a\'b.parquet)");
}

// try_parse_quoted_string()/try_parse_read_parquet_args() (parser.cpp) are
// the hand-rolled, non-recursive replacement for a former std::regex
// pattern that could be crashed via C-stack overflow on attacker-
// controlled input (see that function's own comment) -- each of their
// nullopt-returning shapes below had no test of its own; when unrecognized,
// the preprocessor leaves the text untouched and either hsql's own grammar
// or the "no data source" check ends up rejecting it, so all four still
// surface as SqlError, just via a different path than a clean, purpose-
// built error message.
TEST(SqlParser, RejectsReadParquetWithNonStringArgument) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet(123)")), SqlError);
}

TEST(SqlParser, RejectsReadParquetWithUnclosedStringArgument) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('unclosed")), SqlError);
}

TEST(SqlParser, RejectsReadParquetMissingOpeningParen) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet 'x.parquet'")), SqlError);
}

TEST(SqlParser, RejectsReadParquetMissingClosingParen) {
  EXPECT_THROW((void)(parse_sql("SELECT a FROM read_parquet('x.parquet'")), SqlError);
}

TEST(SqlParser, ParsesHavingClause) {
  const auto stmt = parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING SUM(amount) > 100");
  ASSERT_NE(stmt.having, nullptr);
  const auto* comparison = std::get_if<AstBinary>(&stmt.having->node);
  ASSERT_NE(comparison, nullptr);
  EXPECT_EQ(comparison->op, AstBinaryOp::Gt);
  ASSERT_TRUE(std::holds_alternative<AstAggregate>(comparison->left->node));
  EXPECT_EQ(std::get<AstAggregate>(comparison->left->node).function, AstAggregateFunc::Sum);
}

TEST(SqlParser, HavingIsNullWhenAbsent) {
  const auto stmt = parse_sql("SELECT region FROM read_parquet('/x.parquet') GROUP BY region");
  EXPECT_EQ(stmt.having, nullptr);
}

// `(SELECT ...)` as HAVING's comparison operand -- parses into an
// AstSubquery wrapping a fully-converted nested AstSelectStatement (its
// own FROM/WHERE/aggregate, entirely independent of the outer query's).
TEST(SqlParser, ParsesHavingWithScalarSubquery) {
  const auto stmt = parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING SUM(amount) > (SELECT SUM(amount) * 0.1 FROM read_parquet('/x.parquet'))");
  ASSERT_NE(stmt.having, nullptr);
  const auto* comparison = std::get_if<AstBinary>(&stmt.having->node);
  ASSERT_NE(comparison, nullptr);
  const auto* subquery = std::get_if<AstSubquery>(&comparison->right->node);
  ASSERT_NE(subquery, nullptr);
  ASSERT_NE(subquery->statement, nullptr);
  ASSERT_EQ(subquery->statement->from.paths.size(), 1u);
  EXPECT_EQ(subquery->statement->from.paths[0], "/x.parquet");
  ASSERT_EQ(subquery->statement->select_list.size(), 1u);
  EXPECT_EQ(subquery->statement->having, nullptr);
}

// Regression test for the source-consumption redesign this feature needed:
// preprocess_from_read_parquet() produces one flat, occurrence-ordered
// placeholder list spanning the *entire* raw SQL text, so a subquery with
// its own JOIN chain (repeating the same table twice, matching TPC-H
// Q11's/Q7's own real shape) must not trip the old per-statement
// "preprocessed.sources.size() must exactly match this one statement"
// checks -- each of the outer/subquery JOIN chains must independently
// resolve their own sources by placeholder identity instead.
TEST(SqlParser, SubqueryWithItsOwnJoinChainParsesCorrectly) {
  const auto stmt = parse_sql(
      "SELECT k, SUM(v) AS total FROM read_parquet('/k.parquet') AS a "
      "JOIN read_parquet('/v.parquet') AS b ON a.id = b.id "
      "GROUP BY k HAVING SUM(v) > (SELECT SUM(v) * 0.1 FROM read_parquet('/k2.parquet') AS a "
      "JOIN read_parquet('/v2.parquet') AS b ON a.id = b.id)");
  ASSERT_TRUE(stmt.join.has_value());
  EXPECT_EQ(stmt.join->first.paths, std::vector<std::string>{"/k.parquet"});
  EXPECT_EQ(stmt.join->steps[0].source.paths, std::vector<std::string>{"/v.parquet"});

  ASSERT_NE(stmt.having, nullptr);
  const auto* comparison = std::get_if<AstBinary>(&stmt.having->node);
  ASSERT_NE(comparison, nullptr);
  const auto* subquery = std::get_if<AstSubquery>(&comparison->right->node);
  ASSERT_NE(subquery, nullptr);
  ASSERT_TRUE(subquery->statement->join.has_value());
  EXPECT_EQ(subquery->statement->join->first.paths, std::vector<std::string>{"/k2.parquet"});
  EXPECT_EQ(subquery->statement->join->steps[0].source.paths, std::vector<std::string>{"/v2.parquet"});
}

// Both sql::resolve_subqueries() and the binder walk an AstExpr tree
// recursively -- a subquery sitting directly as HAVING's own comparison
// operand (ParsesHavingWithScalarSubquery above) doesn't exercise that
// recursion at all. Nesting one inside a BETWEEN bound instead does.
TEST(SqlParser, ParsesHavingWithScalarSubqueryNestedInsideBetween) {
  const auto stmt = parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING SUM(amount) BETWEEN (SELECT SUM(amount) * 0.3 FROM read_parquet('/x.parquet')) "
      "AND 200");
  ASSERT_NE(stmt.having, nullptr);
  const auto* between = std::get_if<AstBetween>(&stmt.having->node);
  ASSERT_NE(between, nullptr);
  const auto* subquery = std::get_if<AstSubquery>(&between->lower->node);
  ASSERT_NE(subquery, nullptr);
  ASSERT_NE(subquery->statement, nullptr);
  ASSERT_EQ(subquery->statement->from.paths.size(), 1u);
  EXPECT_EQ(subquery->statement->from.paths[0], "/x.parquet");
}

// Same as above, but nested inside a CASE branch's result instead of a
// BETWEEN bound.
TEST(SqlParser, ParsesHavingWithScalarSubqueryNestedInsideCase) {
  const auto stmt = parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING SUM(amount) > CASE WHEN SUM(amount) > 0 THEN "
      "(SELECT SUM(amount) * 0.3 FROM read_parquet('/x.parquet')) ELSE 0 END");
  ASSERT_NE(stmt.having, nullptr);
  const auto* comparison = std::get_if<AstBinary>(&stmt.having->node);
  ASSERT_NE(comparison, nullptr);
  const auto* case_expr = std::get_if<AstCase>(&comparison->right->node);
  ASSERT_NE(case_expr, nullptr);
  ASSERT_EQ(case_expr->when_then.size(), 1u);
  const auto* subquery = std::get_if<AstSubquery>(&case_expr->when_then[0].second->node);
  ASSERT_NE(subquery, nullptr);
  ASSERT_NE(subquery->statement, nullptr);
  ASSERT_EQ(subquery->statement->from.paths.size(), 1u);
  EXPECT_EQ(subquery->statement->from.paths[0], "/x.parquet");
}

// Regression guard: an ordinary literal IN list inside HAVING must keep
// `subquery == nullptr` and its `list` populated -- same invariant
// InWithLiteralListHasNoSubquery already pins down for WHERE, but HAVING is
// a separate code path (resolve_subqueries() walks it, resolve_in_subqueries()
// never does) that could plausibly mishandle it differently.
TEST(SqlParser, ParsesHavingWithLiteralInListNotMistakenForSubquery) {
  const auto stmt = parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING SUM(amount) > 10 AND region IN ('A', 'B')");
  ASSERT_NE(stmt.having, nullptr);
  const auto* conjunction = std::get_if<AstBinary>(&stmt.having->node);
  ASSERT_NE(conjunction, nullptr);
  EXPECT_EQ(conjunction->op, AstBinaryOp::And);
  const auto* in = std::get_if<AstIn>(&conjunction->right->node);
  ASSERT_NE(in, nullptr);
  EXPECT_EQ(in->subquery, nullptr);
  EXPECT_EQ(in->list.size(), 2u);
}

TEST(SqlParser, RejectsSubqueryOutsideHaving) {
  // convert_expr() happily builds an AstSubquery node wherever hsql's own
  // grammar allows a value expression -- WHERE included -- but nothing
  // downstream ever resolves one there (resolve_subqueries() only ever
  // walks AstSelectStatement::having), so it's the *binder*, not the
  // parser, that's expected to reject this -- see binder_test.cpp's
  // RejectsSubqueryOutsideHaving for that half. This parser-level test
  // only confirms parsing itself succeeds (produces a real AstSubquery
  // node), matching the doc comment on AstSubquery/convert_expr's own
  // kExprSelect case.
  const auto stmt = parse_sql(
      "SELECT a FROM read_parquet('/x.parquet') WHERE a > (SELECT SUM(a) FROM read_parquet('/x.parquet'))");
  ASSERT_NE(stmt.where, nullptr);
  const auto* comparison = std::get_if<AstBinary>(&stmt.where->node);
  ASSERT_NE(comparison, nullptr);
  EXPECT_TRUE(std::holds_alternative<AstSubquery>(comparison->right->node));
}

// `x IN (SELECT ...)` -- TPC-H Q18's shape. Parses into an AstIn with
// `subquery` set to the fully-converted nested statement and `list` left
// empty (mutually exclusive; see ast.hpp's own comment on AstIn).
TEST(SqlParser, ParsesInWithSubquery) {
  const auto stmt = parse_sql(
      "SELECT a FROM read_parquet('/x.parquet') WHERE a IN "
      "(SELECT b FROM read_parquet('/y.parquet') GROUP BY b HAVING SUM(b) > 300)");
  ASSERT_NE(stmt.where, nullptr);
  const auto* in = std::get_if<AstIn>(&stmt.where->node);
  ASSERT_NE(in, nullptr);
  EXPECT_FALSE(in->negated);
  EXPECT_TRUE(in->list.empty());
  ASSERT_NE(in->subquery, nullptr);
  ASSERT_EQ(in->subquery->from.paths.size(), 1u);
  EXPECT_EQ(in->subquery->from.paths[0], "/y.parquet");
  ASSERT_NE(in->subquery->having, nullptr);
}

TEST(SqlParser, ParsesNotInWithSubquery) {
  const auto stmt = parse_sql(
      "SELECT a FROM read_parquet('/x.parquet') WHERE a NOT IN "
      "(SELECT b FROM read_parquet('/y.parquet'))");
  ASSERT_NE(stmt.where, nullptr);
  const auto* in = std::get_if<AstIn>(&stmt.where->node);
  ASSERT_NE(in, nullptr);
  EXPECT_TRUE(in->negated);
  EXPECT_TRUE(in->list.empty());
  ASSERT_NE(in->subquery, nullptr);
}

// Regression guard: a plain literal IN list must keep `subquery == nullptr`
// -- the new field must not accidentally get populated for the
// already-working literal-list path.
TEST(SqlParser, InWithLiteralListHasNoSubquery) {
  const auto stmt = parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a IN (1, 2, 3)");
  ASSERT_NE(stmt.where, nullptr);
  const auto* in = std::get_if<AstIn>(&stmt.where->node);
  ASSERT_NE(in, nullptr);
  EXPECT_EQ(in->list.size(), 3u);
  EXPECT_EQ(in->subquery, nullptr);
}

}  // namespace
}  // namespace kernellake::sql
