#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/planner/binder.hpp"
#include "kernellake/sql/parser.hpp"

namespace kernellake {
namespace {

Schema sales_schema() {
  return Schema({
      Field{"region", string_type(false)},
      Field{"amount", float64_type(false)},
      Field{"event_date", date32_type(false)},
  });
}

Schema priced_sales_schema() {
  return Schema({
      Field{"region", string_type(false)},
      Field{"amount", float64_type(false)},
      Field{"price", decimal_type(10, 2, false)},
  });
}

Schema orders_schema() {
  return Schema({
      Field{"order_id", int64_type(false)},
      Field{"customer_id", int64_type(false)},
      Field{"amount", float64_type(false)},
  });
}

// customer_id is INT32 here (vs. orders_schema()'s INT64) -- for a JOIN
// test that needs two numerically-compatible-but-different-width key
// columns, distinct from customers_schema()'s exact-type-match column.
Schema narrow_customers_schema() {
  return Schema({
      Field{"customer_id", int32_type(false)},
      Field{"name", string_type(false)},
  });
}

Schema customers_schema() {
  return Schema({
      Field{"customer_id", int64_type(false)},
      Field{"name", string_type(false)},
  });
}

Schema lineitem_schema() {
  return Schema({
      Field{"l_extendedprice", float64_type(false)},
      Field{"l_discount", float64_type(false)},
      Field{"l_shipdate", date32_type(false)},
      Field{"l_quantity", int32_type(false)},
  });
}

Schema unsigned_counters_schema() {
  return Schema({
      Field{"signed_count", int64_type(false)},
      Field{"unsigned_count", uint64_type(false)},
  });
}

// A third table for N-way-join tests: joins onto customers_schema() via
// customer_id, deliberately reusing "name" (also present in
// customers_schema()) to exercise cross-source ambiguity.
Schema nations_schema() {
  return Schema({
      Field{"customer_id", int64_type(false)},
      Field{"name", string_type(false)},
  });
}

TEST(Binder, BindsGeneralMvpQuery) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total_amount, COUNT(*) AS order_count "
      "FROM read_parquet('/data/sales/*.parquet') "
      "WHERE event_date >= DATE '2026-01-01' AND amount > 0 "
      "GROUP BY region "
      "LIMIT 100");
  const BoundQuery bound = bind_query(stmt, sales_schema());

  ASSERT_EQ(bound.select_list.size(), 3u);
  EXPECT_EQ(bound.select_list[0].output_name, "region");
  EXPECT_EQ(bound.select_list[1].output_name, "total_amount");
  EXPECT_EQ(bound.select_list[1].expr->result_type().id, TypeId::Float64);
  EXPECT_EQ(bound.select_list[2].output_name, "order_count");
  EXPECT_EQ(bound.select_list[2].expr->result_type().id, TypeId::Int64);
  EXPECT_TRUE(bound.is_aggregate_query);
  ASSERT_NE(bound.where, nullptr);
  EXPECT_EQ(bound.where->result_type().id, TypeId::Boolean);
  ASSERT_EQ(bound.group_by.size(), 1u);
  EXPECT_EQ(bound.limit, 100);
  EXPECT_EQ(bound.output_schema.field_count(), 3u);
}

TEST(Binder, BindsTpchQ6ArithmeticOnGpuShape) {
  const auto stmt = sql::parse_sql(
      "SELECT SUM(l_extendedprice * l_discount) AS revenue "
      "FROM read_parquet('/data/tpch/lineitem/*.parquet') "
      "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' "
      "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24");
  const BoundQuery bound = bind_query(stmt, lineitem_schema());

  ASSERT_EQ(bound.select_list.size(), 1u);
  EXPECT_EQ(bound.select_list[0].output_name, "revenue");
  EXPECT_EQ(bound.select_list[0].expr->result_type().id, TypeId::Float64);
  EXPECT_TRUE(bound.is_aggregate_query);
  ASSERT_NE(bound.where, nullptr);
  EXPECT_EQ(bound.where->result_type().id, TypeId::Boolean);
}

TEST(Binder, RejectsUnknownColumn) {
  const auto stmt = sql::parse_sql("SELECT nonexistent FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, RejectsUngroupedColumnInAggregateQuery) {
  const auto stmt =
      sql::parse_sql("SELECT region, amount, SUM(amount) FROM read_parquet('/x.parquet') GROUP BY region");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, AllowsGroupedColumnInAggregateQuery) {
  const auto stmt =
      sql::parse_sql("SELECT region, SUM(amount) FROM read_parquet('/x.parquet') GROUP BY region");
  EXPECT_NO_THROW((void)(bind_query(stmt, sales_schema())));
}

TEST(Binder, RejectsIncompatibleComparison) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region > 5");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, RejectsNonBooleanWhere) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, RejectsDuplicateOutputNames) {
  const auto stmt = sql::parse_sql("SELECT region AS x, amount AS x FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, RejectsAggregateInWhere) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE SUM(amount) > 0");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, RejectsNestedAggregateFunctions) {
  const auto stmt = sql::parse_sql("SELECT SUM(SUM(amount)) FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, ExpandsStar) {
  const auto stmt = sql::parse_sql("SELECT * FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.select_list.size(), 3u);
  EXPECT_EQ(bound.select_list[0].output_name, "region");
  EXPECT_EQ(bound.select_list[2].output_name, "event_date");
}

TEST(Binder, InsertsSafeNumericCastForComparison) {
  Schema schema({Field{"a", int32_type(false)}});
  const auto stmt = sql::parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a > 1.5");
  const BoundQuery bound = bind_query(stmt, schema);
  ASSERT_NE(bound.where, nullptr);
  EXPECT_EQ(bound.where->result_type().id, TypeId::Boolean);
}

TEST(Binder, NullLiteralComparisonBindsWithoutError) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount = NULL");
  EXPECT_NO_THROW((void)(bind_query(stmt, sales_schema())));
}

TEST(Binder, LikeRequiresStringOperandAndLiteralPattern) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region LIKE 'A%'");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_NE(bound.where, nullptr);
  const auto* like = dynamic_cast<const LikeExpression*>(bound.where.get());
  ASSERT_NE(like, nullptr);
  EXPECT_EQ(like->pattern(), "A%");
  EXPECT_FALSE(like->negated());

  const auto not_stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region NOT LIKE 'A%'");
  const BoundQuery not_bound = bind_query(not_stmt, sales_schema());
  const auto* not_like = dynamic_cast<const LikeExpression*>(not_bound.where.get());
  ASSERT_NE(not_like, nullptr);
  EXPECT_TRUE(not_like->negated());

  const auto numeric_stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount LIKE 'A%'");
  EXPECT_THROW((void)(bind_query(numeric_stmt, sales_schema())), BindingError);
}

TEST(Binder, InDesugarsIntoOrChainOfEqualityComparisons) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region IN ('A', 'B', 'C')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_NE(bound.where, nullptr);
  // (region = 'A') OR (region = 'B') OR (region = 'C'), left-associated.
  const auto* outer_or = dynamic_cast<const BinaryExpression*>(bound.where.get());
  ASSERT_NE(outer_or, nullptr);
  EXPECT_EQ(outer_or->op(), BinaryOperator::Or);
  const auto* inner_or = dynamic_cast<const BinaryExpression*>(outer_or->left().get());
  ASSERT_NE(inner_or, nullptr);
  EXPECT_EQ(inner_or->op(), BinaryOperator::Or);
  const auto* first_eq = dynamic_cast<const BinaryExpression*>(inner_or->left().get());
  ASSERT_NE(first_eq, nullptr);
  EXPECT_EQ(first_eq->op(), BinaryOperator::Equal);
}

TEST(Binder, NotInNegatesTheDesugaredOrChain) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region NOT IN ('A', 'B')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  const auto* not_expr = dynamic_cast<const UnaryExpression*>(bound.where.get());
  ASSERT_NE(not_expr, nullptr);
  EXPECT_EQ(not_expr->op(), UnaryOperator::Not);
}

TEST(Binder, CaseRequiresBooleanConditionsAndUnifiesResultTypes) {
  const auto stmt = sql::parse_sql(
      "SELECT CASE WHEN amount > 500 THEN 1 WHEN amount > 100 THEN 2 ELSE 0 END AS bucket "
      "FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.select_list.size(), 1u);
  const auto* case_expr = dynamic_cast<const CaseExpression*>(bound.select_list[0].expr.get());
  ASSERT_NE(case_expr, nullptr);
  EXPECT_EQ(case_expr->when_then().size(), 2u);
  EXPECT_NE(case_expr->else_branch(), nullptr);
  EXPECT_EQ(case_expr->result_type().id, TypeId::Int64);
  EXPECT_FALSE(case_expr->result_type().nullable);  // ELSE present, all branches non-null

  const auto no_else_stmt =
      sql::parse_sql("SELECT CASE WHEN amount > 500 THEN 1 END AS bucket FROM read_parquet('/x.parquet')");
  const BoundQuery no_else_bound = bind_query(no_else_stmt, sales_schema());
  const auto* no_else_case = dynamic_cast<const CaseExpression*>(no_else_bound.select_list[0].expr.get());
  ASSERT_NE(no_else_case, nullptr);
  EXPECT_TRUE(no_else_case->result_type().nullable);  // no ELSE -> NULL is possible

  const auto non_boolean_condition_stmt =
      sql::parse_sql("SELECT CASE WHEN amount THEN 1 ELSE 0 END AS bucket FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(non_boolean_condition_stmt, sales_schema())), BindingError);

  const auto incompatible_branches_stmt = sql::parse_sql(
      "SELECT CASE WHEN amount > 500 THEN 'high' ELSE 0 END AS bucket FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(incompatible_branches_stmt, sales_schema())), BindingError);
}

TEST(Binder, CastMapsSqlTypeNamesToDataTypes) {
  const auto stmt = sql::parse_sql("SELECT CAST(amount AS BIGINT) AS x FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  const auto* cast = dynamic_cast<const CastExpression*>(bound.select_list[0].expr.get());
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(cast->result_type().id, TypeId::Int64);

  // hsql's grammar requires an explicit length for VARCHAR (bare VARCHAR
  // with no length is a parse error, unlike BIGINT/INT/etc.); the length
  // itself is unused by KernelLake's binder, which only has one unbounded
  // String type.
  const auto varchar_stmt =
      sql::parse_sql("SELECT CAST(amount AS VARCHAR(255)) AS x FROM read_parquet('/x.parquet')");
  const BoundQuery varchar_bound = bind_query(varchar_stmt, sales_schema());
  const auto* varchar_cast = dynamic_cast<const CastExpression*>(varchar_bound.select_list[0].expr.get());
  ASSERT_NE(varchar_cast, nullptr);
  EXPECT_EQ(varchar_cast->result_type().id, TypeId::String);

  const auto decimal_stmt =
      sql::parse_sql("SELECT CAST(amount AS DECIMAL) AS x FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(decimal_stmt, sales_schema())), BindingError);
}

TEST(Binder, GroupByResolvesSelectListAliasForComputedExpressions) {
  // "bucket" is not a base-table column -- it can only resolve against the
  // SELECT-list alias, since that's the only way to name a computed GROUP
  // BY key like a CASE expression.
  const auto stmt = sql::parse_sql(
      "SELECT CASE WHEN amount > 500 THEN 'high' ELSE 'low' END AS bucket, COUNT(*) AS n "
      "FROM read_parquet('/x.parquet') GROUP BY bucket");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.group_by.size(), 1u);
  EXPECT_NE(dynamic_cast<const CaseExpression*>(bound.group_by[0].get()), nullptr);

  const auto unknown_alias_stmt = sql::parse_sql(
      "SELECT region, COUNT(*) AS n FROM read_parquet('/x.parquet') GROUP BY nonexistent_alias");
  EXPECT_THROW((void)(bind_query(unknown_alias_stmt, sales_schema())), BindingError);
}

TEST(Binder, CastToDecimalRequiresExplicitPrecisionAndScale) {
  const auto ok_stmt =
      sql::parse_sql("SELECT CAST(amount AS DECIMAL(10, 2)) AS x FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(ok_stmt, sales_schema());
  const auto* cast = dynamic_cast<const CastExpression*>(bound.select_list[0].expr.get());
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(cast->result_type().id, TypeId::Decimal);
  EXPECT_EQ(cast->result_type().precision, 10);
  EXPECT_EQ(cast->result_type().scale, 2);

  // Precision > 38 (cudf's fixed_point ceiling) is rejected.
  const auto too_wide_stmt =
      sql::parse_sql("SELECT CAST(amount AS DECIMAL(39, 2)) AS x FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(too_wide_stmt, sales_schema())), BindingError);

  // Scale > precision is nonsensical (more digits after the point than
  // total digits) and rejected.
  const auto bad_scale_stmt =
      sql::parse_sql("SELECT CAST(amount AS DECIMAL(5, 10)) AS x FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(bad_scale_stmt, sales_schema())), BindingError);
}

TEST(Binder, DecimalColumnComparedAgainstLiteralCoercesTheLiteral) {
  // `price` is DECIMAL(10,2); the literal 19.99 (parsed as a FLOAT64
  // literal by hsql) must be retyped to match rather than rejected -- see
  // cast_if_needed in binder.cpp.
  const auto stmt = sql::parse_sql("SELECT price FROM read_parquet('/x.parquet') WHERE price > 19.99");
  EXPECT_NO_THROW((void)(bind_query(stmt, priced_sales_schema())));
}

// Regression test: cast_if_needed() used to retype a literal to a DECIMAL
// column's precision/scale with no check that the literal's magnitude
// actually fits -- decimal_raw_value()/make_decimal_scalar()
// (src/execution_gpu/cudf_adapter.cpp) would then silently narrow the
// out-of-range scaled value into a 32-bit raw integer via an unchecked
// static_cast, wrapping it to a huge negative number instead of failing.
// `price` is DECIMAL(10,2) (max magnitude 99999999.99); this literal
// overflows it by one order of magnitude.
TEST(Binder, DecimalLiteralExceedingColumnPrecisionIsRejected) {
  const auto stmt = sql::parse_sql("SELECT price FROM read_parquet('/x.parquet') WHERE price > 999999999.99");
  EXPECT_THROW((void)(bind_query(stmt, priced_sales_schema())), BindingError);
}

TEST(Binder, DecimalColumnComparedAgainstNonLiteralColumnIsRejected) {
  // `amount` is a genuine FLOAT64 column, not a compile-time constant --
  // implicit DECIMAL promotion only retypes literals (see cast_if_needed),
  // so this must fail cleanly at bind time rather than silently
  // misevaluating.
  const auto stmt = sql::parse_sql("SELECT price FROM read_parquet('/x.parquet') WHERE price > amount");
  EXPECT_THROW((void)(bind_query(stmt, priced_sales_schema())), BindingError);
}

TEST(Binder, MixingTwoDifferentDecimalTypesIsRejected) {
  const auto stmt = sql::parse_sql(
      "SELECT price FROM read_parquet('/x.parquet') WHERE price > CAST(amount AS DECIMAL(12, 4))");
  EXPECT_THROW((void)(bind_query(stmt, priced_sales_schema())), BindingError);
}

TEST(Binder, MixingSignedAndUnsignedIntegerTypesInComparisonIsRejected) {
  // A negative signed_count would silently wrap around to a huge positive
  // value if promoted to UINT64 and compared (confirmed against a real GPU:
  // CAST(-5 AS UINT64) == 18446744073709551611) -- must fail cleanly at bind
  // time instead, same as mismatched DECIMALs above.
  const auto stmt = sql::parse_sql(
      "SELECT signed_count FROM read_parquet('/x.parquet') WHERE signed_count < unsigned_count");
  EXPECT_THROW((void)(bind_query(stmt, unsigned_counters_schema())), BindingError);
}

TEST(Binder, MixingSignedAndUnsignedIntegerTypesInArithmeticIsRejected) {
  const auto stmt = sql::parse_sql("SELECT signed_count + unsigned_count FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, unsigned_counters_schema())), BindingError);
}

TEST(Binder, UnaryNegateOnUnsignedColumnIsRejected) {
  // expression_compiler.cpp synthesizes unary '-' as `0 - x`; for an
  // unsigned x that silently two's-complement-wraps instead of producing a
  // negative value (confirmed against a real GPU: `0u - 5u` (UINT32) ==
  // 4294967291) -- must fail cleanly at bind time instead.
  const auto stmt = sql::parse_sql("SELECT -unsigned_count FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, unsigned_counters_schema())), BindingError);
}

TEST(Binder, UnaryNegateOnSignedColumnStillWorks) {
  const auto stmt = sql::parse_sql("SELECT -signed_count FROM read_parquet('/x.parquet')");
  EXPECT_NO_THROW((void)(bind_query(stmt, unsigned_counters_schema())));
}

TEST(Binder, SumOverDecimalPreservesPrecisionAndScale) {
  const auto stmt = sql::parse_sql("SELECT SUM(price) AS total FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, priced_sales_schema());
  ASSERT_EQ(bound.select_list.size(), 1u);
  const DataType& result_type = bound.select_list[0].expr->result_type();
  EXPECT_EQ(result_type.id, TypeId::Decimal);
  EXPECT_EQ(result_type.precision, 10);
  EXPECT_EQ(result_type.scale, 2);
}

TEST(Binder, MinMaxOverDecimalPreservePrecisionAndScale) {
  const auto stmt =
      sql::parse_sql("SELECT MIN(price) AS lo, MAX(price) AS hi FROM read_parquet('/x.parquet')");
  const BoundQuery bound = bind_query(stmt, priced_sales_schema());
  for (const BoundSelectItem& item : bound.select_list) {
    EXPECT_EQ(item.expr->result_type().id, TypeId::Decimal);
    EXPECT_EQ(item.expr->result_type().precision, 10);
    EXPECT_EQ(item.expr->result_type().scale, 2);
  }
}

TEST(Binder, AvgOverDecimalIsRejected) {
  const auto stmt = sql::parse_sql("SELECT AVG(price) AS avg_price FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, priced_sales_schema())), BindingError);
}

TEST(Binder, GroupByPrefersBaseColumnOverSameNamedAlias) {
  // "region" is both a real base-table column and this query's own alias
  // for it -- resolving against the base column either way gives the same
  // answer, but this pins down that base-table resolution is tried first.
  const auto stmt = sql::parse_sql(
      "SELECT region AS region, COUNT(*) AS n FROM read_parquet('/x.parquet') GROUP BY region");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.group_by.size(), 1u);
  const auto* column = dynamic_cast<const ColumnExpression*>(bound.group_by[0].get());
  ASSERT_NE(column, nullptr);
  EXPECT_EQ(column->name(), "region");
}

TEST(Binder, ExtractYearOverDateColumnMatchesExpectedGroupByKeyAndType) {
  const auto stmt = sql::parse_sql(
      "SELECT EXTRACT(YEAR FROM event_date) AS y, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY y");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_EQ(bound.group_by.size(), 1u);
  const auto* extract = dynamic_cast<const ExtractExpression*>(bound.group_by[0].get());
  ASSERT_NE(extract, nullptr);
  EXPECT_EQ(extract->part(), DatePart::Year);
  EXPECT_EQ(extract->result_type().id, TypeId::Int64);
}

TEST(Binder, ExtractOverNonDateColumnIsRejected) {
  const auto stmt = sql::parse_sql("SELECT EXTRACT(YEAR FROM amount) FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, JoinResolvesQualifiedAndUnambiguousUnqualifiedColumns) {
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id, name, amount FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id");
  const BoundQuery bound = bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()});
  ASSERT_TRUE(bound.join.has_value());
  ASSERT_EQ(bound.join->steps.size(), 1u);
  // customer_id is orders_schema()'s index 1, customers_schema()'s index 0.
  EXPECT_EQ(bound.join->steps[0].combined_key_index, 1u);
  EXPECT_EQ(bound.join->steps[0].source_key_index, 0u);

  ASSERT_EQ(bound.select_list.size(), 3u);
  // o.order_id -> orders_schema() index 0 (left side, no offset).
  const auto* order_id = dynamic_cast<const ColumnExpression*>(bound.select_list[0].expr.get());
  ASSERT_NE(order_id, nullptr);
  EXPECT_EQ(order_id->column_index(), 0u);
  // unqualified `name` only exists on the right (customers) side -> combined
  // index = orders_schema().field_count() (3) + 1 (name's local index).
  const auto* name = dynamic_cast<const ColumnExpression*>(bound.select_list[1].expr.get());
  ASSERT_NE(name, nullptr);
  EXPECT_EQ(name->column_index(), 4u);
  // unqualified `amount` only exists on the left (orders) side.
  const auto* amount = dynamic_cast<const ColumnExpression*>(bound.select_list[2].expr.get());
  ASSERT_NE(amount, nullptr);
  EXPECT_EQ(amount->column_index(), 2u);
}

TEST(Binder, JoinRejectsAmbiguousUnqualifiedColumn) {
  // Both sides have a `customer_id` column; referencing it unqualified
  // outside the ON condition must fail rather than silently pick one side.
  const auto stmt = sql::parse_sql(
      "SELECT customer_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinRejectsUnknownTableQualifier) {
  const auto stmt = sql::parse_sql(
      "SELECT z.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinRejectsNonEqualityCondition) {
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id > c.customer_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinRejectsConditionComparingTwoColumnsFromTheSameSide) {
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = o.order_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinRejectsMismatchedKeyTypes) {
  // customer_id is INT64 on both sides in the fixture schemas; compare
  // against a STRING column on the other side instead to force a type
  // mismatch (implicit numeric promotion would insert a CastExpression,
  // which this rejects since it's no longer a bare ColumnExpression).
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.name");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinOnConditionWithRightSideOnlyAuxiliaryPredicateIsAcceptedAndRebased) {
  // TPC-H Q13's own shape: an ON clause combining the required equality key
  // with an extra predicate that references only the newly-joined (right)
  // source's own columns -- must still resolve the key normally, and the
  // extra predicate must come back as right_prefilter, rebased to the
  // customers source's own 0-based schema (name is index 1 there, not the
  // combined row's index 4).
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id AND c.name = 'Bob'");
  const BoundQuery bound = bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()});
  ASSERT_TRUE(bound.join.has_value());
  ASSERT_EQ(bound.join->steps.size(), 1u);
  EXPECT_EQ(bound.join->steps[0].combined_key_index, 1u);
  EXPECT_EQ(bound.join->steps[0].source_key_index, 0u);

  ASSERT_NE(bound.join->steps[0].right_prefilter, nullptr);
  const auto* prefilter = dynamic_cast<const BinaryExpression*>(bound.join->steps[0].right_prefilter.get());
  ASSERT_NE(prefilter, nullptr);
  EXPECT_EQ(prefilter->op(), BinaryOperator::Equal);
  const auto* name_column = dynamic_cast<const ColumnExpression*>(prefilter->left().get());
  ASSERT_NE(name_column, nullptr);
  EXPECT_EQ(name_column->column_index(), 1u);  // customers_schema()'s own "name" index, not 4.
}

TEST(Binder, JoinOnConditionWithLeftSideAuxiliaryPredicateIsRejected) {
  // `o.amount` (already-joined/left side) in an auxiliary ON conjunct: not
  // equivalent to a right-side pre-filter (a LEFT OUTER JOIN would need to
  // null-extend, not drop, a left row failing it) -- rejected unconditionally
  // regardless of join type, see extract_join_step_keys()'s own comment.
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id AND o.amount > 100");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinOnConditionWithCrossSideAuxiliaryPredicateIsRejected) {
  // `o.order_id > c.customer_id` spans both sides -- neither a valid
  // second equality key nor a single-side-only auxiliary predicate.
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id AND o.order_id > c.customer_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinOnConditionWithTwoCandidateEqualityKeysIsRejected) {
  // Two different cross-side equalities ANDed together -- ambiguous, since
  // HashJoinOperator's cudf::hash_join needs exactly one equality key.
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id AND o.order_id = c.customer_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

TEST(Binder, JoinStarExpandsBothSidesInOrder) {
  // Deliberately non-colliding field names on both sides: SELECT * would
  // otherwise hit the ordinary "duplicate output column name" check twice
  // over for a shared name like orders_schema()/customers_schema()'s
  // `customer_id` -- a pre-existing, correct rejection, not a JOIN-star
  // bug, but not what this test is trying to isolate.
  const Schema left({Field{"order_id", int64_type(false)}, Field{"cust_id", int64_type(false)}});
  const Schema right({Field{"cust_key", int64_type(false)}, Field{"name", string_type(false)}});
  const auto stmt = sql::parse_sql(
      "SELECT * FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.cust_id = c.cust_key");
  const BoundQuery bound = bind_query(stmt, std::vector<Schema>{left, right});
  ASSERT_EQ(bound.select_list.size(), 4u);
  EXPECT_EQ(bound.select_list[0].output_name, "order_id");
  EXPECT_EQ(bound.select_list[1].output_name, "cust_id");
  EXPECT_EQ(bound.select_list[2].output_name, "cust_key");
  EXPECT_EQ(bound.select_list[3].output_name, "name");
}

TEST(Binder, JoinStarWithCollidingColumnNamesIsRejected) {
  const auto stmt = sql::parse_sql(
      "SELECT * FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema()})),
               BindingError);
}

// Regression test: a 3+-way JOIN chain used to be rejected outright at
// parse time; this pins down that binding one all the way through
// actually resolves columns correctly across every source in the chain,
// not just the AST shape (see sql_parser_test.cpp's own coverage of that
// part). `n.customer_id` in the third step's ON condition resolves against
// the *combined* [orders, customers] schema so far (not just
// customers_schema(), the immediately-preceding source) -- exercising
// BoundJoinStep::combined_key_index's whole point.
TEST(Binder, ThreeWayJoinResolvesColumnsAcrossEveryStep) {
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id, c.name AS c_name, n.name AS n_name FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id "
      "JOIN read_parquet('/n.parquet') AS n ON o.customer_id = n.customer_id");
  const BoundQuery bound =
      bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema(), nations_schema()});
  ASSERT_TRUE(bound.join.has_value());
  ASSERT_EQ(bound.join->steps.size(), 2u);
  // Step 0: customers joins onto orders via customer_id (orders index 1,
  // customers index 0).
  EXPECT_EQ(bound.join->steps[0].combined_key_index, 1u);
  EXPECT_EQ(bound.join->steps[0].source_key_index, 0u);
  // Step 1: nations joins via o.customer_id -- still index 1 in the
  // combined [orders, customers] schema so far (orders_schema().field_count()
  // == 3, so customers' own fields start at 3, unrelated to this
  // reference), confirming the condition was resolved against orders
  // (source 0), not customers (source 1, the immediately-preceding one).
  EXPECT_EQ(bound.join->steps[1].combined_key_index, 1u);
  EXPECT_EQ(bound.join->steps[1].source_key_index, 0u);

  ASSERT_EQ(bound.select_list.size(), 3u);
  const auto* order_id = dynamic_cast<const ColumnExpression*>(bound.select_list[0].expr.get());
  ASSERT_NE(order_id, nullptr);
  EXPECT_EQ(order_id->column_index(), 0u);  // orders.order_id, no offset
  const auto* c_name = dynamic_cast<const ColumnExpression*>(bound.select_list[1].expr.get());
  ASSERT_NE(c_name, nullptr);
  EXPECT_EQ(c_name->column_index(), 4u);  // customers.name: 3 (orders fields) + 1
  const auto* n_name = dynamic_cast<const ColumnExpression*>(bound.select_list[2].expr.get());
  ASSERT_NE(n_name, nullptr);
  EXPECT_EQ(n_name->column_index(), 6u);  // nations.name: 3 + 2 (customers fields) + 1
}

TEST(Binder, ThreeWayJoinRejectsAmbiguousUnqualifiedColumnFromNonAdjacentSources) {
  // "name" is present on both customers_schema() (source 1) and
  // nations_schema() (source 2) -- not adjacent to each other in the
  // "which one is the immediately-preceding source" sense, but still must
  // be rejected the same as any other cross-source ambiguity.
  const auto stmt = sql::parse_sql(
      "SELECT name FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id "
      "JOIN read_parquet('/n.parquet') AS n ON o.customer_id = n.customer_id");
  EXPECT_THROW(
      (void)(bind_query(stmt, std::vector<Schema>{orders_schema(), customers_schema(), nations_schema()})),
      BindingError);
}

TEST(Binder, HavingReferencingAggregateBinds) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING SUM(amount) > 100");
  const BoundQuery bound = bind_query(stmt, sales_schema());
  ASSERT_NE(bound.having, nullptr);
  EXPECT_EQ(bound.having->result_type().id, TypeId::Boolean);
  const auto* comparison = dynamic_cast<const BinaryExpression*>(bound.having.get());
  ASSERT_NE(comparison, nullptr);
  EXPECT_EQ(comparison->op(), BinaryOperator::Greater);
  // HAVING's own SUM(amount) is structurally identical to the SELECT
  // list's own -- register_aggregate()'s dedup (logical_planner.cpp,
  // exercised indirectly here via the *bound* tree both reference) means
  // this doesn't need its own separate LogicalAggregate slot; verified
  // properly end to end in logical_plan_test.cpp, since bind_query() alone
  // doesn't build the LogicalAggregate.
  const auto* left = dynamic_cast<const AggregateExpression*>(comparison->left().get());
  ASSERT_NE(left, nullptr);
  EXPECT_EQ(left->function(), AggregateFunction::Sum);
}

TEST(Binder, HavingWithoutAggregateOrGroupByIsRejected) {
  // hsql's own grammar only reaches HAVING through `GROUP BY ... HAVING
  // ...` (opt_group's own production), so a real `parse_sql()` call can
  // never actually produce an AstSelectStatement with `having != nullptr`
  // and `group_by.empty()`/no aggregate at the same time -- this
  // specific rejection in bind_query_common() is a defensive fallback,
  // not reachable through the public parsing API today. Construct the
  // AST by hand to exercise it directly anyway, in case that constraint
  // ever changes.
  auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet')");
  stmt.having = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region = 'A'").where;
  EXPECT_THROW((void)bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, HavingReferencingUngroupedColumnIsRejected) {
  // "amount" is neither a GROUP BY key nor wrapped in an aggregate here --
  // the same rule an ungrouped SELECT-list column already gets.
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') "
      "GROUP BY region HAVING amount > 100");
  EXPECT_THROW((void)bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, HavingMustBeBoolean) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') GROUP BY region HAVING "
      "SUM(amount)");
  EXPECT_THROW((void)bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsSubqueryOutsideHaving) {
  // resolve_subqueries() (QueryEngine::plan_logical()) only ever walks
  // AstSelectStatement::having -- a subquery anywhere else (WHERE here)
  // reaches the binder unresolved, which must reject it clearly rather
  // than crash or misinterpret it.
  const auto stmt = sql::parse_sql(
      "SELECT a FROM read_parquet('/x.parquet') WHERE a > (SELECT SUM(a) FROM read_parquet('/x.parquet'))");
  EXPECT_THROW((void)bind_query(stmt, Schema({Field{"a", int64_type(false)}})), BindingError);
}

TEST(Binder, RejectsUnresolvedExists) {
  // sql::rewrite_exists_subqueries() (QueryEngine::plan_logical()) is what
  // actually rewrites a top-level WHERE-clause EXISTS into a join step --
  // bind_query() alone (as called directly here, bypassing QueryEngine)
  // never runs that rewrite, so a query whose FROM has no alias at all
  // (single-table binder mode, where a qualified reference wouldn't even
  // be legal) reaches bind_node(const AstExists&, bool) unresolved.
  const auto stmt = sql::parse_sql(
      "SELECT a FROM read_parquet('/x.parquet') WHERE EXISTS "
      "(SELECT * FROM read_parquet('/y.parquet') AS b WHERE b.k = a)");
  EXPECT_THROW((void)bind_query(stmt, Schema({Field{"a", int64_type(false)}})), BindingError);
}

TEST(Binder, RejectsUnresolvedInSubquery) {
  // sql::resolve_in_subqueries() (QueryEngine::plan_logical()) is what
  // actually resolves an IN-subquery's AstIn::subquery field into a real
  // literal list before binding -- bind_query() alone (as called directly
  // here, bypassing QueryEngine) never runs that resolution, so this must
  // reach bind_node(const AstIn&, bool)'s own dedicated rejection rather
  // than being silently misinterpreted as an empty IN list.
  const auto stmt = sql::parse_sql(
      "SELECT a FROM read_parquet('/x.parquet') WHERE a IN (SELECT a FROM read_parquet('/x.parquet'))");
  EXPECT_THROW((void)bind_query(stmt, Schema({Field{"a", int64_type(false)}})), BindingError);
}

TEST(Binder, QualifiedColumnRefInSingleTableModeIsRejected) {
  // input_schema_ != nullptr (single-table/non-JOIN mode) -- a qualified
  // reference like `a.region` has no side to disambiguate here, unlike JOIN
  // mode where it picks a source.
  const auto stmt = sql::parse_sql("SELECT a.region FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, JoinRejectsKeysOfDifferentNumericWidthRequiringAnImplicitCast) {
  // Unlike JoinRejectsMismatchedKeyTypes above (STRING vs. INT64, which
  // fails earlier in combine_binary()'s "incompatible comparison" check),
  // INT32 and INT64 are numerically compatible -- combine_binary() promotes
  // and wraps the INT32 side in an implicit CastExpression, so the ON
  // condition is no longer a bare `<column> = <column>` by the time it
  // reaches extract_join_step_keys(), which is what this test actually
  // exercises.
  const auto stmt = sql::parse_sql(
      "SELECT o.order_id FROM read_parquet('/o.parquet') AS o "
      "JOIN read_parquet('/c.parquet') AS c ON o.customer_id = c.customer_id");
  EXPECT_THROW((void)(bind_query(stmt, std::vector<Schema>{orders_schema(), narrow_customers_schema()})),
               BindingError);
}

TEST(Binder, SumAndAvgOnNonNumericColumnAreRejected) {
  const auto sum_stmt = sql::parse_sql("SELECT SUM(region) FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(sum_stmt, sales_schema())), BindingError);

  const auto avg_stmt = sql::parse_sql("SELECT AVG(region) FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(avg_stmt, sales_schema())), BindingError);
}

TEST(Binder, LikeWithNonLiteralPatternIsRejected) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region LIKE region");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, ArithmeticOnNonNumericOperandsIsRejected) {
  const auto stmt = sql::parse_sql("SELECT region + amount FROM read_parquet('/x.parquet')");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, AndOrWithNonBooleanOperandIsRejected) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount AND region = 'A'");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, NotWithNonBooleanOperandIsRejected) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE NOT amount");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, BetweenBoundTypeMismatchedWithValueTypeIsRejected) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region BETWEEN 1 AND 10");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, GroupByNonColumnRefExpressionIsRejected) {
  const auto stmt = sql::parse_sql("SELECT amount FROM read_parquet('/x.parquet') GROUP BY amount + 1");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, AggregateOrderByOnNonColumnRefExpressionIsRejected) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') GROUP BY region "
      "ORDER BY total + 1");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

TEST(Binder, AggregateOrderByOnUnknownOutputNameIsRejected) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) AS total FROM read_parquet('/x.parquet') GROUP BY region "
      "ORDER BY nonexistent");
  EXPECT_THROW((void)(bind_query(stmt, sales_schema())), BindingError);
}

}  // namespace
}  // namespace kernellake
