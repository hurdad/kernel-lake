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

Schema lineitem_schema() {
  return Schema({
      Field{"l_extendedprice", float64_type(false)},
      Field{"l_discount", float64_type(false)},
      Field{"l_shipdate", date32_type(false)},
      Field{"l_quantity", int32_type(false)},
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
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsUngroupedColumnInAggregateQuery) {
  const auto stmt = sql::parse_sql(
      "SELECT region, amount, SUM(amount) FROM read_parquet('/x.parquet') GROUP BY region");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, AllowsGroupedColumnInAggregateQuery) {
  const auto stmt = sql::parse_sql(
      "SELECT region, SUM(amount) FROM read_parquet('/x.parquet') GROUP BY region");
  EXPECT_NO_THROW(bind_query(stmt, sales_schema()));
}

TEST(Binder, RejectsIncompatibleComparison) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE region > 5");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsNonBooleanWhere) {
  const auto stmt = sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsDuplicateOutputNames) {
  const auto stmt = sql::parse_sql(
      "SELECT region AS x, amount AS x FROM read_parquet('/x.parquet')");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
}

TEST(Binder, RejectsAggregateInWhere) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE SUM(amount) > 0");
  EXPECT_THROW(bind_query(stmt, sales_schema()), BindingError);
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
  const auto stmt =
      sql::parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE a > 1.5");
  const BoundQuery bound = bind_query(stmt, schema);
  ASSERT_NE(bound.where, nullptr);
  EXPECT_EQ(bound.where->result_type().id, TypeId::Boolean);
}

TEST(Binder, NullLiteralComparisonBindsWithoutError) {
  const auto stmt =
      sql::parse_sql("SELECT region FROM read_parquet('/x.parquet') WHERE amount = NULL");
  EXPECT_NO_THROW(bind_query(stmt, sales_schema()));
}

}  // namespace
}  // namespace kernellake
