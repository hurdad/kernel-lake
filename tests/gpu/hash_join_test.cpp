// End-to-end coverage for two-table INNER JOIN support: real Parquet files
// on each side, through the whole parse -> bind -> plan -> optimize ->
// physical plan -> GPU execution pipeline (HashJoinOperator built on
// cudf::hash_join). Expected values below are computed by hand from the
// fixed fixture data in SetUp(), the same pattern query_engine_execute_test.cpp
// and decimal_test.cpp use; also cross-validated against DuckDB manually
// during development (see docs/ROADMAP.md).
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <map>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class HashJoinQueryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_hash_join_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    orders_path_ = (dir_ / "orders.parquet").string();
    customers_path_ = (dir_ / "customers.parquet").string();

    // orders: 5 rows, one (customer_id=99) with no matching customer.
    // customer 10 has two orders (100+50=150), customer 20 has one (75),
    // customer 30 has one (20).
    {
      arrow::Int64Builder order_id_builder;
      arrow::Int64Builder customer_id_builder;
      arrow::DoubleBuilder amount_builder;
      const std::vector<std::int64_t> order_ids = {1, 2, 3, 4, 5};
      const std::vector<std::int64_t> customer_ids = {10, 10, 20, 30, 99};
      const std::vector<double> amounts = {100.0, 50.0, 75.0, 20.0, 5.0};
      for (std::size_t i = 0; i < order_ids.size(); ++i) {
        ASSERT_TRUE(order_id_builder.Append(order_ids[i]).ok());
        ASSERT_TRUE(customer_id_builder.Append(customer_ids[i]).ok());
        ASSERT_TRUE(amount_builder.Append(amounts[i]).ok());
      }
      std::shared_ptr<arrow::Array> order_id_array, customer_id_array, amount_array;
      ASSERT_TRUE(order_id_builder.Finish(&order_id_array).ok());
      ASSERT_TRUE(customer_id_builder.Finish(&customer_id_array).ok());
      ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
      const auto schema = arrow::schema({arrow::field("order_id", arrow::int64(), false),
                                         arrow::field("customer_id", arrow::int64(), false),
                                         arrow::field("amount", arrow::float64(), false)});
      const auto table = arrow::Table::Make(schema, {order_id_array, customer_id_array, amount_array});
      auto sink = arrow::io::FileOutputStream::Open(orders_path_).ValueOrDie();
      const arrow::Status status =
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5);
      ASSERT_TRUE(status.ok()) << status.ToString();
    }
    // customers: 3 rows (10, 20, 30) -- no customer 99, matching orders'
    // unmatched row.
    {
      arrow::Int64Builder customer_id_builder;
      arrow::StringBuilder name_builder;
      const std::vector<std::int64_t> customer_ids = {10, 20, 30};
      const std::vector<std::string> names = {"Alice", "Bob", "Carol"};
      for (std::size_t i = 0; i < customer_ids.size(); ++i) {
        ASSERT_TRUE(customer_id_builder.Append(customer_ids[i]).ok());
        ASSERT_TRUE(name_builder.Append(names[i]).ok());
      }
      std::shared_ptr<arrow::Array> customer_id_array, name_array;
      ASSERT_TRUE(customer_id_builder.Finish(&customer_id_array).ok());
      ASSERT_TRUE(name_builder.Finish(&name_array).ok());
      const auto schema = arrow::schema(
          {arrow::field("customer_id", arrow::int64(), false), arrow::field("name", arrow::utf8(), false)});
      const auto table = arrow::Table::Make(schema, {customer_id_array, name_array});
      auto sink = arrow::io::FileOutputStream::Open(customers_path_).ValueOrDie();
      const arrow::Status status =
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/3);
      ASSERT_TRUE(status.ok()) << status.ToString();
    }
  }

  void TearDown() override { fs::remove_all(dir_); }

  std::string join_clause(const std::string& condition = "o.customer_id = c.customer_id") const {
    return "FROM read_parquet('" + orders_path_ + "') AS o JOIN read_parquet('" + customers_path_ +
           "') AS c ON " + condition;
  }

  fs::path dir_;
  std::string orders_path_;
  std::string customers_path_;
  QueryEngine engine_{default_config()};
};

TEST_F(HashJoinQueryTest, InnerJoinExcludesUnmatchedRowsFromBothSides) {
  const QueryResult result =
      engine_.execute("SELECT o.order_id, c.name, o.amount " + join_clause() + " ORDER BY o.order_id");
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  // order_id=5 (customer_id=99) has no matching customer and must be excluded.
  ASSERT_EQ(batch->num_rows(), 4);

  const auto order_id_column =
      std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("order_id"));
  const auto name_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("name"));
  const auto amount_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("amount"));
  ASSERT_NE(order_id_column, nullptr);
  ASSERT_NE(name_column, nullptr);
  ASSERT_NE(amount_column, nullptr);

  const std::vector<std::int64_t> expected_order_ids = {1, 2, 3, 4};
  const std::vector<std::string> expected_names = {"Alice", "Alice", "Bob", "Carol"};
  const std::vector<double> expected_amounts = {100.0, 50.0, 75.0, 20.0};
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    EXPECT_EQ(order_id_column->Value(i), expected_order_ids[static_cast<std::size_t>(i)]);
    EXPECT_EQ(name_column->GetString(i), expected_names[static_cast<std::size_t>(i)]);
    EXPECT_DOUBLE_EQ(amount_column->Value(i), expected_amounts[static_cast<std::size_t>(i)]);
  }
}

TEST_F(HashJoinQueryTest, WhereClauseAfterJoinFiltersCorrectly) {
  const QueryResult result = engine_.execute("SELECT o.order_id " + join_clause() + " WHERE o.amount > 30");
  ASSERT_EQ(result.batches.size(), 1u);
  EXPECT_EQ(result.batches.front()->num_rows(), 3);  // 100, 50, 75 -- not 20 or 5
}

TEST_F(HashJoinQueryTest, GroupedSumOverJoinedRowsMatchesExpectedTotals) {
  const QueryResult result = engine_.execute("SELECT c.name, SUM(o.amount) AS total " + join_clause() +
                                             " GROUP BY c.name ORDER BY c.name");
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 3);
  const auto name_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("name"));
  const auto total_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("total"));
  ASSERT_NE(name_column, nullptr);
  ASSERT_NE(total_column, nullptr);

  std::map<std::string, double> totals_by_name;
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    totals_by_name[name_column->GetString(i)] = total_column->Value(i);
  }
  ASSERT_EQ(totals_by_name.size(), 3u);
  EXPECT_DOUBLE_EQ(totals_by_name.at("Alice"), 150.0);  // 100 + 50
  EXPECT_DOUBLE_EQ(totals_by_name.at("Bob"), 75.0);
  EXPECT_DOUBLE_EQ(totals_by_name.at("Carol"), 20.0);
}

TEST_F(HashJoinQueryTest, EmptyBuildSideProducesNoRows) {
  // A genuinely empty customers file (0 rows, not just a WHERE clause that
  // filters everything post-join) -- exercises HashJoinOperator's own
  // right_is_empty_ path directly, where the join never even constructs a
  // cudf::hash_join.
  const std::string empty_customers_path = (dir_ / "empty_customers.parquet").string();
  arrow::Int64Builder customer_id_builder;
  arrow::StringBuilder name_builder;
  std::shared_ptr<arrow::Array> customer_id_array, name_array;
  ASSERT_TRUE(customer_id_builder.Finish(&customer_id_array).ok());
  ASSERT_TRUE(name_builder.Finish(&name_array).ok());
  const auto schema = arrow::schema(
      {arrow::field("customer_id", arrow::int64(), false), arrow::field("name", arrow::utf8(), false)});
  const auto table = arrow::Table::Make(schema, {customer_id_array, name_array});
  auto sink = arrow::io::FileOutputStream::Open(empty_customers_path).ValueOrDie();
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/1).ok());

  const QueryResult result =
      engine_.execute("SELECT o.order_id FROM read_parquet('" + orders_path_ + "') AS o JOIN read_parquet('" +
                      empty_customers_path + "') AS c ON o.customer_id = c.customer_id");
  EXPECT_EQ(result.rows_returned, 0);
}

TEST_F(HashJoinQueryTest, ScalarCountOverJoinMatchesExpectedRowCount) {
  const QueryResult result = engine_.execute("SELECT COUNT(*) AS n " + join_clause());
  ASSERT_EQ(result.batches.size(), 1u);
  const auto n_column =
      std::static_pointer_cast<arrow::Int64Array>(result.batches.front()->GetColumnByName("n"));
  ASSERT_NE(n_column, nullptr);
  EXPECT_EQ(n_column->Value(0), 4);
}

TEST_F(HashJoinQueryTest, RejectsLeftJoinAtParseTime) {
  EXPECT_THROW((void)(engine_.execute("SELECT o.order_id FROM read_parquet('" + orders_path_ +
                               "') AS o LEFT JOIN read_parquet('" + customers_path_ +
                               "') AS c ON o.customer_id = c.customer_id")),
               SqlError);
}

}  // namespace
}  // namespace kernellake
