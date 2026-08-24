// End-to-end coverage for EXISTS/NOT EXISTS support: real Parquet files on
// each side, through the whole parse -> sql::rewrite_exists_subqueries() ->
// bind -> plan -> optimize -> physical plan -> GPU execution pipeline
// (SemiAntiJoinOperator built on cudf::filtered_join). Same fixture shape
// as hash_join_test.cpp's own HashJoinQueryTest (orders/customers), so
// results can be sanity-checked against that file's own hand-computed
// values for the analogous JOIN-with-auxiliary-predicate shape.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class ExistsQueryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_exists_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    orders_path_ = (dir_ / "orders.parquet").string();
    customers_path_ = (dir_ / "customers.parquet").string();

    // orders: 5 rows, one (customer_id=99) with no matching customer at
    // all. customer 10 has two orders, customer 20 one, customer 30 one.
    {
      arrow::Int64Builder order_id_builder;
      arrow::Int64Builder customer_id_builder;
      const std::vector<std::int64_t> order_ids = {1, 2, 3, 4, 5};
      const std::vector<std::int64_t> customer_ids = {10, 10, 20, 30, 99};
      for (std::size_t i = 0; i < order_ids.size(); ++i) {
        ASSERT_TRUE(order_id_builder.Append(order_ids[i]).ok());
        ASSERT_TRUE(customer_id_builder.Append(customer_ids[i]).ok());
      }
      std::shared_ptr<arrow::Array> order_id_array, customer_id_array;
      ASSERT_TRUE(order_id_builder.Finish(&order_id_array).ok());
      ASSERT_TRUE(customer_id_builder.Finish(&customer_id_array).ok());
      const auto schema = arrow::schema({arrow::field("order_id", arrow::int64(), false),
                                         arrow::field("customer_id", arrow::int64(), false)});
      const auto table = arrow::Table::Make(schema, {order_id_array, customer_id_array});
      auto sink = arrow::io::FileOutputStream::Open(orders_path_).ValueOrDie();
      const arrow::Status status =
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5);
      ASSERT_TRUE(status.ok()) << status.ToString();
    }
    // customers: 3 rows (10=Alice, 20=Bob, 30=Carol) -- no customer 99.
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

  // `extra` is an additional conjunct ANDed into the correlated
  // subquery's own WHERE clause, referencing only its own (customers)
  // columns -- e.g. "AND c.name <> 'Bob'", exercising the same "single
  // equality correlation plus a right-side-only auxiliary predicate"
  // shape the JOIN ON-clause auxiliary predicate feature already
  // supports (extract_join_step_keys() is the exact same machinery
  // both go through).
  std::string exists_clause(const std::string& extra = "") const {
    return "FROM read_parquet('" + orders_path_ + "') AS o WHERE EXISTS (SELECT * FROM read_parquet('" +
           customers_path_ + "') AS c WHERE c.customer_id = o.customer_id" + extra + ")";
  }

  std::string not_exists_clause(const std::string& extra = "") const {
    return "FROM read_parquet('" + orders_path_ + "') AS o WHERE NOT EXISTS (SELECT * FROM read_parquet('" +
           customers_path_ + "') AS c WHERE c.customer_id = o.customer_id" + extra + ")";
  }

  fs::path dir_;
  std::string orders_path_;
  std::string customers_path_;
  QueryEngine engine_{default_config()};
};

TEST_F(ExistsQueryTest, ExistsKeepsOnlyOrdersWithAMatchingCustomer) {
  const QueryResult result = engine_.execute("SELECT o.order_id " + exists_clause() + " ORDER BY o.order_id");
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  // order_id=5 (customer_id=99) has no matching customer -- excluded.
  ASSERT_EQ(batch->num_rows(), 4);
  const auto order_id_column =
      std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("order_id"));
  ASSERT_NE(order_id_column, nullptr);
  const std::vector<std::int64_t> expected = {1, 2, 3, 4};
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    EXPECT_EQ(order_id_column->Value(i), expected[static_cast<std::size_t>(i)]);
  }
}

TEST_F(ExistsQueryTest, NotExistsKeepsOnlyOrdersWithNoMatchingCustomer) {
  const QueryResult result =
      engine_.execute("SELECT o.order_id " + not_exists_clause() + " ORDER BY o.order_id");
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 1);
  const auto order_id_column =
      std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("order_id"));
  ASSERT_NE(order_id_column, nullptr);
  EXPECT_EQ(order_id_column->Value(0), 5);
}

// The correlated subquery's own auxiliary predicate (referencing only its
// own source's columns, "c.name <> 'Bob'") is pushed down as a pre-filter
// before the semi-join runs -- the exact same extract_join_step_keys()
// mechanism a real JOIN ON-clause auxiliary predicate already goes
// through. Customer 20 (Bob) is filtered out of the build side entirely,
// so order 3 (customer_id=20) has no match left, same as order 5
// (customer_id=99, never had one).
TEST_F(ExistsQueryTest, ExistsWithAuxiliaryPredicateFiltersBuildSideBeforeMatching) {
  const QueryResult result =
      engine_.execute("SELECT o.order_id " + exists_clause(" AND c.name <> 'Bob'") + " ORDER BY o.order_id");
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 3);
  const auto order_id_column =
      std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("order_id"));
  ASSERT_NE(order_id_column, nullptr);
  const std::vector<std::int64_t> expected = {1, 2, 4};
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    EXPECT_EQ(order_id_column->Value(i), expected[static_cast<std::size_t>(i)]);
  }
}

TEST_F(ExistsQueryTest, NotExistsWithAuxiliaryPredicateKeepsFilteredOutRows) {
  const QueryResult result = engine_.execute(
      "SELECT o.order_id " + not_exists_clause(" AND c.name <> 'Bob'") + " ORDER BY o.order_id");
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 2);
  const auto order_id_column =
      std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("order_id"));
  ASSERT_NE(order_id_column, nullptr);
  const std::vector<std::int64_t> expected = {3, 5};
  for (std::int64_t i = 0; i < batch->num_rows(); ++i) {
    EXPECT_EQ(order_id_column->Value(i), expected[static_cast<std::size_t>(i)]);
  }
}

// An entirely empty build side: EXISTS can never match anything (empty
// result); NOT EXISTS's every probe row trivially "has no match" (the
// whole probe side passes through unchanged) -- exercises
// SemiAntiJoinOperator's own right_is_empty_ branch in next(), the
// SemiAntiJoinOperator-specific analog of HashJoinOperator's identical
// empty-build-side test.
TEST_F(ExistsQueryTest, EmptyBuildSideExistsProducesNoRowsNotExistsProducesAllRows) {
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

  const QueryResult exists_result =
      engine_.execute("SELECT o.order_id FROM read_parquet('" + orders_path_ +
                      "') AS o WHERE EXISTS (SELECT * FROM read_parquet('" + empty_customers_path +
                      "') AS c WHERE c.customer_id = o.customer_id)");
  EXPECT_EQ(exists_result.rows_returned, 0);

  const QueryResult not_exists_result =
      engine_.execute("SELECT o.order_id FROM read_parquet('" + orders_path_ +
                      "') AS o WHERE NOT EXISTS (SELECT * FROM read_parquet('" + empty_customers_path +
                      "') AS c WHERE c.customer_id = o.customer_id)");
  ASSERT_EQ(not_exists_result.rows_returned, 5);  // every order row, unchanged.
}

// EXISTS combined with an ordinary GROUP BY/aggregate over the surviving
// (semi-joined) rows -- exercises the semi-join sitting underneath the
// rest of a real query's own logical plan, not just a bare filter.
TEST_F(ExistsQueryTest, ExistsCombinesWithGroupedAggregate) {
  const QueryResult result = engine_.execute("SELECT COUNT(*) AS n " + exists_clause());
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 1);
  const auto count_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("n"));
  ASSERT_NE(count_column, nullptr);
  EXPECT_EQ(count_column->Value(0), 4);
}

TEST_F(ExistsQueryTest, ExistsCpuBackendMatchesGpuBackend) {
  const std::string sql =
      "SELECT o.order_id " + exists_clause(" AND c.name <> 'Bob'") + " ORDER BY o.order_id";
  const QueryResult gpu_result = engine_.execute(sql);

  EngineConfig cpu_config = default_config();
  cpu_config.engine.backend = "cpu";
  const QueryEngine cpu_engine(cpu_config);
  const QueryResult cpu_result = cpu_engine.execute(sql);

  ASSERT_EQ(gpu_result.batches.size(), 1u);
  ASSERT_EQ(cpu_result.batches.size(), 1u);
  const arrow::RecordBatch& gpu_batch = *gpu_result.batches.front();
  const arrow::RecordBatch& cpu_batch = *cpu_result.batches.front();
  ASSERT_EQ(gpu_batch.num_rows(), cpu_batch.num_rows());
  const auto gpu_order_id =
      std::static_pointer_cast<arrow::Int64Array>(gpu_batch.GetColumnByName("order_id"));
  const auto cpu_order_id =
      std::static_pointer_cast<arrow::Int64Array>(cpu_batch.GetColumnByName("order_id"));
  ASSERT_NE(gpu_order_id, nullptr);
  ASSERT_NE(cpu_order_id, nullptr);
  for (std::int64_t i = 0; i < gpu_batch.num_rows(); ++i) {
    EXPECT_EQ(gpu_order_id->Value(i), cpu_order_id->Value(i)) << "row " << i;
  }
}

}  // namespace
}  // namespace kernellake
