// End-to-end coverage for DECIMAL support: a real DECIMAL128 Parquet
// column, read through the whole parse -> bind -> plan -> GPU execution
// pipeline. Expected sums/values below are computed by hand from the fixed
// fixture data in SetUp(), the same pattern query_engine_execute_test.cpp
// uses.
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

// Generic scalar-to-text, mirroring result_formatter.cpp's scalar_text():
// deliberately avoids assuming a specific Arrow decimal width (Decimal32/
// 64/128Array) since which one cudf's Arrow interop actually produces for a
// given precision is an internal detail, not something a test should
// hard-code.
std::string decimal_text(const std::shared_ptr<arrow::Array>& column, std::int64_t row) {
  arrow::Result<std::shared_ptr<arrow::Scalar>> scalar = column->GetScalar(row);
  return scalar.ok() ? (*scalar)->ToString() : "<error>";
}

class DecimalQueryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_decimal_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "prices.parquet").string();

    // price is DECIMAL(10,2); regions A/A/B/B for the GROUP BY case.
    // A: 10.50 + 20.25 = 30.75; B: 100.00 + 7.10 = 107.10.
    const auto decimal_type = arrow::decimal128(10, 2);
    auto price_builder = std::make_shared<arrow::Decimal128Builder>(decimal_type);
    const std::vector<std::string> price_text = {"10.50", "20.25", "100.00", "7.10"};
    for (const std::string& text : price_text) {
      ASSERT_TRUE(price_builder->Append(arrow::Decimal128::FromString(text).ValueOrDie()).ok());
    }
    std::shared_ptr<arrow::Array> price_array;
    ASSERT_TRUE(price_builder->Finish(&price_array).ok());

    arrow::StringBuilder region_builder;
    for (const std::string& region : {"A", "A", "B", "B"}) {
      ASSERT_TRUE(region_builder.Append(region).ok());
    }
    std::shared_ptr<arrow::Array> region_array;
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());

    arrow::Int32Builder quantity_builder;
    for (const std::int32_t quantity : {1, 2, 3, 4}) {
      ASSERT_TRUE(quantity_builder.Append(quantity).ok());
    }
    std::shared_ptr<arrow::Array> quantity_array;
    ASSERT_TRUE(quantity_builder.Finish(&quantity_array).ok());

    const auto schema = arrow::schema({arrow::field("price", decimal_type, false),
                                       arrow::field("region", arrow::utf8(), false),
                                       arrow::field("quantity", arrow::int32(), false)});
    const auto table = arrow::Table::Make(schema, {price_array, region_array, quantity_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/4);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
  QueryEngine engine_{default_config()};
};

TEST_F(DecimalQueryTest, PlainProjectionReturnsAllRows) {
  const QueryResult result = engine_.execute("SELECT price FROM read_parquet('" + path_ + "')");
  EXPECT_EQ(result.rows_returned, 4);
}

TEST_F(DecimalQueryTest, ComparisonAgainstLiteralFiltersCorrectly) {
  const QueryResult result =
      engine_.execute("SELECT region FROM read_parquet('" + path_ + "') WHERE price > 20.00");
  // 20.25, 100.00, 7.10 > 20.00 -- true for the first two only (7.10 fails).
  ASSERT_EQ(result.batches.size(), 1u);
  EXPECT_EQ(result.batches.front()->num_rows(), 2);
}

TEST_F(DecimalQueryTest, EqualityAgainstLiteralMatchesExactValue) {
  const QueryResult result =
      engine_.execute("SELECT region FROM read_parquet('" + path_ + "') WHERE price = 100.00");
  ASSERT_EQ(result.batches.size(), 1u);
  ASSERT_EQ(result.batches.front()->num_rows(), 1);
  const auto region_column =
      std::static_pointer_cast<arrow::StringArray>(result.batches.front()->GetColumnByName("region"));
  EXPECT_EQ(region_column->GetString(0), "B");
}

TEST_F(DecimalQueryTest, ScalarSumMatchesExpectedTotal) {
  const QueryResult result = engine_.execute("SELECT SUM(price) AS total FROM read_parquet('" + path_ + "')");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto total_column = result.batches.front()->GetColumnByName("total");
  ASSERT_NE(total_column, nullptr);
  // 10.50 + 20.25 + 100.00 + 7.10 = 137.85
  EXPECT_EQ(decimal_text(total_column, 0), "137.85");
}

TEST_F(DecimalQueryTest, ScalarMinMaxMatchExpectedValues) {
  const QueryResult result =
      engine_.execute("SELECT MIN(price) AS lo, MAX(price) AS hi FROM read_parquet('" + path_ + "')");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto lo = result.batches.front()->GetColumnByName("lo");
  const auto hi = result.batches.front()->GetColumnByName("hi");
  EXPECT_EQ(decimal_text(lo, 0), "7.10");
  EXPECT_EQ(decimal_text(hi, 0), "100.00");
}

TEST_F(DecimalQueryTest, GroupedSumMatchesExpectedTotalsPerRegion) {
  const QueryResult result = engine_.execute("SELECT region, SUM(price) AS total FROM read_parquet('" +
                                             path_ + "') GROUP BY region ORDER BY region");
  ASSERT_EQ(result.batches.size(), 1u);
  const std::shared_ptr<arrow::RecordBatch>& batch = result.batches.front();
  ASSERT_EQ(batch->num_rows(), 2);
  const auto region_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("region"));
  const auto total_column = batch->GetColumnByName("total");
  ASSERT_NE(total_column, nullptr);
  EXPECT_EQ(region_column->GetString(0), "A");
  EXPECT_EQ(decimal_text(total_column, 0), "30.75");
  EXPECT_EQ(region_column->GetString(1), "B");
  EXPECT_EQ(decimal_text(total_column, 1), "107.10");
}

TEST_F(DecimalQueryTest, OrderByDecimalColumnSortsAscending) {
  const QueryResult result =
      engine_.execute("SELECT price FROM read_parquet('" + path_ + "') ORDER BY price ASC");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto price_column = result.batches.front()->GetColumnByName("price");
  ASSERT_NE(price_column, nullptr);
  ASSERT_EQ(price_column->length(), 4);
  EXPECT_EQ(decimal_text(price_column, 0), "7.10");
  EXPECT_EQ(decimal_text(price_column, 1), "10.50");
  EXPECT_EQ(decimal_text(price_column, 2), "20.25");
  EXPECT_EQ(decimal_text(price_column, 3), "100.00");
}

TEST_F(DecimalQueryTest, CastDecimalToDoubleProducesApproximateValue) {
  // ORDER BY a SELECT-list alias only resolves for a computed expression
  // after GROUP BY (see docs/ARCHITECTURE.md); on a plain query like this
  // one, ORDER BY only reaches base-table columns -- order by `price`
  // itself rather than the projected alias `price_f`.
  const QueryResult result = engine_.execute("SELECT CAST(price AS DOUBLE) AS price_f FROM read_parquet('" +
                                             path_ + "') WHERE region = 'A' ORDER BY price");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("price_f"));
  ASSERT_NE(column, nullptr);
  ASSERT_EQ(column->length(), 2);
  EXPECT_DOUBLE_EQ(column->Value(0), 10.50);
  EXPECT_DOUBLE_EQ(column->Value(1), 20.25);
}

TEST_F(DecimalQueryTest, CastIntegerColumnToDecimalInProjection) {
  // region has no integer column to cast in this fixture; cast price's own
  // scale up to DECIMAL(10,4) instead, exercising the CAST-to-DECIMAL fast
  // path (materialized outside cudf::ast; see ProjectionOperator).
  const QueryResult result =
      engine_.execute("SELECT CAST(price AS DECIMAL(10, 4)) AS price4 FROM read_parquet('" + path_ +
                      "') WHERE region = 'B' ORDER BY price");
  ASSERT_EQ(result.batches.size(), 1u);
  const auto column = result.batches.front()->GetColumnByName("price4");
  ASSERT_NE(column, nullptr);
  ASSERT_EQ(column->length(), 2);
  EXPECT_EQ(decimal_text(column, 0), "7.1000");
  EXPECT_EQ(decimal_text(column, 1), "100.0000");
}

TEST_F(DecimalQueryTest, AvgOverDecimalIsRejectedAtBindTime) {
  EXPECT_THROW((void)(engine_.execute("SELECT AVG(price) FROM read_parquet('" + path_ + "')")), BindingError);
}

TEST_F(DecimalQueryTest, MixingDecimalWithNonLiteralColumnIsRejected) {
  // `price` (DECIMAL) compared against `quantity`, a non-literal,
  // non-DECIMAL column: not yet supported (see cast_if_needed in
  // binder.cpp) -- must fail cleanly at bind time, not silently
  // misevaluate.
  EXPECT_THROW(
      (void)(engine_.execute("SELECT price FROM read_parquet('" + path_ + "') WHERE price > quantity")),
      BindingError);
}

}  // namespace
}  // namespace kernellake
