#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include "kernellake/execution_gpu/filter_operator.hpp"
#include "kernellake/execution_gpu/projection_operator.hpp"
#include "kernellake/execution_gpu/scalar_aggregate_operator.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

class VectorSourceOperator final : public PhysicalOperator {
 public:
  explicit VectorSourceOperator(std::vector<DeviceBatch> batches) : batches_(std::move(batches)) {}
  void open(ExecutionContext&) override { index_ = 0; }
  std::optional<DeviceBatch> next(ExecutionContext&) override {
    if (index_ >= batches_.size()) return std::nullopt;
    return std::move(batches_[index_++]);
  }
  void close(ExecutionContext&) override {}
  [[nodiscard]] std::string_view name() const noexcept override { return "VectorSource"; }
  [[nodiscard]] OperatorId id() const noexcept override { return 0; }

 private:
  std::vector<DeviceBatch> batches_;
  std::size_t index_ = 0;
};

ExecutionContext make_context() {
  return ExecutionContext{"test-query", 0,       nullptr, rmm::mr::get_current_device_resource_ref(),
                          nullptr,      nullptr, nullptr};
}

std::unique_ptr<cudf::column> filled_column(cudf::type_id type, double value, cudf::size_type num_rows) {
  auto column = cudf::make_numeric_column(cudf::data_type{type}, num_rows);
  cudf::mutable_column_view view = column->mutable_view();
  if (type == cudf::type_id::FLOAT64) {
    auto scalar = cudf::make_fixed_width_scalar<double>(value);
    cudf::fill_in_place(view, 0, num_rows, *scalar);
  } else {
    auto scalar = cudf::make_fixed_width_scalar<int32_t>(static_cast<int32_t>(value));
    cudf::fill_in_place(view, 0, num_rows, *scalar);
  }
  return column;
}

template <typename T>
T single_row_value(const DeviceBatch& batch, cudf::size_type column_index = 0) {
  T host_value{};
  cudaMemcpy(&host_value, batch.view().column(column_index).data<T>(), sizeof(T), cudaMemcpyDeviceToHost);
  return host_value;
}

bool single_row_is_null(const DeviceBatch& batch, cudf::size_type column_index = 0) {
  return batch.view().column(column_index).null_count() > 0;
}

TEST(ScalarAggregateOperator, SumAccumulatesAcrossMultipleBatches) {
  RmmEnvironment env(default_config());
  Schema schema({Field{"amount", float64_type(false)}});
  std::vector<DeviceBatch> batches;
  for (double value : {10.0, 20.0, 30.0}) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(filled_column(cudf::type_id::FLOAT64, value, 5));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }

  auto amount = std::make_shared<ColumnExpression>("amount", 0, float64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"}};

  ScalarAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                             std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);

  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 1u);
  // 5 rows of 10.0 + 5 rows of 20.0 + 5 rows of 30.0 = 300.0
  EXPECT_DOUBLE_EQ(single_row_value<double>(*result), 300.0);
  EXPECT_FALSE(op.next(context).has_value());
  op.close(context);
}

TEST(ScalarAggregateOperator, CountStarCountsRowsAcrossBatchesIncludingZero) {
  RmmEnvironment env(default_config());
  auto count_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::CountStar, nullptr, int64_type(false));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_expr, "n"}};

  {
    // Non-empty case.
    Schema schema({Field{"x", int32_type(false)}});
    std::vector<DeviceBatch> batches;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(filled_column(cudf::type_id::INT32, 1, 7));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
    std::vector<std::unique_ptr<cudf::column>> columns2;
    columns2.push_back(filled_column(cudf::type_id::INT32, 1, 3));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns2)),
                                  std::make_shared<const Schema>(schema)));

    ScalarAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                               std::vector<NamedExpression>{aggregates});
    ExecutionContext context = make_context();
    op.open(context);
    std::optional<DeviceBatch> result = op.next(context);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(single_row_value<std::int64_t>(*result), 10);
    op.close(context);
  }
  {
    // Empty case: COUNT(*) of zero rows is 0, not NULL.
    ScalarAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}),
                               std::vector<NamedExpression>{aggregates});
    ExecutionContext context = make_context();
    op.open(context);
    std::optional<DeviceBatch> result = op.next(context);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(single_row_is_null(*result));
    EXPECT_EQ(single_row_value<std::int64_t>(*result), 0);
    op.close(context);
  }
}

TEST(ScalarAggregateOperator, SumOfEmptyInputIsNullNotZero) {
  RmmEnvironment env(default_config());
  auto amount = std::make_shared<ColumnExpression>("amount", 0, float64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"}};

  ScalarAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}),
                             std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(single_row_is_null(*result));
  op.close(context);
}

TEST(ScalarAggregateOperator, AvgComputesMeanAcrossBatches) {
  RmmEnvironment env(default_config());
  Schema schema({Field{"amount", float64_type(false)}});
  std::vector<DeviceBatch> batches;
  for (double value : {2.0, 4.0, 6.0}) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(filled_column(cudf::type_id::FLOAT64, value, 2));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }

  auto amount = std::make_shared<ColumnExpression>("amount", 0, float64_type(false));
  auto avg_expr = std::make_shared<AggregateExpression>(AggregateFunction::Avg, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{avg_expr, "avg"}};

  ScalarAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                             std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  // (2+2+4+4+6+6)/6 = 4.0
  EXPECT_DOUBLE_EQ(single_row_value<double>(*result), 4.0);
  op.close(context);
}

TEST(ScalarAggregateOperator, FullTpchQ6ShapedPipeline) {
  RmmEnvironment env(default_config());
  // Columns: l_extendedprice (0), l_discount (1), l_quantity (2).
  Schema schema({Field{"l_extendedprice", float64_type(false)}, Field{"l_discount", float64_type(false)},
                 Field{"l_quantity", int32_type(false)}});
  std::vector<DeviceBatch> batches;
  // Batch 1: 4 rows that pass (discount 0.06, quantity 10), price 100.
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(filled_column(cudf::type_id::FLOAT64, 100.0, 4));
    columns.push_back(filled_column(cudf::type_id::FLOAT64, 0.06, 4));
    columns.push_back(filled_column(cudf::type_id::INT32, 10, 4));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }
  // Batch 2: 3 rows that fail quantity < 24 (quantity = 30), should be excluded.
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(filled_column(cudf::type_id::FLOAT64, 200.0, 3));
    columns.push_back(filled_column(cudf::type_id::FLOAT64, 0.06, 3));
    columns.push_back(filled_column(cudf::type_id::INT32, 30, 3));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }
  // Batch 3: 2 rows that pass (discount 0.05, quantity 5), price 50.
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(filled_column(cudf::type_id::FLOAT64, 50.0, 2));
    columns.push_back(filled_column(cudf::type_id::FLOAT64, 0.05, 2));
    columns.push_back(filled_column(cudf::type_id::INT32, 5, 2));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }

  auto discount = std::make_shared<ColumnExpression>("l_discount", 1, float64_type(false));
  auto lower = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(0.05));
  auto upper = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(0.07));
  auto between = std::make_shared<BetweenExpression>(discount, lower, upper);

  auto quantity = std::make_shared<ColumnExpression>("l_quantity", 2, int32_type(false));
  auto quantity_i64 = std::make_shared<CastExpression>(quantity, int64_type(false));
  auto twenty_four = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(24));
  auto quantity_cmp = std::make_shared<BinaryExpression>(BinaryOperator::Less, quantity_i64, twenty_four,
                                                         boolean_type(false));
  auto predicate =
      std::make_shared<BinaryExpression>(BinaryOperator::And, between, quantity_cmp, boolean_type(false));

  auto extendedprice = std::make_shared<ColumnExpression>("l_extendedprice", 0, float64_type(false));
  auto revenue_expr = std::make_shared<BinaryExpression>(BinaryOperator::Multiply, extendedprice, discount,
                                                         float64_type(false));
  std::vector<NamedExpression> projection_items = {NamedExpression{revenue_expr, "revenue"}};

  auto revenue_col = std::make_shared<ColumnExpression>("revenue", 0, float64_type(false));
  auto sum_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::Sum, revenue_col, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total_revenue"}};

  auto source = std::make_unique<VectorSourceOperator>(std::move(batches));
  auto filter = std::make_unique<FilterOperator>(1, std::move(source), predicate);
  auto projection = std::make_unique<ProjectionOperator>(2, std::move(filter), std::move(projection_items));
  ScalarAggregateOperator aggregate(3, std::move(projection), std::move(aggregates));

  ExecutionContext context = make_context();
  aggregate.open(context);
  std::optional<DeviceBatch> result = aggregate.next(context);
  ASSERT_TRUE(result.has_value());
  // Expected: batch 1 (4 rows * 100*0.06=6.0) + batch 3 (2 rows * 50*0.05=2.5)
  //         = 24.0 + 5.0 = 29.0. Batch 2 is excluded by quantity < 24.
  EXPECT_DOUBLE_EQ(single_row_value<double>(*result), 29.0);
  EXPECT_FALSE(aggregate.next(context).has_value());
  aggregate.close(context);
}

}  // namespace
}  // namespace kernellake
