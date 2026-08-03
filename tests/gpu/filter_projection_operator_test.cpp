#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include "kernellake/execution_gpu/filter_operator.hpp"
#include "kernellake/execution_gpu/projection_operator.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

// Test double: yields a fixed, pre-built sequence of batches, one per call
// to next(), then reports exhausted.
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

Schema one_int_column_schema() {
  return Schema({Field{"a", int32_type(false)}});
}

DeviceBatch make_filled_batch(int32_t fill_value, cudf::size_type num_rows) {
  std::unique_ptr<cudf::column> column =
      cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  std::unique_ptr<cudf::scalar> value = cudf::make_fixed_width_scalar<int32_t>(fill_value);
  cudf::mutable_column_view view = column->mutable_view();
  cudf::fill_in_place(view, 0, num_rows, *value);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                     std::make_shared<const Schema>(one_int_column_schema()));
}

ExecutionContext make_context() {
  return ExecutionContext{"test-query", 0,       nullptr, rmm::mr::get_current_device_resource_ref(),
                          nullptr,      nullptr, nullptr};
}

TEST(FilterOperator, PassesMatchingBatchesAndSkipsEmptyOnes) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(10, 5));  // all rows pass a > 3
  batches.push_back(make_filled_batch(1, 5));   // all rows fail a > 3 -> skipped entirely
  batches.push_back(make_filled_batch(20, 3));  // all rows pass

  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto three = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(3));
  auto a_i64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto predicate =
      std::make_shared<BinaryExpression>(BinaryOperator::Greater, a_i64, three, boolean_type(false));

  FilterOperator filter(1, std::make_unique<VectorSourceOperator>(std::move(batches)), predicate);
  ExecutionContext context = make_context();
  filter.open(context);

  std::optional<DeviceBatch> first = filter.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 5u);

  std::optional<DeviceBatch> second = filter.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 3u);  // the all-failing batch was skipped

  EXPECT_FALSE(filter.next(context).has_value());
  filter.close(context);
}

TEST(ProjectionOperator, EvaluatesArithmeticAcrossMultipleBatches) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(5, 4));
  batches.push_back(make_filled_batch(7, 2));

  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto ten = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(10));
  auto a_i64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto doubled = std::make_shared<BinaryExpression>(BinaryOperator::Add, a_i64, ten, int64_type(false));

  std::vector<NamedExpression> items = {NamedExpression{doubled, "result"}};
  ProjectionOperator projection(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                                std::move(items));
  ExecutionContext context = make_context();
  projection.open(context);

  std::optional<DeviceBatch> first = projection.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 4u);
  EXPECT_EQ(first->schema().field(0).name, "result");

  std::optional<DeviceBatch> second = projection.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 2u);

  EXPECT_FALSE(projection.next(context).has_value());
  projection.close(context);
}

}  // namespace
}  // namespace kernellake
