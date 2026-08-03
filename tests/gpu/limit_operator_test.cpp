#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include "kernellake/execution_gpu/limit_operator.hpp"
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

TEST(LimitOperator, PassesBatchesUnchangedUntilLimitReached) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(1, 5));
  batches.push_back(make_filled_batch(2, 5));

  LimitOperator limit(1, std::make_unique<VectorSourceOperator>(std::move(batches)), 10);
  ExecutionContext context = make_context();
  limit.open(context);

  auto first = limit.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 5u);
  auto second = limit.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 5u);
  EXPECT_FALSE(limit.next(context).has_value());
  limit.close(context);
}

TEST(LimitOperator, TruncatesBatchThatWouldExceedLimit) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(1, 5));
  batches.push_back(make_filled_batch(2, 5));

  LimitOperator limit(1, std::make_unique<VectorSourceOperator>(std::move(batches)), 7);
  ExecutionContext context = make_context();
  limit.open(context);

  auto first = limit.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 5u);  // under limit, passed through whole
  auto second = limit.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 2u);  // truncated: only 2 more needed to reach 7
  EXPECT_FALSE(limit.next(context).has_value());
  limit.close(context);
}

TEST(LimitOperator, ZeroLimitProducesNoBatches) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(1, 5));

  LimitOperator limit(1, std::make_unique<VectorSourceOperator>(std::move(batches)), 0);
  ExecutionContext context = make_context();
  limit.open(context);
  EXPECT_FALSE(limit.next(context).has_value());
  limit.close(context);
}

}  // namespace
}  // namespace kernellake
