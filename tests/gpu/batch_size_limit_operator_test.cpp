#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

#include <vector>

#include "kernellake/execution_gpu/batch_size_limit_operator.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

class VectorSourceOperator final : public PhysicalOperator {
 public:
  explicit VectorSourceOperator(std::vector<DeviceBatch> batches) : batches_(std::move(batches)) {}
  void open(ExecutionContext&) override { index_ = 0; }
  std::optional<DeviceBatch> next(ExecutionContext&) override {
    if (index_ >= batches_.size()) {
      return std::nullopt;
    }
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

// A distinguishable [start, start+1, ..., start+num_rows-1] column, not a
// constant fill -- lets tests confirm row order/content survives a split,
// not just row counts.
DeviceBatch make_sequence_batch(int32_t start, cudf::size_type num_rows) {
  const std::unique_ptr<cudf::scalar> init = cudf::make_fixed_width_scalar<int32_t>(start);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::sequence(num_rows, *init));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                     std::make_shared<const Schema>(one_int_column_schema()));
}

std::vector<int32_t> read_column(const DeviceBatch& batch) {
  const cudf::column_view column = batch.view().column(0);
  std::vector<int32_t> values(static_cast<std::size_t>(column.size()));
  cudaMemcpy(values.data(), column.data<int32_t>(), values.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
  return values;
}

TEST(BatchSizeLimitOperator, PassesBatchWithinCapThroughUnchanged) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_sequence_batch(0, 5));

  BatchSizeLimitOperator limiter(1, std::make_unique<VectorSourceOperator>(std::move(batches)), 10);
  ExecutionContext context = make_context();
  limiter.open(context);

  auto batch = limiter.next(context);
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->row_count(), 5u);
  EXPECT_EQ(read_column(*batch), (std::vector<int32_t>{0, 1, 2, 3, 4}));
  EXPECT_FALSE(limiter.next(context).has_value());
  limiter.close(context);
}

TEST(BatchSizeLimitOperator, SplitsOversizedBatchIntoCappedChunksPreservingOrder) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_sequence_batch(0, 10));  // [0..9], max_rows=4 -> chunks of 4,4,2

  BatchSizeLimitOperator limiter(1, std::make_unique<VectorSourceOperator>(std::move(batches)), 4);
  ExecutionContext context = make_context();
  limiter.open(context);

  auto first = limiter.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 4u);
  EXPECT_EQ(read_column(*first), (std::vector<int32_t>{0, 1, 2, 3}));

  auto second = limiter.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 4u);
  EXPECT_EQ(read_column(*second), (std::vector<int32_t>{4, 5, 6, 7}));

  auto third = limiter.next(context);
  ASSERT_TRUE(third.has_value());
  EXPECT_EQ(third->row_count(), 2u);
  EXPECT_EQ(read_column(*third), (std::vector<int32_t>{8, 9}));

  EXPECT_FALSE(limiter.next(context).has_value());
  limiter.close(context);
}

TEST(BatchSizeLimitOperator, ExactMultipleOfCapProducesNoTrailingEmptyChunk) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_sequence_batch(0, 8));  // exactly 2x max_rows

  BatchSizeLimitOperator limiter(1, std::make_unique<VectorSourceOperator>(std::move(batches)), 4);
  ExecutionContext context = make_context();
  limiter.open(context);

  ASSERT_TRUE(limiter.next(context).has_value());
  ASSERT_TRUE(limiter.next(context).has_value());
  EXPECT_FALSE(limiter.next(context).has_value());
  limiter.close(context);
}

TEST(BatchSizeLimitOperator, OnlyOversizedBatchesAreSplitAmongMultipleFromChild) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_sequence_batch(0, 3));    // under cap: passes through
  batches.push_back(make_sequence_batch(100, 7));  // over cap: split into 5 + 2

  BatchSizeLimitOperator limiter(1, std::make_unique<VectorSourceOperator>(std::move(batches)), 5);
  ExecutionContext context = make_context();
  limiter.open(context);

  auto first = limiter.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(read_column(*first), (std::vector<int32_t>{0, 1, 2}));

  auto second = limiter.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(read_column(*second), (std::vector<int32_t>{100, 101, 102, 103, 104}));

  auto third = limiter.next(context);
  ASSERT_TRUE(third.has_value());
  EXPECT_EQ(read_column(*third), (std::vector<int32_t>{105, 106}));

  EXPECT_FALSE(limiter.next(context).has_value());
  limiter.close(context);
}

TEST(BatchSizeLimitOperator, EmptyChildProducesNoBatches) {
  RmmEnvironment env(default_config());
  BatchSizeLimitOperator limiter(1, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}), 5);
  ExecutionContext context = make_context();
  limiter.open(context);
  EXPECT_FALSE(limiter.next(context).has_value());
  limiter.close(context);
}

TEST(BatchSizeLimitOperator, NameAndIdReportItsOwnValues) {
  RmmEnvironment env(default_config());
  BatchSizeLimitOperator limiter(42, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}), 5);
  EXPECT_EQ(limiter.name(), "BatchSizeLimit");
  EXPECT_EQ(limiter.id(), 42u);
}

}  // namespace
}  // namespace kernellake
