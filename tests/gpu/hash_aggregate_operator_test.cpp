#include <gtest/gtest.h>

#include <map>

#include "kernellake/execution_gpu/hash_aggregate_operator.hpp"
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

template <typename T>
std::unique_ptr<cudf::column> column_from_host(const std::vector<T>& values, cudf::type_id type) {
  rmm::device_buffer data(values.size() * sizeof(T), cudf::get_default_stream());
  cudaMemcpy(data.data(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
  return std::make_unique<cudf::column>(cudf::data_type{type}, static_cast<cudf::size_type>(values.size()),
                                        std::move(data), rmm::device_buffer{}, 0);
}

template <typename T>
std::vector<T> copy_to_host(const cudf::column_view& view) {
  std::vector<T> host(static_cast<std::size_t>(view.size()));
  cudaMemcpy(host.data(), view.data<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return host;
}

Schema region_amount_schema() {
  return Schema({Field{"region", int32_type(false)}, Field{"amount", float64_type(false)}});
}

DeviceBatch make_batch(const std::vector<int32_t>& regions, const std::vector<double>& amounts) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_from_host(regions, cudf::type_id::INT32));
  columns.push_back(column_from_host(amounts, cudf::type_id::FLOAT64));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                     std::make_shared<const Schema>(region_amount_schema()));
}

TEST(HashAggregateOperator, MergesPartialGroupsAcrossBatches) {
  RmmEnvironment env(default_config());

  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 1, 2, 2}, {10.0, 10.0, 20.0, 20.0}));
  batches.push_back(make_batch({1, 2, 3}, {5.0, 5.0, 5.0}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};

  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  auto count_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::CountStar, nullptr, int64_type(false));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"},
                                             NamedExpression{count_expr, "n"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);

  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 3u);  // three distinct regions
  EXPECT_FALSE(op.next(context).has_value());

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<double> total_values = copy_to_host<double>(result->view().column(1));
  const std::vector<std::int64_t> count_values = copy_to_host<std::int64_t>(result->view().column(2));

  std::map<int32_t, std::pair<double, std::int64_t>> by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) {
    by_region[region_values[i]] = {total_values[i], count_values[i]};
  }

  ASSERT_EQ(by_region.size(), 3u);
  EXPECT_DOUBLE_EQ(by_region[1].first, 25.0);  // 10+10+5
  EXPECT_EQ(by_region[1].second, 3);
  EXPECT_DOUBLE_EQ(by_region[2].first, 45.0);  // 20+20+5
  EXPECT_EQ(by_region[2].second, 3);
  EXPECT_DOUBLE_EQ(by_region[3].first, 5.0);
  EXPECT_EQ(by_region[3].second, 1);

  op.close(context);
}

// Regression test: cudf::groupby::streaming_groupby::aggregate() requires a
// single call's row count to not exceed max_distinct_keys (an
// implementation encoding-scheme constraint, unrelated to actual GROUP BY
// cardinality -- see cudf/groupby.hpp's own streaming_groupby doc comment;
// max_distinct_keys separately also caps *cumulative* distinct keys across
// the object's lifetime, confirmed by a real "Distinct key count (3) would
// exceed max_distinct_keys (2)" failure from an earlier, wrongly-designed
// version of this test that used 3 distinct keys against a cap of 2 --
// fixed here by using only 2 distinct keys, well under the cap, while still
// exceeding it in raw row count). ParquetScanOperator's own pass splitting
// is purely memory-based (pass_read_limit_bytes), so a single incoming
// batch can legitimately exceed max_distinct_keys in row count at real
// scale (confirmed by a real SF10 TPC-H Q1 run: a 59.6M-row single-pass
// batch, only ~6 actual distinct keys, against the default 10M
// max_distinct_keys). Exercises the same code path with a deliberately
// tiny max_distinct_keys (2) against a single 5-row batch spanning only 2
// distinct keys, forcing HashAggregateOperator::process_batch() to slice
// it into multiple aggregate() calls within one process_batch() -- not
// across separate next()-returned batches, which
// MergesPartialGroupsAcrossBatches above already covers.
TEST(HashAggregateOperator, SplitsSingleBatchExceedingMaxDistinctKeys) {
  RmmEnvironment env(default_config());

  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 1, 1, 2, 2}, {10.0, 10.0, 10.0, 20.0, 20.0}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};

  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  auto count_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::CountStar, nullptr, int64_type(false));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"},
                                             NamedExpression{count_expr, "n"}};

  // max_distinct_keys=2, deliberately smaller than this single 5-row batch
  // (but not smaller than its 2 actual distinct keys) -- forces at least 3
  // aggregate() calls (slices of at most 2 rows each) within the one
  // process_batch() call below.
  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates), /*max_distinct_keys=*/2);
  ExecutionContext context = make_context();
  op.open(context);

  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 2u);  // two distinct regions
  EXPECT_FALSE(op.next(context).has_value());

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<double> total_values = copy_to_host<double>(result->view().column(1));
  const std::vector<std::int64_t> count_values = copy_to_host<std::int64_t>(result->view().column(2));

  std::map<int32_t, std::pair<double, std::int64_t>> by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) {
    by_region[region_values[i]] = {total_values[i], count_values[i]};
  }

  ASSERT_EQ(by_region.size(), 2u);
  EXPECT_DOUBLE_EQ(by_region[1].first, 30.0);  // 10+10+10
  EXPECT_EQ(by_region[1].second, 3);
  EXPECT_DOUBLE_EQ(by_region[2].first, 40.0);  // 20+20
  EXPECT_EQ(by_region[2].second, 2);

  op.close(context);
}

// Regression coverage: MIN/MAX in a GROUP BY context had zero test
// coverage anywhere before this test (SUM/COUNT dominate every other
// fixture in this file and in the end-to-end GPU suite) -- a materially
// different code path from SUM's accumulate-and-add, not just a
// parameter variation.
TEST(HashAggregateOperator, GroupedMinMaxMatchExpectedValues) {
  RmmEnvironment env(default_config());

  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 1, 1, 2, 2}, {10.0, 20.0, 5.0, 100.0, 7.0}));
  batches.push_back(make_batch({2}, {3.0}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};

  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto min_expr = std::make_shared<AggregateExpression>(AggregateFunction::Min, amount, float64_type(true));
  auto max_expr = std::make_shared<AggregateExpression>(AggregateFunction::Max, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{min_expr, "lo"},
                                             NamedExpression{max_expr, "hi"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);

  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 2u);
  EXPECT_FALSE(op.next(context).has_value());

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<double> lo_values = copy_to_host<double>(result->view().column(1));
  const std::vector<double> hi_values = copy_to_host<double>(result->view().column(2));

  std::map<int32_t, std::pair<double, double>> by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) {
    by_region[region_values[i]] = {lo_values[i], hi_values[i]};
  }

  ASSERT_EQ(by_region.size(), 2u);
  EXPECT_DOUBLE_EQ(by_region[1].first, 5.0);
  EXPECT_DOUBLE_EQ(by_region[1].second, 20.0);
  EXPECT_DOUBLE_EQ(by_region[2].first, 3.0);
  EXPECT_DOUBLE_EQ(by_region[2].second, 100.0);

  op.close(context);
}

TEST(HashAggregateOperator, EmptyInputProducesZeroRowResult) {
  RmmEnvironment env(default_config());
  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}),
                           std::move(group_by), std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 0u);
  op.close(context);
}

}  // namespace
}  // namespace kernellake
