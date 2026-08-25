#include <gtest/gtest.h>

#include <arrow/api.h>

#include <map>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/arrow_bridge.hpp"
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

// Regression coverage for COUNT(DISTINCT ...) (added for TPC-H Q16): the
// GPU backend can't merge a per-batch "distinct count" across batches via
// the same SUM-of-partials trick every other aggregate here uses (see
// HashAggregateOperator's own CountDistinct comment) -- deliberately
// repeats supplier 200 for region 1 and supplier 300 for region 2 *across*
// two separate batches (not just within one), so a naive
// sum-of-per-batch-distinct-counts implementation would double-count them
// (region 1 -> 3, region 2 -> 3) instead of the true cross-batch distinct
// count (region 1 -> 2 {100, 200}, region 2 -> 2 {300, 400}).
TEST(HashAggregateOperator, CountDistinctMergesAcrossBatchesWithoutDoubleCountingRepeatedValues) {
  RmmEnvironment env(default_config());

  Schema region_supplier_schema({Field{"region", int32_type(false)}, Field{"supplier", int32_type(false)}});
  auto make_region_supplier_batch = [&](const std::vector<int32_t>& regions,
                                        const std::vector<int32_t>& suppliers) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_from_host(regions, cudf::type_id::INT32));
    columns.push_back(column_from_host(suppliers, cudf::type_id::INT32));
    return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                       std::make_shared<const Schema>(region_supplier_schema));
  };

  std::vector<DeviceBatch> batches;
  batches.push_back(make_region_supplier_batch({1, 1, 1, 2}, {100, 100, 200, 300}));
  batches.push_back(make_region_supplier_batch({1, 2, 2}, {200, 300, 400}));
  batches.push_back(make_region_supplier_batch({3}, {500}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};

  auto supplier = std::make_shared<ColumnExpression>("supplier", 1, int32_type(false));
  auto count_distinct_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::CountDistinct, supplier, int64_type(false));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_distinct_expr, "supplier_cnt"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);

  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 3u);
  EXPECT_FALSE(op.next(context).has_value());

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<std::int64_t> count_values = copy_to_host<std::int64_t>(result->view().column(1));
  std::map<int32_t, std::int64_t> by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) {
    by_region[region_values[i]] = count_values[i];
  }

  ASSERT_EQ(by_region.size(), 3u);
  EXPECT_EQ(by_region[1], 2);  // {100, 200}
  EXPECT_EQ(by_region[2], 2);  // {300, 400}
  EXPECT_EQ(by_region[3], 1);  // {500}

  op.close(context);
}

// The GPU backend's own restriction (see HashAggregateOperator's class
// comment): unlike the CPU/Acero backend, COUNT(DISTINCT ...) can't share
// a GROUP BY with another aggregate here.
TEST(HashAggregateOperator, OpenThrowsWhenCountDistinctIsCombinedWithAnotherAggregate) {
  RmmEnvironment env(default_config());

  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 2}, {10.0, 20.0}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};

  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto count_distinct_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::CountDistinct, amount, int64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_distinct_expr, "distinct_cnt"},
                                             NamedExpression{sum_expr, "total"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  EXPECT_THROW(op.open(context), ExecutionError);
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

// Regression coverage for flush_pending()'s size-adaptive batching (see its
// own comment in hash_aggregate_operator.cpp): accumulated_ is no longer
// folded on every single batch, only when pending_partials_ has caught up
// in row count -- this exercises many batches with heavily repeated keys,
// so correctness has to hold both across mid-loop flushes (whenever the
// threshold is crossed) and the mandatory final flush in next() (for
// whatever's still pending when child_ is exhausted), without knowing or
// depending on exactly when either happens.
TEST(HashAggregateOperator, AccumulatesCorrectlyAcrossManySmallBatchesWithRepeatedKeys) {
  RmmEnvironment env(default_config());

  constexpr int kBatchCount = 23;  // deliberately not a multiple of kDistinctKeys
  constexpr int kDistinctKeys = 5;
  std::vector<DeviceBatch> batches;
  batches.reserve(kBatchCount);
  for (int i = 0; i < kBatchCount; ++i) {
    batches.push_back(make_batch({i % kDistinctKeys}, {1.0}));
  }

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
  EXPECT_EQ(result->row_count(), static_cast<std::size_t>(kDistinctKeys));
  EXPECT_FALSE(op.next(context).has_value());

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<double> total_values = copy_to_host<double>(result->view().column(1));
  const std::vector<std::int64_t> count_values = copy_to_host<std::int64_t>(result->view().column(2));

  std::map<int32_t, std::pair<double, std::int64_t>> by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) {
    by_region[region_values[i]] = {total_values[i], count_values[i]};
  }
  ASSERT_EQ(by_region.size(), static_cast<std::size_t>(kDistinctKeys));
  for (int key = 0; key < kDistinctKeys; ++key) {
    // key appears once every kDistinctKeys batches out of kBatchCount total.
    const std::int64_t expected_count = (kBatchCount - key + kDistinctKeys - 1) / kDistinctKeys;
    EXPECT_DOUBLE_EQ(by_region[key].first, static_cast<double>(expected_count));
    EXPECT_EQ(by_region[key].second, expected_count);
  }

  op.close(context);
}

// Regression coverage: max_distinct_keys is now only checked when
// flush_pending() actually runs, not after every single batch (see its own
// comment). The final flush in next() always runs once child_ is
// exhausted regardless of whether any mid-loop flush ever triggered, so a
// cardinality cap violation spread across many small batches -- none of
// which individually would have crossed the threshold that triggers an
// early flush -- must still be caught by the time next() returns, not
// silently allowed through.
TEST(HashAggregateOperator, ExceedingMaxDistinctKeysAcrossManyDeferredBatchesStillThrows) {
  RmmEnvironment env(default_config());

  // 10 batches, each introducing one brand-new distinct key (never
  // repeated) -- accumulated_ starts at 1 row after the first (immediate)
  // flush, so every batch after that only ever contributes 1 pending row at
  // a time, keeping pending_rows_ well under accumulated_'s own row count
  // and deferring the fold for many batches in a row.
  constexpr int kBatchCount = 10;
  std::vector<DeviceBatch> batches;
  batches.reserve(kBatchCount);
  for (int i = 0; i < kBatchCount; ++i) {
    batches.push_back(make_batch({i}, {1.0}));
  }

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"}};

  // Well under kBatchCount's 10 real distinct keys, so this must throw by
  // the time all input is consumed -- whether or not any mid-loop flush
  // happened to trigger before then.
  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates), /*max_distinct_keys=*/5);
  ExecutionContext context = make_context();
  op.open(context);
  EXPECT_THROW(op.next(context), ExecutionError);
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

// Constructor's own validation had no coverage: GROUP BY with zero keys is
// a real planner bug (a scalar aggregate with no GROUP BY clause at all
// should go through ScalarAggregateOperator instead, never this class),
// not a user-facing SQL error.
TEST(HashAggregateOperator, ConstructorThrowsWhenGroupByIsEmpty) {
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto sum_expr = std::make_shared<AggregateExpression>(AggregateFunction::Sum, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"}};

  EXPECT_THROW(
      {
        HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}),
                                 std::vector<NamedExpression>{}, std::move(aggregates));
      },
      PlanningError);
}

// Same rationale as HashAggregateOperator's own open()-time check --
// mirrors ScalarAggregateOperator.OpenThrowsWhenAggregateItemIsNotAnAggregateExpression.
TEST(HashAggregateOperator, OpenThrowsWhenAggregateItemIsNotAnAggregateExpression) {
  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  std::vector<NamedExpression> aggregates = {NamedExpression{amount, "amount"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}),
                           std::move(group_by), std::move(aggregates));
  ExecutionContext context = make_context();
  EXPECT_THROW({ op.open(context); }, ExecutionError);
}

// AggregateFunction::Count (as opposed to CountStar, which every other
// COUNT-shaped test in this file uses) had no coverage of its own --
// open()'s dedicated Count case (ValueColumnKind::CountColumnOnes) is a
// separate branch from CountStar's.
TEST(HashAggregateOperator, GroupedCountOfColumnMatchesRowCountPerGroup) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 1, 1, 2, 2}, {10.0, 20.0, 5.0, 100.0, 7.0}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto count_expr = std::make_shared<AggregateExpression>(AggregateFunction::Count, amount, int64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_expr, "n"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 2u);

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<std::int64_t> counts = copy_to_host<std::int64_t>(result->view().column(1));
  std::map<int32_t, std::int64_t> counts_by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) counts_by_region[region_values[i]] = counts[i];
  EXPECT_EQ(counts_by_region[1], 3);
  EXPECT_EQ(counts_by_region[2], 2);
  op.close(context);
}

// next()'s AggregateOutputKind::Average finalization (the divide-two-
// SUM-of-ones-derived-columns path) had no coverage -- open()'s own AVG
// setup is exercised elsewhere, but no existing test actually calls
// next() with real data behind a grouped AVG.
TEST(HashAggregateOperator, GroupedAvgComputesCorrectMeans) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 1, 2, 2, 2}, {10.0, 20.0, 3.0, 6.0, 9.0}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto avg_expr = std::make_shared<AggregateExpression>(AggregateFunction::Avg, amount, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{avg_expr, "avg"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 2u);

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<double> avgs = copy_to_host<double>(result->view().column(1));
  std::map<int32_t, double> avg_by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) avg_by_region[region_values[i]] = avgs[i];
  EXPECT_DOUBLE_EQ(avg_by_region[1], 15.0);  // (10+20)/2
  EXPECT_DOUBLE_EQ(avg_by_region[2], 6.0);   // (3+6+9)/3
  op.close(context);
}

// compile_expr()/materialize_case(): a CASE with no ELSE as a GROUP BY
// value expression had no coverage -- mirrors
// ScalarAggregateOperator.SumOverCaseWithNoElseTreatsUnmatchedRowsAsNull.
TEST(HashAggregateOperator, GroupedSumOverCaseWithNoElseTreatsUnmatchedRowsAsNull) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 1}, {10.0, 20.0}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};

  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto thousand = std::make_shared<LiteralExpression>(LiteralExpression::make_float64(1000.0));
  auto condition =
      std::make_shared<BinaryExpression>(BinaryOperator::Greater, amount, thousand, boolean_type(false));
  CaseExpression::WhenThen branch{condition, amount};
  auto case_expr = std::make_shared<CaseExpression>(std::vector<CaseExpression::WhenThen>{branch}, nullptr,
                                                    float64_type(true));
  auto sum_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::Sum, case_expr, float64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{sum_expr, "total"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 1u);
  EXPECT_TRUE(result->view().column(1).null_count() > 0);
  op.close(context);
}

// compile_expr()/materialize_extract(): EXTRACT(... FROM ...) as an
// aggregate's own argument had no coverage.
TEST(HashAggregateOperator, GroupedMaxOverExtractedYearMatchesExpectedValue) {
  RmmEnvironment env(default_config());
  Schema schema({Field{"region", int32_type(false)}, Field{"event_date", date32_type(false)}});
  std::vector<DeviceBatch> batches;
  {
    std::vector<std::int32_t> regions = {1, 1};
    auto region_column = column_from_host(regions, cudf::type_id::INT32);
    // 2024-01-01, 2025-06-15 as days-since-epoch.
    std::vector<std::int32_t> days = {19723, 20254};
    auto date_column = column_from_host(days, cudf::type_id::TIMESTAMP_DAYS);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(region_column));
    columns.push_back(std::move(date_column));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto event_date = std::make_shared<ColumnExpression>("event_date", 1, date32_type(false));
  auto extract_year = std::make_shared<ExtractExpression>(DatePart::Year, event_date, int64_type(false));
  auto max_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::Max, extract_year, int64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{max_expr, "latest_year"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 1u);
  EXPECT_EQ(copy_to_host<std::int64_t>(result->view().column(1))[0], 2025);
  op.close(context);
}

// compile_expr()/materialize()'s CastExpression-to-Decimal branch had no
// coverage as a GROUP BY value expression.
TEST(HashAggregateOperator, GroupedCountOverDecimalCastExpressionCountsEveryRow) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_batch({1, 1, 1}, {12.34, 5.0, 9.9}));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(false));
  auto decimal_cast =
      std::make_shared<CastExpression>(amount, decimal_type(/*precision=*/10, /*scale=*/2, false));
  auto count_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::Count, decimal_cast, int64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_expr, "n"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 1u);
  EXPECT_EQ(copy_to_host<std::int64_t>(result->view().column(1))[0], 3);
  op.close(context);
}

// compile_expr()/materialize_like(): the negated (NOT LIKE) branch had no
// coverage as a GROUP BY value expression -- mirrors
// ScalarAggregateOperator.CountOfNotLikeExpressionCountsEveryNonNullRow.
TEST(HashAggregateOperator, GroupedCountOfNotLikeExpressionCountsEveryNonNullRow) {
  RmmEnvironment env(default_config());
  const std::shared_ptr<const Schema> schema = std::make_shared<const Schema>(
      Schema({Field{"region", int32_type(false)}, Field{"name", string_type(false)}}));

  arrow::Int32Builder region_builder;
  arrow::StringBuilder name_builder;
  for (const auto& [region, name] :
       std::vector<std::pair<int32_t, std::string>>{{1, "apple"}, {1, "banana"}, {2, "avocado"}}) {
    ASSERT_TRUE(region_builder.Append(region).ok());
    ASSERT_TRUE(name_builder.Append(name).ok());
  }
  std::shared_ptr<arrow::Array> region_array, name_array;
  ASSERT_TRUE(region_builder.Finish(&region_array).ok());
  ASSERT_TRUE(name_builder.Finish(&name_array).ok());
  const auto arrow_schema = arrow::schema(
      {arrow::field("region", arrow::int32(), false), arrow::field("name", arrow::utf8(), false)});
  const auto arrow_batch = arrow::RecordBatch::Make(arrow_schema, 3, {region_array, name_array});

  std::vector<DeviceBatch> batches;
  batches.push_back(from_arrow_record_batch(*arrow_batch, schema));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto name = std::make_shared<ColumnExpression>("name", 1, string_type(false));
  auto not_like = std::make_shared<LikeExpression>(name, "a%", /*negated=*/true);
  auto count_expr =
      std::make_shared<AggregateExpression>(AggregateFunction::Count, not_like, int64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_expr, "n"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 2u);

  const std::vector<int32_t> region_values = copy_to_host<int32_t>(result->view().column(0));
  const std::vector<std::int64_t> counts = copy_to_host<std::int64_t>(result->view().column(1));
  std::map<int32_t, std::int64_t> counts_by_region;
  for (std::size_t i = 0; i < region_values.size(); ++i) counts_by_region[region_values[i]] = counts[i];
  EXPECT_EQ(counts_by_region[1], 2);
  EXPECT_EQ(counts_by_region[2], 1);
  op.close(context);
}

// materialize_count_ones()'s nullable-argument null-mask-copy branch had
// no coverage -- GroupedCountOfColumnMatchesRowCountPerGroup above uses a
// non-nullable argument column (nullable() is false, so this branch is
// never entered), unlike ScalarAggregateOperator's own COUNT tests, which
// already exercise a real nullable argument via CountOfColumnExcludesNulls.
TEST(HashAggregateOperator, GroupedCountOfNullableColumnExcludesNulls) {
  RmmEnvironment env(default_config());
  const std::shared_ptr<const Schema> schema = std::make_shared<const Schema>(
      Schema({Field{"region", int32_type(false)}, Field{"amount", float64_type(true)}}));

  arrow::Int32Builder region_builder;
  arrow::DoubleBuilder amount_builder;
  ASSERT_TRUE(region_builder.Append(1).ok());
  ASSERT_TRUE(amount_builder.Append(10.0).ok());
  ASSERT_TRUE(region_builder.Append(1).ok());
  ASSERT_TRUE(amount_builder.AppendNull().ok());
  ASSERT_TRUE(region_builder.Append(1).ok());
  ASSERT_TRUE(amount_builder.Append(20.0).ok());
  std::shared_ptr<arrow::Array> region_array, amount_array;
  ASSERT_TRUE(region_builder.Finish(&region_array).ok());
  ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
  const auto arrow_schema = arrow::schema(
      {arrow::field("region", arrow::int32(), false), arrow::field("amount", arrow::float64(), true)});
  const auto arrow_batch = arrow::RecordBatch::Make(arrow_schema, 3, {region_array, amount_array});

  std::vector<DeviceBatch> batches;
  batches.push_back(from_arrow_record_batch(*arrow_batch, schema));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(true));
  auto count_expr = std::make_shared<AggregateExpression>(AggregateFunction::Count, amount, int64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_expr, "n"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->row_count(), 1u);
  EXPECT_EQ(copy_to_host<std::int64_t>(result->view().column(1))[0], 2);  // excludes the 1 NULL row
  op.close(context);
}

// Regression test: COUNT(argument) is physically SUM(argument's own null
// mask copied onto an all-1s column) -- see
// HashAggregateOperator::is_count_result_'s own comment for why. A group
// whose argument is NULL in *every* one of its rows (region 2 here, a
// single all-NULL row -- reachable for real via a LEFT OUTER JOIN's own
// null-extended row, e.g. a customer with zero matching orders) used to
// make that physical SUM come back NULL for the whole group -- correct for
// a genuine SUM, but not for COUNT, whose zero-non-null-values answer is
// 0, never NULL, per SQL. Region 1 (one valid, one NULL row) exercises the
// ordinary non-degenerate case in the same result, so a fix that merely
// forced every COUNT output to 0 unconditionally would still fail this.
TEST(HashAggregateOperator, GroupedCountOfEntirelyNullGroupReturnsZeroNotNull) {
  RmmEnvironment env(default_config());
  const std::shared_ptr<const Schema> schema = std::make_shared<const Schema>(
      Schema({Field{"region", int32_type(false)}, Field{"amount", float64_type(true)}}));

  arrow::Int32Builder region_builder;
  arrow::DoubleBuilder amount_builder;
  ASSERT_TRUE(region_builder.Append(1).ok());
  ASSERT_TRUE(amount_builder.Append(10.0).ok());
  ASSERT_TRUE(region_builder.Append(1).ok());
  ASSERT_TRUE(amount_builder.AppendNull().ok());
  ASSERT_TRUE(region_builder.Append(2).ok());
  ASSERT_TRUE(amount_builder.AppendNull().ok());
  std::shared_ptr<arrow::Array> region_array, amount_array;
  ASSERT_TRUE(region_builder.Finish(&region_array).ok());
  ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
  const auto arrow_schema = arrow::schema(
      {arrow::field("region", arrow::int32(), false), arrow::field("amount", arrow::float64(), true)});
  const auto arrow_batch = arrow::RecordBatch::Make(arrow_schema, 3, {region_array, amount_array});

  std::vector<DeviceBatch> batches;
  batches.push_back(from_arrow_record_batch(*arrow_batch, schema));

  auto region = std::make_shared<ColumnExpression>("region", 0, int32_type(false));
  std::vector<NamedExpression> group_by = {NamedExpression{region, "region"}};
  auto amount = std::make_shared<ColumnExpression>("amount", 1, float64_type(true));
  auto count_expr = std::make_shared<AggregateExpression>(AggregateFunction::Count, amount, int64_type(true));
  std::vector<NamedExpression> aggregates = {NamedExpression{count_expr, "n"}};

  HashAggregateOperator op(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(group_by),
                           std::move(aggregates));
  ExecutionContext context = make_context();
  op.open(context);
  std::optional<DeviceBatch> result = op.next(context);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->row_count(), 2u);

  const std::vector<std::int32_t> regions = copy_to_host<std::int32_t>(result->view().column(0));
  const cudf::column_view count_view = result->view().column(1);
  ASSERT_EQ(count_view.null_count(), 0);  // never NULL, regardless of region.
  const std::vector<std::int64_t> counts = copy_to_host<std::int64_t>(count_view);
  std::map<std::int32_t, std::int64_t> count_by_region;
  for (std::size_t i = 0; i < regions.size(); ++i) count_by_region[regions[i]] = counts[i];
  EXPECT_EQ(count_by_region.at(1), 1);  // one valid, one NULL row.
  EXPECT_EQ(count_by_region.at(2), 0);  // entirely NULL -- 0, not NULL.
  op.close(context);
}

}  // namespace
}  // namespace kernellake
