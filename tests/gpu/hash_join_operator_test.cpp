#include <gtest/gtest.h>

#include <algorithm>
#include <map>

#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/execution_gpu/hash_join_operator.hpp"
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

// A real, non-null CudaStream (unlike some other GPU unit tests' plain
// `nullptr` ExecutionContext::stream) -- the partitioned join's spill path
// (hash_join_operator.cpp's spill_partitioned()) specifically relies on
// context.stream being a genuine *blocking* stream distinct from the null
// stream to_arrow_host()/from_arrow() implicitly use (see that function's
// own comment), so these tests need the real production stream shape to
// actually exercise that ordering, not accidentally side-step it via
// same-stream trivial ordering.
ExecutionContext make_context(cudaStream_t stream) {
  return ExecutionContext{"test-query", 0,       stream, rmm::mr::get_current_device_resource_ref(),
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

Schema key_value_schema(const std::string& key_name, const std::string& value_name) {
  return Schema({Field{key_name, int32_type(false)}, Field{value_name, float64_type(false)}});
}

std::shared_ptr<const Schema> left_schema() {
  return std::make_shared<const Schema>(key_value_schema("lkey", "lval"));
}

std::shared_ptr<const Schema> right_schema() {
  return std::make_shared<const Schema>(key_value_schema("rkey", "rval"));
}

std::shared_ptr<const Schema> join_output_schema() {
  return std::make_shared<const Schema>(
      Schema({Field{"lkey", int32_type(false)}, Field{"lval", float64_type(false)},
              Field{"rkey", int32_type(false)}, Field{"rval", float64_type(false)}}));
}

DeviceBatch make_left_batch(const std::vector<int32_t>& keys, const std::vector<double>& values) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_from_host(keys, cudf::type_id::INT32));
  columns.push_back(column_from_host(values, cudf::type_id::FLOAT64));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), left_schema());
}

DeviceBatch make_right_batch(const std::vector<int32_t>& keys, const std::vector<double>& values) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_from_host(keys, cudf::type_id::INT32));
  columns.push_back(column_from_host(values, cudf::type_id::FLOAT64));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)), right_schema());
}

// Drains `op` fully, collecting every (lkey, lval, rkey, rval) row across
// however many next() calls/buckets it takes -- the partitioned path can
// return matches from any bucket in any order, so tests compare against a
// key-indexed map rather than asserting on a specific row order.
std::map<int32_t, std::pair<double, double>> drain_join(HashJoinOperator& op, ExecutionContext& context) {
  std::map<int32_t, std::pair<double, double>> by_key;
  while (std::optional<DeviceBatch> batch = op.next(context)) {
    const std::vector<int32_t> lkeys = copy_to_host<int32_t>(batch->view().column(0));
    const std::vector<double> lvals = copy_to_host<double>(batch->view().column(1));
    const std::vector<double> rvals = copy_to_host<double>(batch->view().column(3));
    for (std::size_t i = 0; i < lkeys.size(); ++i) {
      by_key[lkeys[i]] = {lvals[i], rvals[i]};
    }
  }
  return by_key;
}

// Regression coverage for the grace/partitioned hash join path
// (HashJoinOperator's `partition_count > 1` mode -- see its own doc
// comment and docs/ROADMAP.md's "HashJoinOperator streaming gap" entry
// for the real SF1000 TPC-H OOMs, Q12/Q14/Q19, this was built to fix).
// These tests force partition_count directly via the constructor with
// tiny hand-built fixtures spanning multiple buckets -- correctness of
// the partitioning itself, not a memory-pressure/performance test (that's
// what the real SF1000 benchmarks.local re-run verifies for real, on
// real GPU hardware, separately).

TEST(HashJoinOperator, PartitionedJoinMatchesAcrossMultipleBuckets) {
  RmmEnvironment env(default_config());
  const CudaStream stream;
  ExecutionContext context = make_context(stream.get());

  // Six distinct keys, split across two incoming batches on each side (not
  // one key per batch -- exercises spill_partitioned()'s per-batch
  // hash_partition() call being invoked more than once per side).
  std::vector<DeviceBatch> left_batches;
  left_batches.push_back(make_left_batch({1, 2, 3}, {10.0, 20.0, 30.0}));
  left_batches.push_back(make_left_batch({4, 5, 6}, {40.0, 50.0, 60.0}));
  std::vector<DeviceBatch> right_batches;
  right_batches.push_back(make_right_batch({1, 2, 3}, {100.0, 200.0, 300.0}));
  right_batches.push_back(make_right_batch({4, 5, 6}, {400.0, 500.0, 600.0}));

  // partition_count=4 against 6 distinct keys: by the pigeonhole principle
  // at least some buckets hold multiple keys and (very likely) at least
  // one bucket is empty on one or both sides -- exercising both the
  // multi-key-per-bucket accumulation path and the empty-bucket skip path
  // (next()'s own `if (!probe_batches.empty() && !build_partitions_[...]
  // .empty())` guard) in the same test, without needing to know exactly
  // which bucket MURMUR3 assigns which key to.
  HashJoinOperator op(1, std::make_unique<VectorSourceOperator>(std::move(left_batches)),
                      std::make_unique<VectorSourceOperator>(std::move(right_batches)), /*left_key_index=*/0,
                      /*right_key_index=*/0, join_output_schema(), /*partition_count=*/4);
  op.open(context);
  const std::map<int32_t, std::pair<double, double>> by_key = drain_join(op, context);
  op.close(context);

  ASSERT_EQ(by_key.size(), 6u);
  for (int32_t key = 1; key <= 6; ++key) {
    ASSERT_EQ(by_key.count(key), 1u) << "key " << key;
    EXPECT_DOUBLE_EQ(by_key.at(key).first, key * 10.0);
    EXPECT_DOUBLE_EQ(by_key.at(key).second, key * 100.0);
  }
}

// A key's rows deliberately split across two separate incoming batches on
// *both* sides (not just spread across different keys, as the test above
// already covers) -- proves cudf::hash_partition()'s deterministic
// MURMUR3+fixed-seed hashing really does route the *same* key to the
// *same* bucket index across repeated, separate calls (once per incoming
// batch), which is the entire correctness foundation this design depends
// on. If that ever didn't hold, this key's rows would land in different
// buckets on the two sides and silently produce zero matches for it
// instead of the real join result.
TEST(HashJoinOperator, PartitionedJoinKeySpanningMultipleBatchesOnBothSides) {
  RmmEnvironment env(default_config());
  const CudaStream stream;
  ExecutionContext context = make_context(stream.get());

  std::vector<DeviceBatch> left_batches;
  left_batches.push_back(make_left_batch({7}, {1.0}));
  left_batches.push_back(make_left_batch({7, 9}, {2.0, 90.0}));
  std::vector<DeviceBatch> right_batches;
  right_batches.push_back(make_right_batch({9}, {900.0}));
  right_batches.push_back(make_right_batch({7}, {700.0}));

  HashJoinOperator op(1, std::make_unique<VectorSourceOperator>(std::move(left_batches)),
                      std::make_unique<VectorSourceOperator>(std::move(right_batches)), /*left_key_index=*/0,
                      /*right_key_index=*/0, join_output_schema(), /*partition_count=*/8);
  op.open(context);

  // key 7 appears twice on the left (lval 1.0 and 2.0), once on the right
  // (rval 700.0) -- a real INNER JOIN produces two output rows for it, not
  // one; collect all rows for key 7 explicitly instead of using
  // drain_join()'s last-write-wins map (which would only see one of them).
  std::vector<double> key7_lvals;
  double key9_lval = 0.0;
  double key9_rval = 0.0;
  double key7_rval = 0.0;
  while (std::optional<DeviceBatch> batch = op.next(context)) {
    const std::vector<int32_t> lkeys = copy_to_host<int32_t>(batch->view().column(0));
    const std::vector<double> lvals = copy_to_host<double>(batch->view().column(1));
    const std::vector<double> rvals = copy_to_host<double>(batch->view().column(3));
    for (std::size_t i = 0; i < lkeys.size(); ++i) {
      if (lkeys[i] == 7) {
        key7_lvals.push_back(lvals[i]);
        key7_rval = rvals[i];
      } else {
        ASSERT_EQ(lkeys[i], 9);
        key9_lval = lvals[i];
        key9_rval = rvals[i];
      }
    }
  }
  op.close(context);

  ASSERT_EQ(key7_lvals.size(), 2u);
  std::sort(key7_lvals.begin(), key7_lvals.end());
  EXPECT_DOUBLE_EQ(key7_lvals[0], 1.0);
  EXPECT_DOUBLE_EQ(key7_lvals[1], 2.0);
  EXPECT_DOUBLE_EQ(key7_rval, 700.0);
  EXPECT_DOUBLE_EQ(key9_lval, 90.0);
  EXPECT_DOUBLE_EQ(key9_rval, 900.0);
}

// Build side empty overall (not just one empty bucket among others, as
// PartitionedJoinMatchesAcrossMultipleBuckets already covers incidentally)
// -- every bucket's build_partitions_ entry is empty, so next() should
// walk straight through every bucket and report exhausted without ever
// calling ensure_partition_built()/ever touching right_schema_ (which
// stays null the whole time, since right_ never produced a batch).
TEST(HashJoinOperator, PartitionedJoinEmptyBuildSideProducesNoRows) {
  RmmEnvironment env(default_config());
  const CudaStream stream;
  ExecutionContext context = make_context(stream.get());

  std::vector<DeviceBatch> left_batches;
  left_batches.push_back(make_left_batch({1, 2, 3}, {10.0, 20.0, 30.0}));

  HashJoinOperator op(1, std::make_unique<VectorSourceOperator>(std::move(left_batches)),
                      std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}),
                      /*left_key_index=*/0,
                      /*right_key_index=*/0, join_output_schema(), /*partition_count=*/4);
  op.open(context);
  EXPECT_FALSE(op.next(context).has_value());
  op.close(context);
}

}  // namespace
}  // namespace kernellake
