#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>

#include <filesystem>

#include "kernellake/execution/parquet_scan_operator.hpp"
#include "kernellake/execution/sort_operator.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

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

Schema one_int_column_schema() {
  return Schema({Field{"a", int32_type(false)}});
}

DeviceBatch make_int_batch(const std::vector<std::int32_t>& values) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_from_host(values, cudf::type_id::INT32));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                     std::make_shared<const Schema>(one_int_column_schema()));
}

TEST(SortOperator, SortsAscendingAcrossMultipleBatches) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_int_batch({5, 1, 3}));
  batches.push_back(make_int_batch({4, 2}));

  auto key = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  std::vector<LogicalSort::Key> keys = {LogicalSort::Key{key, /*ascending=*/true}};
  SortOperator sort(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(keys));
  ExecutionContext context = make_context();
  sort.open(context);

  std::optional<DeviceBatch> result = sort.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->row_count(), 5u);
  EXPECT_EQ(copy_to_host<std::int32_t>(result->view().column(0)), (std::vector<std::int32_t>{1, 2, 3, 4, 5}));
  EXPECT_FALSE(sort.next(context).has_value());  // blocking operator: exactly one output batch
  sort.close(context);
}

TEST(SortOperator, SortsDescendingByComputedExpression) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_int_batch({5, 1, 3, 4, 2}));

  auto column = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  // Sort by -a ascending == sort by a descending; exercises the
  // compute_column (non-plain-column) path rather than the direct-column-
  // reference fast path. cudf::ast requires exact operand type matches (no
  // implicit coercion) and only supports widening CASTs to INT64/UINT64/
  // FLOAT64, so the INT32 column is cast up to INT64 to match the zero
  // literal -- the same kind of cast the real binder would insert.
  auto widened_column = std::make_shared<CastExpression>(column, int64_type(false));
  auto zero = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(0));
  auto negated =
      std::make_shared<BinaryExpression>(BinaryOperator::Subtract, zero, widened_column, int64_type(false));
  std::vector<LogicalSort::Key> keys = {LogicalSort::Key{negated, /*ascending=*/true}};
  SortOperator sort(1, std::make_unique<VectorSourceOperator>(std::move(batches)), std::move(keys));
  ExecutionContext context = make_context();
  sort.open(context);

  std::optional<DeviceBatch> result = sort.next(context);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(copy_to_host<std::int32_t>(result->view().column(0)), (std::vector<std::int32_t>{5, 4, 3, 2, 1}));
  sort.close(context);
}

TEST(SortOperator, EmptyInputProducesNoBatches) {
  RmmEnvironment env(default_config());
  auto key = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  std::vector<LogicalSort::Key> keys = {LogicalSort::Key{key, true}};
  SortOperator sort(1, std::make_unique<VectorSourceOperator>(std::vector<DeviceBatch>{}), std::move(keys));
  ExecutionContext context = make_context();
  sort.open(context);
  EXPECT_FALSE(sort.next(context).has_value());
  sort.close(context);
}

class SortOperatorStringKeyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_sort_string_key_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "data.parquet").string();

    arrow::StringBuilder region_builder;
    for (const std::string& region : {"banana", "apple", "cherry"}) {
      ASSERT_TRUE(region_builder.Append(region).ok());
    }
    std::shared_ptr<arrow::Array> region_array;
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    const auto schema = arrow::schema({arrow::field("region", arrow::utf8(), false)});
    const auto table = arrow::Table::Make(schema, {region_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, 3);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
};

TEST_F(SortOperatorStringKeyTest, SortsPlainStringColumnReferenceWithoutHittingNonFixedWidthBug) {
  RmmEnvironment env(default_config());
  Schema schema({Field{"region", string_type(false)}});
  std::vector<PhysicalFileFragment> fragments = {PhysicalFileFragment{Uri(path_), 3, 1, {0}, {}, {}}};
  LocalObjectStore store;
  auto scan = std::make_unique<ParquetScanOperator>(1, fragments, std::vector<std::string>{"region"},
                                                    std::make_shared<const Schema>(schema), store);

  auto key = std::make_shared<ColumnExpression>("region", 0, string_type(false));
  std::vector<LogicalSort::Key> keys = {LogicalSort::Key{key, /*ascending=*/true}};
  SortOperator sort(2, std::move(scan), std::move(keys));
  ExecutionContext context = make_context();
  sort.open(context);

  std::optional<DeviceBatch> result = sort.next(context);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->row_count(), 3u);
  const cudf::column_view region_view = result->view().column(0);
  std::vector<std::string> ordered;
  for (cudf::size_type i = 0; i < region_view.size(); ++i) {
    const std::unique_ptr<cudf::scalar> element = cudf::get_element(region_view, i);
    ordered.push_back(static_cast<const cudf::string_scalar&>(*element).to_string());
  }
  EXPECT_EQ(ordered, (std::vector<std::string>{"apple", "banana", "cherry"}));
  sort.close(context);
}

}  // namespace
}  // namespace kernellake
