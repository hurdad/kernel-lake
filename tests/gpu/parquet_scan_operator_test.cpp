#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <filesystem>

#include "kernellake/execution/parquet_scan_operator.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

ExecutionContext make_context() {
  return ExecutionContext{"test-query", 0,       nullptr, rmm::mr::get_current_device_resource_ref(),
                          nullptr,      nullptr, nullptr};
}

template <typename T>
std::vector<T> copy_to_host(const cudf::column_view& view) {
  std::vector<T> host(static_cast<std::size_t>(view.size()));
  cudaMemcpy(host.data(), view.data<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return host;
}

class ParquetScanOperatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_parquet_scan_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();

    // Row group 0: id 0-4, region "A". Row group 1: id 5-9, region "B".
    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::StringBuilder region_builder;
    for (int64_t i = 0; i < 10; ++i) {
      ASSERT_TRUE(id_builder.Append(i).ok());
      ASSERT_TRUE(amount_builder.Append(static_cast<double>(i) * 1.5).ok());
      ASSERT_TRUE(region_builder.Append(i < 5 ? "A" : "B").ok());
    }
    std::shared_ptr<arrow::Array> id_array, amount_array, region_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false),
                                       arrow::field("amount", arrow::float64(), false),
                                       arrow::field("region", arrow::utf8(), false)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array, region_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
};

TEST_F(ParquetScanOperatorTest, ReadsAllSelectedColumnsAndRowGroups) {
  RmmEnvironment env(default_config());
  std::vector<PhysicalFileFragment> fragments = {PhysicalFileFragment{Uri(path_), 10, 2, {0, 1}, {}, {}}};
  Schema schema({Field{"id", int64_type(false)}, Field{"amount", float64_type(false)}});

  ParquetScanOperator scan(1, fragments, {"id", "amount"}, std::make_shared<const Schema>(schema));
  ExecutionContext context = make_context();
  scan.open(context);

  std::size_t total_rows = 0;
  std::vector<std::int64_t> all_ids;
  while (std::optional<DeviceBatch> batch = scan.next(context)) {
    EXPECT_EQ(batch->column_count(), 2u);
    total_rows += batch->row_count();
    std::vector<std::int64_t> ids = copy_to_host<std::int64_t>(batch->view().column(0));
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());
  }
  EXPECT_EQ(total_rows, 10u);
  std::sort(all_ids.begin(), all_ids.end());
  for (std::int64_t i = 0; i < 10; ++i) EXPECT_EQ(all_ids[static_cast<std::size_t>(i)], i);
  scan.close(context);
}

TEST_F(ParquetScanOperatorTest, ReadsOnlySelectedRowGroup) {
  RmmEnvironment env(default_config());
  // Only row group 1 (id 5-9) selected, matching a pruning decision that
  // proved row group 0 unnecessary.
  std::vector<PhysicalFileFragment> fragments = {
      PhysicalFileFragment{Uri(path_), 10, 2, {1}, {0}, {"row_group 0 skipped: test"}}};
  Schema schema({Field{"id", int64_type(false)}});

  ParquetScanOperator scan(1, fragments, {"id"}, std::make_shared<const Schema>(schema));
  ExecutionContext context = make_context();
  scan.open(context);

  std::size_t total_rows = 0;
  std::vector<std::int64_t> all_ids;
  while (std::optional<DeviceBatch> batch = scan.next(context)) {
    total_rows += batch->row_count();
    std::vector<std::int64_t> ids = copy_to_host<std::int64_t>(batch->view().column(0));
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());
  }
  EXPECT_EQ(total_rows, 5u);
  for (std::int64_t id : all_ids) EXPECT_GE(id, 5);
  scan.close(context);
}

TEST_F(ParquetScanOperatorTest, EmptyFragmentListProducesNoBatches) {
  RmmEnvironment env(default_config());
  Schema schema({Field{"id", int64_type(false)}});
  ParquetScanOperator scan(1, {}, {"id"}, std::make_shared<const Schema>(schema));
  ExecutionContext context = make_context();
  scan.open(context);
  EXPECT_FALSE(scan.next(context).has_value());
  scan.close(context);
}

}  // namespace
}  // namespace kernellake
