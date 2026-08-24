#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <map>

#include "kernellake/execution_gpu/parquet_scan_operator.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/storage/local_object_store.hpp"

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

  LocalObjectStore store;
  ParquetScanOperator scan(1, fragments, {"id", "amount"}, std::make_shared<const Schema>(schema), store);
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

  LocalObjectStore store;
  ParquetScanOperator scan(1, fragments, {"id"}, std::make_shared<const Schema>(schema), store);
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

TEST_F(ParquetScanOperatorTest, MaterializesPartitionColumnsPerFragment) {
  RmmEnvironment env(default_config());
  // Two "fragments" over the same file, one per row group, each assigned a
  // different constant partition value -- simulating two Hive partition
  // directories (e.g. period_code=100/part-0.parquet,
  // period_code=200/part-0.parquet) without needing two physical files,
  // since PhysicalFileFragment doesn't care whether two fragments share an
  // underlying path. Exercises the per-fragment reader path (see
  // ParquetScanOperator's own class comment on why partition_columns
  // forces that mode) and confirms each fragment's own constant value is
  // attached only to that fragment's own rows, never mixed across
  // fragments.
  std::vector<PhysicalFileFragment> fragments = {
      PhysicalFileFragment{Uri(path_), 5, 2, {0}, {1}, {}, {std::int64_t{100}}},
      PhysicalFileFragment{Uri(path_), 5, 2, {1}, {0}, {}, {std::int64_t{200}}},
  };
  std::vector<PartitionColumn> partition_columns = {PartitionColumn{"period_code", int64_type(false)}};
  Schema schema({Field{"id", int64_type(false)}, Field{"period_code", int64_type(false)}});

  LocalObjectStore store;
  ParquetScanOperator scan(1, fragments, {"id"}, std::make_shared<const Schema>(schema), store,
                           /*pass_read_limit_bytes=*/0, partition_columns);
  ExecutionContext context = make_context();
  scan.open(context);

  std::map<std::int64_t, std::int64_t> period_by_id;
  while (std::optional<DeviceBatch> batch = scan.next(context)) {
    ASSERT_EQ(batch->column_count(), 2u);
    const std::vector<std::int64_t> ids = copy_to_host<std::int64_t>(batch->view().column(0));
    const std::vector<std::int64_t> periods = copy_to_host<std::int64_t>(batch->view().column(1));
    ASSERT_EQ(ids.size(), periods.size());
    for (std::size_t i = 0; i < ids.size(); ++i) period_by_id[ids[i]] = periods[i];
  }
  scan.close(context);

  ASSERT_EQ(period_by_id.size(), 10u);
  for (std::int64_t id = 0; id < 5; ++id) EXPECT_EQ(period_by_id.at(id), 100);
  for (std::int64_t id = 5; id < 10; ++id) EXPECT_EQ(period_by_id.at(id), 200);
}

// Wraps a real LocalObjectStore but strips a fake non-"file" scheme prefix
// off every Uri before delegating -- lets a test construct a fragment
// whose Uri::scheme() is something other than "file" (routing open()/
// open_current_fragment() through their ObjectStoreDatasource branch,
// never exercised by every other test in this file, which all use plain
// local paths) while still reading a real, real GPU-decoded file on disk,
// without needing an actual S3/GCS/Azure backend.
class FakeSchemeObjectStore final : public ObjectStore {
 public:
  explicit FakeSchemeObjectStore(ObjectStore& delegate) : delegate_(delegate) {}

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override {
    return delegate_.list(strip(prefix));
  }
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override {
    return delegate_.list_recursive(strip(prefix));
  }
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override {
    return delegate_.open(strip(uri));
  }

 private:
  static Uri strip(const Uri& uri) {
    constexpr std::string_view kPrefix = "fake://";
    return Uri(uri.value().substr(kPrefix.size()));
  }

  ObjectStore& delegate_;
};

// open()'s non-local (ObjectStoreDatasource) branch had no coverage --
// every other test in this file uses a plain "file"-scheme Uri, always
// taking the all-local cudf::io::source_info fast path instead.
TEST_F(ParquetScanOperatorTest, NonLocalSchemeRoutesThroughObjectStoreDatasource) {
  RmmEnvironment env(default_config());
  std::vector<PhysicalFileFragment> fragments = {
      PhysicalFileFragment{Uri("fake://" + path_), 10, 2, {0, 1}, {}, {}}};
  Schema schema({Field{"id", int64_type(false)}, Field{"amount", float64_type(false)}});

  LocalObjectStore local_store;
  FakeSchemeObjectStore store(local_store);
  ParquetScanOperator scan(1, fragments, {"id", "amount"}, std::make_shared<const Schema>(schema), store);
  ExecutionContext context = make_context();
  scan.open(context);

  std::size_t total_rows = 0;
  while (std::optional<DeviceBatch> batch = scan.next(context)) total_rows += batch->row_count();
  EXPECT_EQ(total_rows, 10u);
  scan.close(context);
}

// open_current_fragment()'s non-local branch (the per-fragment/partitioned
// mode's own equivalent of the above) had no coverage either.
TEST_F(ParquetScanOperatorTest, NonLocalSchemeRoutesThroughObjectStoreDatasourceInPerFragmentMode) {
  RmmEnvironment env(default_config());
  std::vector<PhysicalFileFragment> fragments = {
      PhysicalFileFragment{Uri("fake://" + path_), 10, 2, {0, 1}, {}, {}, {std::int64_t{7}}}};
  std::vector<PartitionColumn> partition_columns = {PartitionColumn{"batch_id", int64_type(false)}};
  Schema schema({Field{"id", int64_type(false)}, Field{"amount", float64_type(false)},
                 Field{"batch_id", int64_type(false)}});

  LocalObjectStore local_store;
  FakeSchemeObjectStore store(local_store);
  ParquetScanOperator scan(1, fragments, {"id", "amount"}, std::make_shared<const Schema>(schema), store,
                           /*pass_read_limit_bytes=*/std::numeric_limits<std::size_t>::max(),
                           partition_columns);
  ExecutionContext context = make_context();
  scan.open(context);

  std::size_t total_rows = 0;
  while (std::optional<DeviceBatch> batch = scan.next(context)) {
    total_rows += batch->row_count();
    const std::vector<std::int64_t> batch_ids = copy_to_host<std::int64_t>(batch->view().column(2));
    for (std::int64_t id : batch_ids) EXPECT_EQ(id, 7);
  }
  EXPECT_EQ(total_rows, 10u);
  scan.close(context);
}

TEST_F(ParquetScanOperatorTest, EmptyFragmentListProducesNoBatches) {
  RmmEnvironment env(default_config());
  Schema schema({Field{"id", int64_type(false)}});
  LocalObjectStore store;
  ParquetScanOperator scan(1, {}, {"id"}, std::make_shared<const Schema>(schema), store);
  ExecutionContext context = make_context();
  scan.open(context);
  EXPECT_FALSE(scan.next(context).has_value());
  scan.close(context);
}

}  // namespace
}  // namespace kernellake
