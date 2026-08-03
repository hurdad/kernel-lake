// Integration coverage for the gap flagged in docs/ROADMAP.md: dictionary-
// encoded strings, multiple files with mismatched row-group layouts, and
// datasets that don't fit in a single execution batch, all exercised
// through the real pipeline (build_physical_plan + build_operator_tree,
// the same path QueryEngine::execute() uses) rather than only manually.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>

#include <filesystem>
#include <map>
#include <numeric>

#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/execution_gpu/operator_builder.hpp"
#include "kernellake/execution_gpu/parquet_scan_operator.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/optimizer/optimizer.hpp"
#include "kernellake/planner/binder.hpp"
#include "kernellake/planner/logical_planner.hpp"
#include "kernellake/sql/parser.hpp"
#include "kernellake/storage/file_discovery.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

ExecutionContext make_context() {
  return ExecutionContext{"test-query", 0,       nullptr, rmm::mr::get_current_device_resource_ref(),
                          nullptr,      nullptr, nullptr};
}

// Writes `row_count` rows (region alternating A/B starting with A, amount
// constant at `amount_value`) to `path`, split into row groups of
// `chunk_size` rows, with dictionary encoding on `region` either forced on
// or off -- so the two files this test writes have deliberately mismatched
// row-group layouts (different chunk sizes -> different row-group counts)
// and different encodings for the same logical column.
void write_file(const std::string& path, int row_count, int chunk_size, double amount_value,
                bool dictionary_encoding) {
  arrow::StringBuilder region_builder;
  arrow::DoubleBuilder amount_builder;
  for (int i = 0; i < row_count; ++i) {
    ASSERT_TRUE(region_builder.Append(i % 2 == 0 ? "A" : "B").ok());
    ASSERT_TRUE(amount_builder.Append(amount_value).ok());
  }
  std::shared_ptr<arrow::Array> region_array, amount_array;
  ASSERT_TRUE(region_builder.Finish(&region_array).ok());
  ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
  const auto schema = arrow::schema(
      {arrow::field("region", arrow::utf8(), false), arrow::field("amount", arrow::float64(), false)});
  const auto table = arrow::Table::Make(schema, {region_array, amount_array});

  parquet::WriterProperties::Builder builder;
  if (dictionary_encoding) {
    builder.enable_dictionary();
  } else {
    builder.disable_dictionary();
  }
  const std::shared_ptr<parquet::WriterProperties> properties = builder.build();

  auto sink = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  const arrow::Status status =
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, chunk_size, properties);
  ASSERT_TRUE(status.ok()) << status.ToString();
}

class MultiBatchIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_multi_batch_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);

    // File A: 300 rows, 3 row groups of 100, dictionary-encoded region.
    write_file((dir_ / "part-a.parquet").string(), 300, 100, 1.0, /*dictionary_encoding=*/true);
    // File B: 250 rows, a single row group of 250, plain (non-dictionary)
    // region -- deliberately mismatched against file A's row-group layout
    // and encoding.
    write_file((dir_ / "part-b.parquet").string(), 250, 250, 2.0, /*dictionary_encoding=*/false);

    glob_ = (dir_ / "*.parquet").string();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string glob_;
};

TEST_F(MultiBatchIntegrationTest, ScanEmitsMultipleBatchesAcrossMismatchedFilesWithSmallPassLimit) {
  RmmEnvironment env(default_config());
  LocalObjectStore store;
  const std::vector<ObjectInfo> files = discover_parquet_files(store, {glob_});
  ASSERT_EQ(files.size(), 2u);

  std::vector<PhysicalFileFragment> fragments;
  for (const ObjectInfo& file : files) {
    // 300-row file: 3 row groups; 250-row file: 1 row group -- genuinely
    // mismatched layouts, not just different files.
    const int row_groups = file.uri.value().find("part-a") != std::string::npos ? 3 : 1;
    const int rows = file.uri.value().find("part-a") != std::string::npos ? 300 : 250;
    std::vector<int> selected(static_cast<std::size_t>(row_groups));
    std::iota(selected.begin(), selected.end(), 0);
    fragments.push_back(PhysicalFileFragment{file.uri, rows, row_groups, selected, {}, {}});
  }
  Schema schema({Field{"region", string_type(false)}, Field{"amount", float64_type(false)}});

  // A few hundred bytes is far smaller than the ~550 rows' decompressed
  // size, forcing chunked_parquet_reader into many passes instead of one
  // per file.
  ParquetScanOperator scan(1, fragments, {"region", "amount"}, std::make_shared<const Schema>(schema), store,
                           /*pass_read_limit_bytes=*/256);
  ExecutionContext context = make_context();
  scan.open(context);

  int batch_count = 0;
  std::size_t total_rows = 0;
  while (std::optional<DeviceBatch> batch = scan.next(context)) {
    ++batch_count;
    total_rows += batch->row_count();
  }
  scan.close(context);

  EXPECT_EQ(total_rows, 550u);
  // Proving this is genuinely multi-batch, not one batch per file.
  EXPECT_GT(batch_count, 2);
}

TEST_F(MultiBatchIntegrationTest, FullPipelineCorrectAcrossMultipleBatchesAndMismatchedRowGroups) {
  const auto stmt = sql::parse_sql("SELECT region, SUM(amount) AS total, COUNT(*) AS n FROM read_parquet('" +
                                   glob_ + "') GROUP BY region");
  Schema base_schema({Field{"region", string_type(false)}, Field{"amount", float64_type(false)}});
  const BoundQuery bound = bind_query(stmt, base_schema);
  LogicalPlanPtr logical = build_logical_plan(bound, base_schema);
  logical = optimize(std::move(logical));

  LocalObjectStore store;
  const PhysicalPlanPtr physical = build_physical_plan(logical, store);

  RmmEnvironment env(default_config());
  const CudaStream stream;
  ExecutionContext context{"test-query", 0,       stream.get(), rmm::mr::get_current_device_resource_ref(),
                           nullptr,      nullptr, nullptr};

  // Same deliberately tiny pass-read limit as the scan-only test above, so
  // this exercises the full Scan -> HashAggregate -> ArrowResult pipeline
  // (the same one QueryEngine::execute() builds) under genuine multi-batch
  // conditions instead of the common case of one batch per file.
  const std::unique_ptr<PhysicalOperator> root =
      build_operator_tree(physical, store, /*pass_read_limit_bytes=*/256);
  root->open(context);
  std::map<std::string, std::pair<double, std::int64_t>> totals_by_region;
  std::int64_t rows_returned = 0;
  while (std::optional<DeviceBatch> batch = root->next(context)) {
    rows_returned += static_cast<std::int64_t>(batch->row_count());
    const cudf::table_view view = batch->view();
    std::vector<double> totals(static_cast<std::size_t>(view.num_rows()));
    std::vector<std::int64_t> counts(static_cast<std::size_t>(view.num_rows()));
    cudaMemcpy(totals.data(), view.column(1).data<double>(), totals.size() * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(counts.data(), view.column(2).data<std::int64_t>(), counts.size() * sizeof(std::int64_t),
               cudaMemcpyDeviceToHost);
    for (cudf::size_type i = 0; i < view.num_rows(); ++i) {
      const std::unique_ptr<cudf::scalar> region_scalar = cudf::get_element(view.column(0), i);
      const auto& region_str = static_cast<const cudf::string_scalar&>(*region_scalar);
      totals_by_region[region_str.to_string()] = {totals[static_cast<std::size_t>(i)],
                                                  counts[static_cast<std::size_t>(i)]};
    }
  }
  root->close(context);

  EXPECT_EQ(rows_returned, 2);
  ASSERT_EQ(totals_by_region.size(), 2u);
  EXPECT_DOUBLE_EQ(totals_by_region.at("A").first, 400.0);
  EXPECT_EQ(totals_by_region.at("A").second, 275);
  EXPECT_DOUBLE_EQ(totals_by_region.at("B").first, 400.0);
  EXPECT_EQ(totals_by_region.at("B").second, 275);
}

}  // namespace
}  // namespace kernellake
