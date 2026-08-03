#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "kernellake/common/errors.hpp"
#include "kernellake/generator/sample_data_generator.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class SampleDataGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_sample_data_generator_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
  }

  void TearDown() override { fs::remove_all(dir_); }

  [[nodiscard]] std::shared_ptr<arrow::Table> read_table(const std::string& path) const {
    std::shared_ptr<arrow::io::ReadableFile> file = arrow::io::ReadableFile::Open(path).ValueOrDie();
    std::unique_ptr<parquet::arrow::FileReader> reader =
        parquet::arrow::OpenFile(file, arrow::default_memory_pool()).ValueOrDie();
    return reader->ReadTable().ValueOrDie();
  }

  fs::path dir_;
};

TEST_F(SampleDataGeneratorTest, WritesRequestedRowCountAndFileCount) {
  SampleDataGeneratorOptions options;
  options.output_dir = dir_.string();
  options.rows = 1000;
  options.files = 3;
  options.row_group_rows = 200;
  options.seed = 7;

  const SampleDataGenerationResult result = generate_sample_data(options);
  EXPECT_EQ(result.rows_written, 1000);
  ASSERT_EQ(result.file_paths.size(), 3u);

  std::int64_t total_rows = 0;
  for (const std::string& path : result.file_paths) {
    ASSERT_TRUE(fs::exists(path));
    total_rows += read_table(path)->num_rows();
  }
  EXPECT_EQ(total_rows, 1000);
}

TEST_F(SampleDataGeneratorTest, ProducesExpectedSchemaAndUniqueSequentialOrderIds) {
  SampleDataGeneratorOptions options;
  options.output_dir = dir_.string();
  options.rows = 500;
  options.files = 2;
  options.seed = 1;

  const SampleDataGenerationResult result = generate_sample_data(options);
  std::set<std::int64_t> order_ids;
  for (const std::string& path : result.file_paths) {
    const std::shared_ptr<arrow::Table> table = read_table(path);
    const std::shared_ptr<arrow::Schema>& schema = table->schema();
    ASSERT_EQ(schema->num_fields(), 8);
    EXPECT_EQ(schema->field(0)->name(), "order_id");
    EXPECT_EQ(schema->field(1)->name(), "customer_id");
    EXPECT_EQ(schema->field(2)->name(), "region");
    EXPECT_EQ(schema->field(3)->name(), "amount");
    EXPECT_EQ(schema->field(4)->name(), "event_date");
    EXPECT_EQ(schema->field(5)->name(), "event_time");
    EXPECT_EQ(schema->field(6)->name(), "category");
    EXPECT_EQ(schema->field(7)->name(), "discount");
    EXPECT_TRUE(schema->field(7)->nullable());
    EXPECT_FALSE(schema->field(0)->nullable());

    const std::shared_ptr<arrow::ChunkedArray> order_id_column = table->column(0);
    for (int chunk = 0; chunk < order_id_column->num_chunks(); ++chunk) {
      const auto array = std::static_pointer_cast<arrow::Int64Array>(order_id_column->chunk(chunk));
      for (std::int64_t i = 0; i < array->length(); ++i) order_ids.insert(array->Value(i));
    }
  }
  // Sequential and unique across every file, with no gaps.
  EXPECT_EQ(order_ids.size(), 500u);
  EXPECT_EQ(*order_ids.begin(), 0);
  EXPECT_EQ(*order_ids.rbegin(), 499);
}

TEST_F(SampleDataGeneratorTest, SameSeedProducesByteIdenticalOutput) {
  SampleDataGeneratorOptions options;
  options.output_dir = (dir_ / "run1").string();
  options.rows = 300;
  options.files = 1;
  options.seed = 99;
  const SampleDataGenerationResult first = generate_sample_data(options);

  options.output_dir = (dir_ / "run2").string();
  const SampleDataGenerationResult second = generate_sample_data(options);

  ASSERT_EQ(first.file_paths.size(), 1u);
  ASSERT_EQ(second.file_paths.size(), 1u);
  ASSERT_TRUE(first.file_paths[0] != second.file_paths[0]);

  std::ifstream file_a(first.file_paths[0], std::ios::binary);
  std::ifstream file_b(second.file_paths[0], std::ios::binary);
  std::ostringstream contents_a, contents_b;
  contents_a << file_a.rdbuf();
  contents_b << file_b.rdbuf();
  EXPECT_EQ(contents_a.str(), contents_b.str());
}

TEST_F(SampleDataGeneratorTest, NullRateOfOneMakesDiscountAlwaysNull) {
  SampleDataGeneratorOptions options;
  options.output_dir = dir_.string();
  options.rows = 200;
  options.files = 1;
  options.null_rate = 1.0;
  options.seed = 3;

  const SampleDataGenerationResult result = generate_sample_data(options);
  const std::shared_ptr<arrow::Table> table = read_table(result.file_paths.front());
  EXPECT_EQ(table->column(7)->null_count(), 200);
}

TEST_F(SampleDataGeneratorTest, RejectsInvalidOptions) {
  SampleDataGeneratorOptions options;
  options.output_dir = dir_.string();
  options.rows = 0;
  EXPECT_THROW((void)(generate_sample_data(options)), ConfigurationError);

  options.rows = 10;
  options.files = 0;
  EXPECT_THROW((void)(generate_sample_data(options)), ConfigurationError);

  options.files = 1;
  options.null_rate = 1.5;
  EXPECT_THROW((void)(generate_sample_data(options)), ConfigurationError);
}

}  // namespace
}  // namespace kernellake
