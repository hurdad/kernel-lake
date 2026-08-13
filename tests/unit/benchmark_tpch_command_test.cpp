// run_benchmark_tpch() is a pure (args, config) -> int function, driven
// directly here rather than shelling out -- see query_command_test.cpp's
// own comment on why. Primary purpose: regression coverage for the real
// bug this closes (benchmark_tpch_command.cpp:187-196) -- --scale-factor/
// --query/--iterations/--warmup-iterations used to be parsed with raw
// std::stod/std::stoi *outside* run_benchmark_tpch()'s own try block, so a
// malformed value (e.g. --query abc) threw std::invalid_argument uncaught
// instead of producing the same clean "kernellake benchmark tpch: ..."
// error every sibling command gives for a bad argument.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "kernellake/cli/commands.hpp"
#include "kernellake/common/config.hpp"

namespace kernellake::cli {
namespace {

namespace fs = std::filesystem;

TEST(BenchmarkTpchCommand, InvalidQueryArgumentProducesCleanErrorNotAnUncaughtThrow) {
  // Before the fix, std::stoi(std::string("abc")) threw
  // std::invalid_argument straight out of the argument-parsing loop,
  // uncaught by run_benchmark_tpch()'s own try block further down --
  // EXPECT_EQ (a normal return, not EXPECT_THROW/a crash) is the point of
  // this test.
  const std::vector<std::string_view> args = {"--data", "/nonexistent/*.parquet", "--query", "abc", "--mode",
                                              "warm"};
  EXPECT_EQ(run_benchmark_tpch(args, default_config()), 1);
}

TEST(BenchmarkTpchCommand, InvalidScaleFactorArgumentProducesCleanErrorNotAnUncaughtThrow) {
  const std::vector<std::string_view> args = {
      "--data", "/nonexistent/*.parquet", "--query", "1", "--mode", "warm", "--scale-factor", "not-a-number"};
  EXPECT_EQ(run_benchmark_tpch(args, default_config()), 1);
}

TEST(BenchmarkTpchCommand, InvalidIterationsArgumentProducesCleanErrorNotAnUncaughtThrow) {
  const std::vector<std::string_view> args = {"--data", "/nonexistent/*.parquet", "--query", "1", "--mode",
                                              "warm",   "--iterations",           "xyz"};
  EXPECT_EQ(run_benchmark_tpch(args, default_config()), 1);
}

TEST(BenchmarkTpchCommand, InvalidWarmupIterationsArgumentProducesCleanErrorNotAnUncaughtThrow) {
  const std::vector<std::string_view> args = {
      "--data", "/nonexistent/*.parquet", "--query",     "1", "--mode",
      "warm",   "--warmup-iterations",    "not-a-number"};
  EXPECT_EQ(run_benchmark_tpch(args, default_config()), 1);
}

TEST(BenchmarkTpchCommand, RejectsMissingQuery) {
  const std::vector<std::string_view> args = {"--data", "/nonexistent/*.parquet", "--mode", "warm"};
  EXPECT_EQ(run_benchmark_tpch(args, default_config()), 1);
}

TEST(BenchmarkTpchCommand, RejectsInvalidMode) {
  const std::vector<std::string_view> args = {"--data", "/nonexistent/*.parquet", "--query", "1", "--mode",
                                              "hot"};
  EXPECT_EQ(run_benchmark_tpch(args, default_config()), 1);
}

class BenchmarkTpchCommandRealRunTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_benchmark_tpch_command_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    data_path_ = (dir_ / "sales.parquet").string();
    query_file_path_ = (dir_ / "q.sql").string();
    output_path_ = (dir_ / "report.json").string();

    arrow::Int64Builder id_builder;
    for (std::int64_t i = 0; i < 5; ++i) {
      ASSERT_TRUE(id_builder.Append(i).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
    const auto table = arrow::Table::Make(schema, {id_array});
    auto sink = arrow::io::FileOutputStream::Open(data_path_).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5);
    ASSERT_TRUE(status.ok()) << status.ToString();

    // {data} is substituted with --data's own value by
    // strip_comments_and_substitute() (benchmark_tpch_command.cpp), the
    // same placeholder scheme benchmarks/tpch/queries/*.sql itself uses.
    std::ofstream query_file(query_file_path_);
    query_file << "SELECT COUNT(*) AS n FROM read_parquet('{data}')";
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string data_path_;
  std::string query_file_path_;
  std::string output_path_;
};

// End-to-end: valid --scale-factor/--query/--iterations/--warmup-iterations
// values (the same arguments the tests above feed malformed) parse and
// drive a real query run, closing the loop on the parsing fix rather than
// only exercising its failure path.
TEST_F(BenchmarkTpchCommandRealRunTest, ValidNumericArgumentsProduceARealBenchmarkReport) {
  // default_config().engine.backend is "gpu" -- run_benchmark_tpch() has no
  // --backend flag of its own (unlike run_query()), so the config itself
  // must already say "cpu" for this build (KERNELLAKE_WITH_CUDA=OFF) to run
  // the query at all instead of throwing "query execution requires GPU
  // operators...".
  EngineConfig config = default_config();
  config.engine.backend = "cpu";

  const std::vector<std::string_view> args = {"--data",
                                              data_path_,
                                              "--query-file",
                                              query_file_path_,
                                              "--query",
                                              "1",
                                              "--mode",
                                              "warm",
                                              "--scale-factor",
                                              "0.01",
                                              "--iterations",
                                              "1",
                                              "--warmup-iterations",
                                              "0",
                                              "--output",
                                              output_path_};
  EXPECT_EQ(run_benchmark_tpch(args, config), 0);

  std::ifstream report_file(output_path_);
  std::ostringstream contents;
  contents << report_file.rdbuf();
  const nlohmann::json report = nlohmann::json::parse(contents.str());
  EXPECT_EQ(report["query"], 1);
  EXPECT_DOUBLE_EQ(report["scale_factor"].get<double>(), 0.01);
  EXPECT_EQ(report["iteration_results"][0]["rows_returned"], 1);
}

}  // namespace
}  // namespace kernellake::cli
