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
  // default_config().engine.backend is "gpu" -- run_benchmark_tpch() has no
  // --backend flag of its own (unlike run_query()), so the config itself
  // must already say "cpu" for this build (KERNELLAKE_WITH_CUDA=OFF) to run
  // the query at all instead of throwing "query execution requires GPU
  // operators...".
  EngineConfig config_ = [] {
    EngineConfig config = default_config();
    config.engine.backend = "cpu";
    return config;
  }();
};

TEST_F(BenchmarkTpchCommandRealRunTest, RejectsMissingData) {
  const std::vector<std::string_view> args = {"--query-file", query_file_path_, "--query", "1", "--mode",
                                              "warm"};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 1);
}

TEST_F(BenchmarkTpchCommandRealRunTest, RejectsExecutionOnlyModeAsNotYetImplemented) {
  const std::vector<std::string_view> args = {"--data",  data_path_, "--query-file", query_file_path_,
                                              "--query", "1",        "--mode",       "execution-only"};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 1);
}

TEST_F(BenchmarkTpchCommandRealRunTest, RejectsNonPositiveIterations) {
  const std::vector<std::string_view> args = {"--data",       data_path_, "--query-file", query_file_path_,
                                              "--query",      "1",        "--mode",       "warm",
                                              "--iterations", "0"};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 1);
}

TEST_F(BenchmarkTpchCommandRealRunTest, RejectsUnreadableQueryFile) {
  const std::string missing_query_file = (dir_ / "does-not-exist.sql").string();
  const std::vector<std::string_view> args = {"--data",  data_path_, "--query-file", missing_query_file,
                                              "--query", "1",        "--mode",       "warm"};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 1);
}

// Exercises one of the seven near-identical extra-table discover_parquet_files()
// blocks' "no files matched" branch (--part-data/--orders-data/etc. are all the
// same shape -- see the source file's own comment on why there are seven) --
// representative of all seven, which otherwise share this exact control flow.
TEST_F(BenchmarkTpchCommandRealRunTest, RejectsWhenExtraTableGlobMatchesNoFiles) {
  const fs::path empty_dir = dir_ / "no_part_data";
  fs::create_directories(empty_dir);
  const std::string part_glob = (empty_dir / "*.parquet").string();
  const std::vector<std::string_view> args = {"--data",      data_path_, "--query-file", query_file_path_,
                                              "--query",     "1",        "--mode",       "warm",
                                              "--part-data", part_glob};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 1);
}

// Covers the assignment/discovery/report-population lines for every one of
// the seven optional extra-table flags in a single run (they don't need to
// be referenced by the query's own SQL -- discover_parquet_files() and the
// report["..._data"] population both run purely off whether each flag was
// given, independent of what strip_comments_and_substitute() does with it).
TEST_F(BenchmarkTpchCommandRealRunTest, IncludesEveryOptionalTableGlobInReport) {
  const std::vector<std::string_view> args = {
      "--data",          data_path_, "--query-file",    query_file_path_, "--query",         "1",
      "--mode",          "warm",     "--part-data",     data_path_,       "--orders-data",   data_path_,
      "--customer-data", data_path_, "--nation-data",   data_path_,       "--supplier-data", data_path_,
      "--region-data",   data_path_, "--partsupp-data", data_path_,       "--output",        output_path_};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 0);

  std::ifstream report_file(output_path_);
  std::ostringstream contents;
  contents << report_file.rdbuf();
  const nlohmann::json report = nlohmann::json::parse(contents.str());
  EXPECT_EQ(report["part_data"], data_path_);
  EXPECT_EQ(report["orders_data"], data_path_);
  EXPECT_EQ(report["customer_data"], data_path_);
  EXPECT_EQ(report["nation_data"], data_path_);
  EXPECT_EQ(report["supplier_data"], data_path_);
  EXPECT_EQ(report["region_data"], data_path_);
  EXPECT_EQ(report["partsupp_data"], data_path_);
}

// mode=="cold" exercises evict_from_page_cache()'s POSIX_FADV_DONTNEED path
// (best-effort, no assertion on its actual effect -- see the source's own
// comment on why this can only ever be a hint) -- no prior test used cold
// mode at all.
TEST_F(BenchmarkTpchCommandRealRunTest, ColdModeEvictsPageCacheAndStillProducesAReport) {
  const std::vector<std::string_view> args = {"--data",   data_path_,  "--query-file", query_file_path_,
                                              "--query",  "1",         "--mode",       "cold",
                                              "--output", output_path_};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 0);
}

// >=2 iterations exercises median_of()'s even-count averaging branch and
// stddev_of()'s real (non-degenerate, size>=2) computation -- the existing
// single-iteration test only ever hits their size==1 shortcuts. Non-zero
// warmup_iterations also exercises the warmup loop body itself.
TEST_F(BenchmarkTpchCommandRealRunTest, MultipleIterationsWithWarmupComputeRealStatistics) {
  const std::vector<std::string_view> args = {"--data",
                                              data_path_,
                                              "--query-file",
                                              query_file_path_,
                                              "--query",
                                              "1",
                                              "--mode",
                                              "warm",
                                              "--iterations",
                                              "2",
                                              "--warmup-iterations",
                                              "1",
                                              "--output",
                                              output_path_};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 0);

  std::ifstream report_file(output_path_);
  std::ostringstream contents;
  contents << report_file.rdbuf();
  const nlohmann::json report = nlohmann::json::parse(contents.str());
  EXPECT_EQ(report["iteration_results"].size(), 2u);
  EXPECT_EQ(report["warmup_iterations"], 1);
}

// No --output writes the JSON report to stdout instead of a file (see
// write_query_result-adjacent std::printf branch) -- no prior test omitted
// --output. stdout isn't captured/asserted on here (same limitation
// query_command_test.cpp's own --stats test notes for stderr), only that
// the run still succeeds.
TEST_F(BenchmarkTpchCommandRealRunTest, WritesReportToStdoutWhenNoOutputPathGiven) {
  const std::vector<std::string_view> args = {"--data",  data_path_, "--query-file", query_file_path_,
                                              "--query", "1",        "--mode",       "warm"};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 0);
}

TEST_F(BenchmarkTpchCommandRealRunTest, ThrowsCleanErrorWhenOutputPathIsUnwritable) {
  const std::string bad_output = (dir_ / "does-not-exist" / "report.json").string();
  const std::vector<std::string_view> args = {"--data",   data_path_, "--query-file", query_file_path_,
                                              "--query",  "1",        "--mode",       "warm",
                                              "--output", bad_output};
  EXPECT_EQ(run_benchmark_tpch(args, config_), 1);
}

// End-to-end: valid --scale-factor/--query/--iterations/--warmup-iterations
// values (the same arguments the tests above feed malformed) parse and
// drive a real query run, closing the loop on the parsing fix rather than
// only exercising its failure path.
TEST_F(BenchmarkTpchCommandRealRunTest, ValidNumericArgumentsProduceARealBenchmarkReport) {
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
  EXPECT_EQ(run_benchmark_tpch(args, config_), 0);

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
