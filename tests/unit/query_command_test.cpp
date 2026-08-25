// run_query() is a pure (args, config) -> int function (see commands.hpp's
// own doc comment) reachable without spinning up a process, so this drives
// it directly rather than shelling out to the kernellake binary --
// src/cli/ had zero test coverage before this file (and
// result_formatter_test.cpp/inspect_parquet_command_test.cpp/
// benchmark_tpch_command_test.cpp alongside it).
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "kernellake/cli/commands.hpp"
#include "kernellake/common/config.hpp"

#include "cli_command_test_support.hpp"

namespace kernellake::cli {
namespace {

namespace fs = std::filesystem;

class QueryCommandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_command_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();
    output_path_ = (dir_ / "out").string();
    write_id_column_parquet(path_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  [[nodiscard]] std::string read_output() const {
    std::ifstream stream(output_path_);
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
  }

  fs::path dir_;
  std::string path_;
  std::string output_path_;
  CliConfig config_ = cpu_backend_config();
};

TEST_F(QueryCommandTest, RejectsSqlAndFileTogetherAsMutuallyExclusive) {
  const std::vector<std::string_view> args = {"--sql", "SELECT 1", "--file", "/nonexistent.sql"};
  EXPECT_EQ(run_query(args, config_), 1);
}

TEST_F(QueryCommandTest, RejectsNeitherSqlNorFile) {
  const std::vector<std::string_view> args = {"--format", "table"};
  EXPECT_EQ(run_query(args, config_), 1);
}

TEST_F(QueryCommandTest, RejectsInvalidFormatValue) {
  // sql is a named local (not a temporary inline in the initializer list
  // below) so it outlives args -- args' elements are non-owning
  // std::string_views, and a std::string_view over a temporary's buffer
  // would dangle the instant the initializer-list's full expression ends.
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql, "--format", "yaml"};
  EXPECT_EQ(run_query(args, config_), 1);
}

TEST_F(QueryCommandTest, RejectsInvalidBackendValue) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql, "--backend", "tpu"};
  EXPECT_EQ(run_query(args, config_), 1);
}

TEST_F(QueryCommandTest, RejectsMalformedSqlWithCleanErrorNotACrash) {
  const std::vector<std::string_view> args = {"--sql", "SELECT this is not valid sql"};
  EXPECT_EQ(run_query(args, config_), 1);
}

TEST_F(QueryCommandTest, RunsRealQueryAndWritesJsonlOutput) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql, "--format", "jsonl", "--output", output_path_};
  EXPECT_EQ(run_query(args, config_), 0);
  const std::string output = read_output();
  EXPECT_NE(output.find("{\"id\":0}"), std::string::npos);
  EXPECT_NE(output.find("{\"id\":1}"), std::string::npos);
  EXPECT_NE(output.find("{\"id\":2}"), std::string::npos);
}

TEST_F(QueryCommandTest, RejectsUnreadableSqlFile) {
  // Named local, not an inline temporary -- see RunsRealQueryAndWritesJsonlOutput's own
  // ReadsSqlFromFileArgument-adjacent comment on this file's dangling-std::string_view pitfall.
  const std::string missing_file = (dir_ / "does-not-exist.sql").string();
  const std::vector<std::string_view> args = {"--file", missing_file};
  EXPECT_EQ(run_query(args, config_), 1);
}

// print_stats()/print_optional() (query_command.cpp) were entirely
// uncovered -- no prior test passed --stats. They write to stderr, which
// (unlike the query result itself) has no --output-style redirect this
// command exposes, so this only asserts the run completes successfully
// rather than capturing/asserting on the printed text.
TEST_F(QueryCommandTest, RunsWithStatsFlagWithoutError) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql",    sql,          "--format", "jsonl",
                                              "--output", output_path_, "--stats"};
  EXPECT_EQ(run_query(args, config_), 0);
}

// Exercises the effective_config.engine.backend assignment (only reached
// with a *valid* override) -- RejectsInvalidBackendValue above only covers
// the rejection path.
TEST_F(QueryCommandTest, AcceptsExplicitCpuBackendOverride) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql",    sql,     "--backend", "cpu",
                                              "--format", "jsonl", "--output",  output_path_};
  EXPECT_EQ(run_query(args, config_), 0);
  EXPECT_NE(read_output().find("{\"id\":0}"), std::string::npos);
}

TEST_F(QueryCommandTest, ReadsSqlFromFileArgument) {
  const std::string sql_path = (dir_ / "query.sql").string();
  std::ofstream sql_file(sql_path);
  sql_file << "SELECT id FROM read_parquet('" << path_ << "')";
  sql_file.close();

  const std::vector<std::string_view> args = {"--file", sql_path,   "--format",
                                              "jsonl",  "--output", output_path_};
  EXPECT_EQ(run_query(args, config_), 0);
  EXPECT_NE(read_output().find("{\"id\":0}"), std::string::npos);
}

}  // namespace
}  // namespace kernellake::cli
