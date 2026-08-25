// run_explain() is a pure (args, config) -> int function (see
// commands.hpp's own doc comment), same shape as run_query() -- see
// query_command_test.cpp's own header comment. src/cli/explain_command.cpp
// had zero test coverage before this file.
#include <gtest/gtest.h>

#include <filesystem>

#include "kernellake/cli/commands.hpp"
#include "kernellake/common/config.hpp"

#include "cli_command_test_support.hpp"

namespace kernellake::cli {
namespace {

namespace fs = std::filesystem;

class ExplainCommandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_explain_command_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();
    write_id_column_parquet(path_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
  CliConfig config_ = cpu_backend_config();
};

TEST_F(ExplainCommandTest, RejectsMissingSql) {
  const std::vector<std::string_view> args = {"--format", "text"};
  EXPECT_EQ(run_explain(args, config_), 1);
}

TEST_F(ExplainCommandTest, RejectsInvalidFormatValue) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql, "--format", "yaml"};
  EXPECT_EQ(run_explain(args, config_), 1);
}

TEST_F(ExplainCommandTest, RejectsMalformedSqlWithCleanErrorNotACrash) {
  const std::vector<std::string_view> args = {"--sql", "SELECT this is not valid sql"};
  EXPECT_EQ(run_explain(args, config_), 1);
}

TEST_F(ExplainCommandTest, ExplainsPhysicalPlanAsText) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql};
  EXPECT_EQ(run_explain(args, config_), 0);
}

TEST_F(ExplainCommandTest, ExplainsPhysicalPlanAsJson) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql, "--format", "json"};
  EXPECT_EQ(run_explain(args, config_), 0);
}

TEST_F(ExplainCommandTest, ExplainsLogicalPlanWhenRequested) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql, "--logical"};
  EXPECT_EQ(run_explain(args, config_), 0);
}

TEST_F(ExplainCommandTest, ExplainsLogicalPlanAsJson) {
  const std::string sql = "SELECT id FROM read_parquet('" + path_ + "')";
  const std::vector<std::string_view> args = {"--sql", sql, "--logical", "--format", "json"};
  EXPECT_EQ(run_explain(args, config_), 0);
}

}  // namespace
}  // namespace kernellake::cli
