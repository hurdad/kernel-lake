// run_explain() is a pure (args, config) -> int function (see
// commands.hpp's own doc comment), same shape as run_query() -- see
// query_command_test.cpp's own header comment. src/cli/explain_command.cpp
// had zero test coverage before this file.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/cli/commands.hpp"
#include "kernellake/common/config.hpp"

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

    arrow::Int64Builder id_builder;
    for (std::int64_t i = 0; i < 3; ++i) {
      ASSERT_TRUE(id_builder.Append(i).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
    const auto table = arrow::Table::Make(schema, {id_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/3);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
  EngineConfig config_ = cpu_backend_config();

 private:
  [[nodiscard]] static EngineConfig cpu_backend_config() {
    EngineConfig config = default_config();
    config.engine.backend = "cpu";
    return config;
  }
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
