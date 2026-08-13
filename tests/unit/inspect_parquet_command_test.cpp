// run_inspect_parquet() prints to stdout directly (no --output flag exists
// for this command -- see commands.hpp) rather than returning a string, so
// this uses gtest's own testing::internal::CaptureStdout()/
// GetCapturedStdout() to observe it, the standard way to test
// stdout-writing code without extra plumbing.
//
// MODERATE gap: literal_to_string() (inspect_parquet_command.cpp:17-31)
// formats doubles with a fixed 6 decimal places (std::to_string's own
// default) and, in --format json, wraps every LiteralStorage variant --
// including INT64 and DOUBLE, not just STRING -- in a JSON *string* rather
// than a JSON number, since to_json() assigns literal_to_string()'s
// std::string return value straight into the nlohmann::json field
// regardless of the underlying column type. These tests lock in that
// current (imperfect but documented) behavior rather than fixing it --
// fixing it is optional/lower priority per this test's own scope, and
// changing to_json()'s number-vs-string typing would be a a real (if
// arguably corrective) behavior change to every JSON consumer of this
// command's output, not just a bug fix invisible to callers.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include <filesystem>

#include "kernellake/cli/commands.hpp"
#include "kernellake/common/config.hpp"

namespace kernellake::cli {
namespace {

namespace fs = std::filesystem;

class InspectParquetCommandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_inspect_parquet_command_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();

    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::StringBuilder region_builder;
    for (std::int64_t i = 0; i < 5; ++i) {
      ASSERT_TRUE(id_builder.Append(i).ok());
      ASSERT_TRUE(amount_builder.Append(1.5 + static_cast<double>(i)).ok());
      ASSERT_TRUE(region_builder.Append(i == 0 ? "A" : "B").ok());
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

TEST_F(InspectParquetCommandTest, RejectsMissingPath) {
  const std::vector<std::string_view> args = {"--format", "text"};
  EXPECT_EQ(run_inspect_parquet(args, default_config()), 1);
}

TEST_F(InspectParquetCommandTest, RejectsInvalidFormat) {
  const std::vector<std::string_view> args = {"--path", path_, "--format", "yaml"};
  EXPECT_EQ(run_inspect_parquet(args, default_config()), 1);
}

TEST_F(InspectParquetCommandTest, TextFormatLocksInFixedSixDecimalDoubleFormatting) {
  const std::vector<std::string_view> args = {"--path", path_, "--format", "text"};
  testing::internal::CaptureStdout();
  const int rc = run_inspect_parquet(args, default_config());
  const std::string output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(rc, 0);

  // amount's row-group min/max are 1.5 and 5.5 -- std::to_string(double)
  // always renders exactly 6 decimal places, not the shortest
  // round-trippable form, so "1.500000"/"5.500000", not "1.5"/"5.5".
  EXPECT_NE(output.find("min=1.500000 max=5.500000"), std::string::npos) << output;
  // id (INT64) has no decimal point at all -- literal_to_string()'s
  // std::to_string(std::int64_t) branch, not the DOUBLE one.
  EXPECT_NE(output.find("min=0 max=4"), std::string::npos) << output;
  // region (STRING) is wrapped in single quotes by literal_to_string()'s
  // own STRING branch.
  EXPECT_NE(output.find("min='A' max='B'"), std::string::npos) << output;
}

TEST_F(InspectParquetCommandTest, JsonFormatLocksInStringTypedMinMaxForEveryColumnType) {
  const std::vector<std::string_view> args = {"--path", path_, "--format", "json"};
  testing::internal::CaptureStdout();
  const int rc = run_inspect_parquet(args, default_config());
  const std::string output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(rc, 0);

  const nlohmann::json parsed = nlohmann::json::parse(output);
  ASSERT_EQ(parsed.size(), 1u);
  const nlohmann::json& columns = parsed[0]["row_groups"][0]["columns"];

  // The documented inconsistency: id/amount are numeric Parquet columns,
  // but to_json() still emits their min/max as JSON *strings* (via
  // literal_to_string()), not JSON numbers -- only region (already a
  // string column) "accidentally" has the right JSON type.
  ASSERT_TRUE(columns["id"]["min"].is_string());
  EXPECT_EQ(columns["id"]["min"].get<std::string>(), "0");
  EXPECT_EQ(columns["id"]["max"].get<std::string>(), "4");

  ASSERT_TRUE(columns["amount"]["min"].is_string());
  EXPECT_EQ(columns["amount"]["min"].get<std::string>(), "1.500000");
  EXPECT_EQ(columns["amount"]["max"].get<std::string>(), "5.500000");

  ASSERT_TRUE(columns["region"]["min"].is_string());
  // STRING values additionally carry literal_to_string()'s own single
  // quotes *inside* the JSON string -- "'A'", not "A".
  EXPECT_EQ(columns["region"]["min"].get<std::string>(), "'A'");
  EXPECT_EQ(columns["region"]["max"].get<std::string>(), "'B'");
}

}  // namespace
}  // namespace kernellake::cli
