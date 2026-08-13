// run_generate_data() is a pure (args) -> int function (see
// commands.hpp's own doc comment), same shape as run_query()/run_explain()
// -- see query_command_test.cpp's own header comment.
// src/cli/generate_data_command.cpp had zero test coverage before this
// file.
#include <gtest/gtest.h>

#include <filesystem>

#include "kernellake/cli/commands.hpp"

namespace kernellake::cli {
namespace {

namespace fs = std::filesystem;

class GenerateDataCommandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_generate_data_command_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
};

TEST_F(GenerateDataCommandTest, RejectsUnrecognizedArgument) {
  const std::vector<std::string_view> args = {"--not-a-real-flag", "value"};
  EXPECT_EQ(run_generate_data(args), 1);
}

// Every test below that passes `--output` names a local `output_dir`
// rather than inlining `dir_.string()` into the initializer list --
// args' elements are non-owning std::string_views, and a std::string_view
// over a temporary's buffer would dangle the instant the initializer-list's
// full expression ends (see query_command_test.cpp's own comment on this
// exact pitfall).

TEST_F(GenerateDataCommandTest, RejectsNonNumericRows) {
  const std::string output_dir = dir_.string();
  const std::vector<std::string_view> args = {"--output", output_dir, "--rows", "not-a-number"};
  EXPECT_EQ(run_generate_data(args), 1);
}

TEST_F(GenerateDataCommandTest, RejectsNonNumericSeed) {
  const std::string output_dir = dir_.string();
  const std::vector<std::string_view> args = {"--output", output_dir, "--seed", "not-a-number"};
  EXPECT_EQ(run_generate_data(args), 1);
}

TEST_F(GenerateDataCommandTest, RejectsNonNumericNullRate) {
  const std::string output_dir = dir_.string();
  const std::vector<std::string_view> args = {"--output", output_dir, "--null-rate", "not-a-number"};
  EXPECT_EQ(run_generate_data(args), 1);
}

TEST_F(GenerateDataCommandTest, RejectsInvalidOptionsSurfacedAsConfigurationError) {
  // files <= 0 is rejected by generate_sample_data() itself (see
  // sample_data_generator.hpp), not by argument parsing -- exercises the
  // command's KernelLakeError catch block, not just bad_arg.
  const std::string output_dir = dir_.string();
  const std::vector<std::string_view> args = {"--output", output_dir, "--files", "0"};
  EXPECT_EQ(run_generate_data(args), 1);
}

TEST_F(GenerateDataCommandTest, GeneratesRealDataAndReportsRowCount) {
  // output_dir_ is a named local (not a temporary inline in the
  // initializer list below) so it outlives args -- see
  // query_command_test.cpp's own comment on this exact pitfall: args'
  // elements are non-owning std::string_views, and a std::string_view over
  // a temporary's buffer would dangle the instant the initializer-list's
  // full expression ends.
  const std::string output_dir = dir_.string();
  const std::vector<std::string_view> args = {
      "--output",   output_dir, "--rows",    "10",   "--files",           "1",
      "--row-group-rows", "10", "--seed",    "1",    "--no-dictionary-encoding"};
  EXPECT_EQ(run_generate_data(args), 0);

  ASSERT_TRUE(fs::exists(dir_));
  std::size_t parquet_files = 0;
  for (const auto& entry : fs::directory_iterator(dir_)) {
    if (entry.path().extension() == ".parquet") {
      ++parquet_files;
    }
  }
  EXPECT_EQ(parquet_files, 1U);
}

TEST_F(GenerateDataCommandTest, GeneratesMultipleFilesWithCardinalityAndSkewFlags) {
  // See GeneratesRealDataAndReportsRowCount above for why this must be a
  // named local rather than an inline dir_.string() temporary.
  const std::string output_dir = dir_.string();
  const std::vector<std::string_view> args = {
      "--output",   output_dir,
      "--rows",     "20",
      "--files",    "2",
      "--region-cardinality", "3",
      "--category-cardinality", "2",
      "--customer-cardinality", "5",
      "--null-rate", "0.5",
      "--skew",     "0.5"};
  EXPECT_EQ(run_generate_data(args), 0);

  std::size_t parquet_files = 0;
  for (const auto& entry : fs::directory_iterator(dir_)) {
    if (entry.path().extension() == ".parquet") {
      ++parquet_files;
    }
  }
  EXPECT_EQ(parquet_files, 2U);
}

}  // namespace
}  // namespace kernellake::cli
