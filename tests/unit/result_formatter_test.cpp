// src/cli/ had zero test coverage before this file -- see
// src/cli/CMakeLists.txt's kernellake_cli static library, split out from the
// kernellake executable specifically so this (and the other src/cli/
// *_test.cpp files) can link against the real command/formatter logic
// without dragging in main().
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "kernellake/cli/result_formatter.hpp"

namespace kernellake::cli {
namespace {

namespace fs = std::filesystem;

class ResultFormatterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_result_formatter_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "out").string();
  }

  void TearDown() override { fs::remove_all(dir_); }

  [[nodiscard]] std::string read_output() const {
    std::ifstream stream(path_, std::ios::binary);
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
  }

  fs::path dir_;
  std::string path_;
};

// CRITICAL bug fix regression test: json_escape() (result_formatter.cpp)
// used to only escape `" \ \n \r \t`, leaving every other C0 control
// character (0x00-0x1F) -- embedded NUL, backspace, form feed, etc. -- to
// pass through raw, producing invalid JSON in --format jsonl output. This
// builds a STRING value containing exactly those three unescaped-before
// bytes and checks both the byte-for-byte expected escaped form and that
// the resulting line is valid, parseable JSON.
TEST_F(ResultFormatterTest, JsonLinesEscapesEveryC0ControlCharacter) {
  const auto schema = arrow::schema({arrow::field("s", arrow::utf8(), false)});
  arrow::StringBuilder builder;
  const std::string raw_value = std::string("a") + '\x00' + "b" + '\x08' + "c" + '\x0C' + "d";
  ASSERT_TRUE(builder.Append(raw_value).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::JsonLines, path_);
  const std::string output = read_output();

  EXPECT_EQ(output, "{\"s\":\"a\\u0000b\\bc\\fd\"}\n");

  const nlohmann::json parsed = nlohmann::json::parse(output);
  EXPECT_EQ(parsed["s"].get<std::string>(), raw_value);
}

TEST_F(ResultFormatterTest, JsonLinesEscapesQuoteBackslashAndWhitespaceControls) {
  const auto schema = arrow::schema({arrow::field("s", arrow::utf8(), false)});
  arrow::StringBuilder builder;
  ASSERT_TRUE(builder.Append(std::string("a\"b\\c\nd\re\tf")).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::JsonLines, path_);
  const std::string output = read_output();

  EXPECT_EQ(output, "{\"s\":\"a\\\"b\\\\c\\nd\\re\\tf\"}\n");
  EXPECT_NO_THROW({ const nlohmann::json parsed = nlohmann::json::parse(output); });
}

TEST_F(ResultFormatterTest, JsonLinesRendersNullAndNumericColumnsUnquoted) {
  const auto schema =
      arrow::schema({arrow::field("n", arrow::int64(), true), arrow::field("s", arrow::utf8(), true)});
  arrow::Int64Builder n_builder;
  arrow::StringBuilder s_builder;
  ASSERT_TRUE(n_builder.AppendNull().ok());
  ASSERT_TRUE(s_builder.Append("x").ok());
  std::shared_ptr<arrow::Array> n_array, s_array;
  ASSERT_TRUE(n_builder.Finish(&n_array).ok());
  ASSERT_TRUE(s_builder.Finish(&s_array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {n_array, s_array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::JsonLines, path_);
  EXPECT_EQ(read_output(), "{\"n\":null,\"s\":\"x\"}\n");
}

// MODERATE gap: scalar_text() (result_formatter.cpp) renders every value via
// arrow::Scalar::ToString(), with no KernelLake-side formatting of its own,
// for DATE32/DECIMAL128/TIMESTAMP. These lock in Arrow's current output
// format for each (verified directly against a real arrow::Scalar::ToString()
// call before being hardcoded here) so a future Arrow upgrade that silently
// changes that formatting gets caught by this suite instead of only
// showing up as a surprise in real --format jsonl/table output.
TEST_F(ResultFormatterTest, JsonLinesLocksInDate32ScalarTextFormat) {
  const auto schema = arrow::schema({arrow::field("d", arrow::date32(), false)});
  arrow::Date32Builder builder;
  ASSERT_TRUE(builder.Append(19723).ok());  // 2024-01-01, days since epoch.
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::JsonLines, path_);
  EXPECT_EQ(read_output(), "{\"d\":\"2024-01-01\"}\n");
}

TEST_F(ResultFormatterTest, JsonLinesLocksInDecimal128ScalarTextFormat) {
  const auto type = arrow::decimal128(10, 2);
  const auto schema = arrow::schema({arrow::field("amt", type, false)});
  arrow::Decimal128Builder builder(type);
  const arrow::Result<arrow::Decimal128> value = arrow::Decimal128::FromString("12345.67");
  ASSERT_TRUE(value.ok()) << value.status().ToString();
  ASSERT_TRUE(builder.Append(*value).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::JsonLines, path_);
  EXPECT_EQ(read_output(), "{\"amt\":\"12345.67\"}\n");
}

TEST_F(ResultFormatterTest, JsonLinesLocksInTimestampScalarTextFormat) {
  // kernellake/types/arrow_adapter.cpp maps its TIMESTAMP type to exactly
  // arrow::timestamp(arrow::TimeUnit::MICRO) with no timezone -- matching
  // that here, not an arbitrary unit, since ToString()'s rendering depends
  // on both.
  const auto type = arrow::timestamp(arrow::TimeUnit::MICRO);
  const auto schema = arrow::schema({arrow::field("ts", type, false)});
  arrow::TimestampBuilder builder(type, arrow::default_memory_pool());
  ASSERT_TRUE(builder.Append(1704067200000000LL).ok());  // 2024-01-01 00:00:00 UTC.
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::JsonLines, path_);
  EXPECT_EQ(read_output(), "{\"ts\":\"2024-01-01 00:00:00.000000\"}\n");
}

TEST(ParseResultFormat, RecognizesEveryDocumentedFormatName) {
  EXPECT_EQ(parse_result_format("table"), ResultFormat::Table);
  EXPECT_EQ(parse_result_format("csv"), ResultFormat::Csv);
  EXPECT_EQ(parse_result_format("jsonl"), ResultFormat::JsonLines);
  EXPECT_EQ(parse_result_format("arrow"), ResultFormat::ArrowIpc);
}

TEST(ParseResultFormat, RejectsUnknownFormatName) {
  EXPECT_EQ(parse_result_format("xml"), std::nullopt);
}

}  // namespace
}  // namespace kernellake::cli
