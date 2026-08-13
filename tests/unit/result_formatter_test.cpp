// src/cli/ had zero test coverage before this file -- see
// src/cli/CMakeLists.txt's kernellake_cli static library, split out from the
// kernellake executable specifically so this (and the other src/cli/
// *_test.cpp files) can link against the real command/formatter logic
// without dragging in main().
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "kernellake/common/errors.hpp"
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

// write_table_format() was entirely uncovered -- every prior test in this
// file exercised ResultFormat::JsonLines only. Column widths must span both
// the header and the widest cell in that column ("bb" widens "s" past its
// own 1-char header), locked in byte-for-byte including the
// dashes-then-two-spaces separator row.
TEST_F(ResultFormatterTest, TableFormatAlignsColumnsToWidestCellOrHeader) {
  const auto schema =
      arrow::schema({arrow::field("n", arrow::int64(), false), arrow::field("s", arrow::utf8(), false)});
  arrow::Int64Builder n_builder;
  arrow::StringBuilder s_builder;
  ASSERT_TRUE(n_builder.Append(1).ok());
  ASSERT_TRUE(n_builder.Append(2).ok());
  ASSERT_TRUE(s_builder.Append("a").ok());
  ASSERT_TRUE(s_builder.Append("bb").ok());
  std::shared_ptr<arrow::Array> n_array, s_array;
  ASSERT_TRUE(n_builder.Finish(&n_array).ok());
  ASSERT_TRUE(s_builder.Finish(&s_array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 2, {n_array, s_array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::Table, path_);
  EXPECT_EQ(read_output(), "n  s   \n-  --  \n1  a   \n2  bb  \n");
}

TEST_F(ResultFormatterTest, TableFormatRendersNullAsLiteralText) {
  const auto schema = arrow::schema({arrow::field("n", arrow::int64(), true)});
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendNull().ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::Table, path_);
  EXPECT_EQ(read_output(), "n     \n----  \nNULL  \n");
}

// write_csv_format() was entirely uncovered. Arrow's WriteOptions::Defaults()
// uses "\n" line endings, writes a header row, and (QuotingStyle::Needed)
// always quotes string-typed headers/values but never numeric ones --
// verified directly against a real WriteCSV() call before being hardcoded
// here, same "locks in" approach as scalar_text()'s tests above.
TEST_F(ResultFormatterTest, CsvFormatWritesHeaderAndRows) {
  const auto schema =
      arrow::schema({arrow::field("n", arrow::int64(), false), arrow::field("s", arrow::utf8(), false)});
  arrow::Int64Builder n_builder;
  arrow::StringBuilder s_builder;
  ASSERT_TRUE(n_builder.Append(1).ok());
  ASSERT_TRUE(n_builder.Append(2).ok());
  ASSERT_TRUE(s_builder.Append("a").ok());
  ASSERT_TRUE(s_builder.Append("b").ok());
  std::shared_ptr<arrow::Array> n_array, s_array;
  ASSERT_TRUE(n_builder.Finish(&n_array).ok());
  ASSERT_TRUE(s_builder.Finish(&s_array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 2, {n_array, s_array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::Csv, path_);
  EXPECT_EQ(read_output(), "\"n\",\"s\"\n1,\"a\"\n2,\"b\"\n");
}

TEST_F(ResultFormatterTest, CsvFormatQuotesValuesContainingDelimiter) {
  const auto schema = arrow::schema({arrow::field("s", arrow::utf8(), false)});
  arrow::StringBuilder builder;
  ASSERT_TRUE(builder.Append("a,b").ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 1, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::Csv, path_);
  EXPECT_EQ(read_output(), "\"s\"\n\"a,b\"\n");
}

// write_arrow_ipc_format() was entirely uncovered. Round-trips the written
// file back through Arrow's own IPC reader rather than asserting on raw
// bytes (the IPC container format isn't meant to be hand-verified).
TEST_F(ResultFormatterTest, ArrowIpcFormatRoundTripsThroughRealReader) {
  const auto schema = arrow::schema({arrow::field("n", arrow::int64(), false)});
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.Append(7).ok());
  ASSERT_TRUE(builder.Append(9).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto batch = arrow::RecordBatch::Make(schema, 2, {array});

  QueryResult result;
  result.schema = schema;
  result.batches = {batch};

  write_query_result(result, ResultFormat::ArrowIpc, path_);

  auto input = arrow::io::ReadableFile::Open(path_);
  ASSERT_TRUE(input.ok()) << input.status().ToString();
  auto reader = arrow::ipc::RecordBatchFileReader::Open(*input);
  ASSERT_TRUE(reader.ok()) << reader.status().ToString();
  ASSERT_EQ((*reader)->num_record_batches(), 1);
  auto read_batch = (*reader)->ReadRecordBatch(0);
  ASSERT_TRUE(read_batch.ok()) << read_batch.status().ToString();
  ASSERT_TRUE((*read_batch)->schema()->Equals(*schema));
  ASSERT_EQ((*read_batch)->num_rows(), 2);
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>((*read_batch)->column(0))->Value(0), 7);
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>((*read_batch)->column(0))->Value(1), 9);
}

// Every format funnels through open_binary_sink()/fopen() for its output
// file, and an unwritable path (a directory that doesn't exist, so the
// underlying open(2) fails) is the one error path none of the format
// functions handle specially -- exercises each one's own
// arrow_status_message()-wrapped ExecutionError.
TEST_F(ResultFormatterTest, TableFormatThrowsOnUnopenableOutputPath) {
  const std::string bad_path = (dir_ / "does-not-exist" / "out").string();
  QueryResult result;
  result.schema = arrow::schema({arrow::field("n", arrow::int64(), false)});
  EXPECT_THROW(write_query_result(result, ResultFormat::Table, bad_path), ExecutionError);
}

TEST_F(ResultFormatterTest, CsvFormatThrowsOnUnopenableOutputPath) {
  const std::string bad_path = (dir_ / "does-not-exist" / "out").string();
  QueryResult result;
  result.schema = arrow::schema({arrow::field("n", arrow::int64(), false)});
  EXPECT_THROW(write_query_result(result, ResultFormat::Csv, bad_path), ExecutionError);
}

TEST_F(ResultFormatterTest, ArrowIpcFormatThrowsOnUnopenableOutputPath) {
  const std::string bad_path = (dir_ / "does-not-exist" / "out").string();
  QueryResult result;
  result.schema = arrow::schema({arrow::field("n", arrow::int64(), false)});
  EXPECT_THROW(write_query_result(result, ResultFormat::ArrowIpc, bad_path), ExecutionError);
}

}  // namespace
}  // namespace kernellake::cli
