#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/scalar/scalar.hpp>

#include <filesystem>

#include "kernellake/execution_gpu/filter_operator.hpp"
#include "kernellake/execution_gpu/parquet_scan_operator.hpp"
#include "kernellake/execution_gpu/projection_operator.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

// Test double: yields a fixed, pre-built sequence of batches, one per call
// to next(), then reports exhausted.
class VectorSourceOperator final : public PhysicalOperator {
 public:
  explicit VectorSourceOperator(std::vector<DeviceBatch> batches) : batches_(std::move(batches)) {}

  void open(ExecutionContext&) override { index_ = 0; }
  std::optional<DeviceBatch> next(ExecutionContext&) override {
    if (index_ >= batches_.size()) return std::nullopt;
    return std::move(batches_[index_++]);
  }
  void close(ExecutionContext&) override {}
  [[nodiscard]] std::string_view name() const noexcept override { return "VectorSource"; }
  [[nodiscard]] OperatorId id() const noexcept override { return 0; }

 private:
  std::vector<DeviceBatch> batches_;
  std::size_t index_ = 0;
};

Schema one_int_column_schema() {
  return Schema({Field{"a", int32_type(false)}});
}

DeviceBatch make_filled_batch(int32_t fill_value, cudf::size_type num_rows) {
  std::unique_ptr<cudf::column> column =
      cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  std::unique_ptr<cudf::scalar> value = cudf::make_fixed_width_scalar<int32_t>(fill_value);
  cudf::mutable_column_view view = column->mutable_view();
  cudf::fill_in_place(view, 0, num_rows, *value);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                     std::make_shared<const Schema>(one_int_column_schema()));
}

ExecutionContext make_context() {
  return ExecutionContext{"test-query", 0,       nullptr, rmm::mr::get_current_device_resource_ref(),
                          nullptr,      nullptr, nullptr};
}

// Builds a fixed-width column directly from host data via a raw device
// buffer -- works for any fixed-width type_id (including TIMESTAMP_DAYS,
// which the EXTRACT test below needs), unlike cudf::make_numeric_column
// which only accepts numeric types. Mirrors sort_operator_test.cpp's own
// identically-named helper.
template <typename T>
std::unique_ptr<cudf::column> column_from_host(const std::vector<T>& values, cudf::type_id type) {
  rmm::device_buffer data(values.size() * sizeof(T), cudf::get_default_stream());
  cudaMemcpy(data.data(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
  return std::make_unique<cudf::column>(cudf::data_type{type}, static_cast<cudf::size_type>(values.size()),
                                        std::move(data), rmm::device_buffer{}, 0);
}

TEST(FilterOperator, PassesMatchingBatchesAndSkipsEmptyOnes) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(10, 5));  // all rows pass a > 3
  batches.push_back(make_filled_batch(1, 5));   // all rows fail a > 3 -> skipped entirely
  batches.push_back(make_filled_batch(20, 3));  // all rows pass

  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto three = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(3));
  auto a_i64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto predicate =
      std::make_shared<BinaryExpression>(BinaryOperator::Greater, a_i64, three, boolean_type(false));

  FilterOperator filter(1, std::make_unique<VectorSourceOperator>(std::move(batches)), predicate);
  ExecutionContext context = make_context();
  filter.open(context);

  std::optional<DeviceBatch> first = filter.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 5u);

  std::optional<DeviceBatch> second = filter.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 3u);  // the all-failing batch was skipped

  EXPECT_FALSE(filter.next(context).has_value());
  filter.close(context);
}

TEST(ProjectionOperator, EvaluatesArithmeticAcrossMultipleBatches) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(5, 4));
  batches.push_back(make_filled_batch(7, 2));

  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto ten = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(10));
  auto a_i64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto doubled = std::make_shared<BinaryExpression>(BinaryOperator::Add, a_i64, ten, int64_type(false));

  std::vector<NamedExpression> items = {NamedExpression{doubled, "result"}};
  ProjectionOperator projection(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                                std::move(items));
  ExecutionContext context = make_context();
  projection.open(context);

  std::optional<DeviceBatch> first = projection.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 4u);
  EXPECT_EQ(first->schema().field(0).name, "result");

  std::optional<DeviceBatch> second = projection.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 2u);

  EXPECT_FALSE(projection.next(context).has_value());
  projection.close(context);
}

// Regression coverage: compile_value()'s plain-literal fast path and
// materialize_value()'s literal_scalar branch had no dedicated test --
// EvaluatesArithmeticAcrossMultipleBatches above only exercises a literal
// as a *sub-expression* inside a BinaryExpression (compiled via
// cudf::ast), never a literal as an entire top-level projected item (e.g.
// `SELECT 42 AS constant_col, a FROM ...`), which takes the
// cudf::make_column_from_scalar fast path directly instead.
TEST(ProjectionOperator, EvaluatesLiteralProjectionAlongsideColumn) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(1, 3));
  batches.push_back(make_filled_batch(2, 2));

  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto constant = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(42));

  std::vector<NamedExpression> items = {NamedExpression{constant, "constant_col"}, NamedExpression{a, "a"}};
  ProjectionOperator projection(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                                std::move(items));
  ExecutionContext context = make_context();
  projection.open(context);

  std::optional<DeviceBatch> first = projection.next(context);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->row_count(), 3u);
  EXPECT_EQ(first->schema().field(0).name, "constant_col");
  const cudf::column_view constant_view = first->view().column(0);
  for (cudf::size_type i = 0; i < constant_view.size(); ++i) {
    const std::unique_ptr<cudf::scalar> element = cudf::get_element(constant_view, i);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int64_t>&>(*element).value(), 42);
  }

  std::optional<DeviceBatch> second = projection.next(context);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->row_count(), 2u);

  EXPECT_FALSE(projection.next(context).has_value());
  projection.close(context);
}

// Regression coverage: compile_item()'s CASE-expression branch and
// materialize_case() had zero coverage through ProjectionOperator --
// CASE expressions were previously only ever tested as a GROUP BY key or
// aggregate argument (HashAggregateOperator/ScalarAggregateOperator's own
// copies of this same fold-from-the-last-branch-backward algorithm), never
// as a plain SELECT-list item (e.g. `SELECT CASE WHEN a > 5 THEN 100 ELSE
// 200 END FROM ...`), which is ProjectionOperator's own code path.
TEST(ProjectionOperator, EvaluatesCaseExpressionAcrossBatches) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(10, 3));  // 10 > 5 -> THEN branch
  batches.push_back(make_filled_batch(1, 2));   // 1 <= 5 -> ELSE branch

  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto five = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(5));
  auto a_i64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto condition =
      std::make_shared<BinaryExpression>(BinaryOperator::Greater, a_i64, five, boolean_type(false));
  auto then_value = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(100));
  auto else_value = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(200));
  std::vector<CaseExpression::WhenThen> when_then = {CaseExpression::WhenThen{condition, then_value}};
  auto case_expr = std::make_shared<CaseExpression>(std::move(when_then), else_value, int64_type(false));

  std::vector<NamedExpression> items = {NamedExpression{case_expr, "bucket"}};
  ProjectionOperator projection(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                                std::move(items));
  ExecutionContext context = make_context();
  projection.open(context);

  std::optional<DeviceBatch> first = projection.next(context);
  ASSERT_TRUE(first.has_value());
  ASSERT_EQ(first->row_count(), 3u);
  const cudf::column_view first_view = first->view().column(0);
  for (cudf::size_type i = 0; i < first_view.size(); ++i) {
    const std::unique_ptr<cudf::scalar> element = cudf::get_element(first_view, i);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int64_t>&>(*element).value(), 100);
  }

  std::optional<DeviceBatch> second = projection.next(context);
  ASSERT_TRUE(second.has_value());
  ASSERT_EQ(second->row_count(), 2u);
  const cudf::column_view second_view = second->view().column(0);
  for (cudf::size_type i = 0; i < second_view.size(); ++i) {
    const std::unique_ptr<cudf::scalar> element = cudf::get_element(second_view, i);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int64_t>&>(*element).value(), 200);
  }

  EXPECT_FALSE(projection.next(context).has_value());
  projection.close(context);
}

// materialize_case()'s no-ELSE path had no coverage -- the test above
// always supplies an else_branch. Rows where no WHEN matches must
// evaluate to NULL (the default-constructed-scalar path), not 0 or the
// column's own type's zero value.
TEST(ProjectionOperator, EvaluatesCaseExpressionWithNoElseAsNullForUnmatchedRows) {
  RmmEnvironment env(default_config());
  std::vector<DeviceBatch> batches;
  batches.push_back(make_filled_batch(1, 2));  // 1 <= 5 for every row -- no WHEN matches.

  auto a = std::make_shared<ColumnExpression>("a", 0, int32_type(false));
  auto five = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(5));
  auto a_i64 = std::make_shared<CastExpression>(a, int64_type(false));
  auto condition =
      std::make_shared<BinaryExpression>(BinaryOperator::Greater, a_i64, five, boolean_type(false));
  auto then_value = std::make_shared<LiteralExpression>(LiteralExpression::make_int64(100));
  std::vector<CaseExpression::WhenThen> when_then = {CaseExpression::WhenThen{condition, then_value}};
  auto case_expr = std::make_shared<CaseExpression>(std::move(when_then), nullptr, int64_type(true));

  std::vector<NamedExpression> items = {NamedExpression{case_expr, "bucket"}};
  ProjectionOperator projection(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                                std::move(items));
  ExecutionContext context = make_context();
  projection.open(context);

  std::optional<DeviceBatch> result = projection.next(context);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->row_count(), 2u);
  EXPECT_EQ(result->view().column(0).null_count(), 2);
  EXPECT_FALSE(projection.next(context).has_value());
  projection.close(context);
}

// Regression coverage: compile_value()'s EXTRACT branch and
// materialize_extract() had zero coverage through ProjectionOperator --
// EXTRACT was previously only ever tested as a GROUP BY key (e.g. TPC-H
// Q7/Q9's `l_year`), never as a plain SELECT-list item. Also exercises all
// three DatePart values (cudf_adapter.cpp's to_cudf_datetime_component()
// previously only ever got called with Year in the whole GPU suite).
TEST(ProjectionOperator, EvaluatesExtractForYearMonthAndDay) {
  RmmEnvironment env(default_config());
  // Days since the Unix epoch (1970-01-01): 19723 -> 2024-01-01, 19999 ->
  // 2024-10-03 (cudf::type_id::TIMESTAMP_DAYS' own storage representation).
  Schema schema({Field{"d", date32_type(false)}});
  std::vector<DeviceBatch> batches;
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_from_host<std::int32_t>({19723}, cudf::type_id::TIMESTAMP_DAYS));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_from_host<std::int32_t>({19999}, cudf::type_id::TIMESTAMP_DAYS));
    batches.push_back(DeviceBatch(std::make_unique<cudf::table>(std::move(columns)),
                                  std::make_shared<const Schema>(schema)));
  }

  auto d = std::make_shared<ColumnExpression>("d", 0, date32_type(false));
  auto year_expr = std::make_shared<ExtractExpression>(DatePart::Year, d, int64_type(false));
  auto month_expr = std::make_shared<ExtractExpression>(DatePart::Month, d, int64_type(false));
  auto day_expr = std::make_shared<ExtractExpression>(DatePart::Day, d, int64_type(false));

  std::vector<NamedExpression> items = {NamedExpression{year_expr, "y"}, NamedExpression{month_expr, "m"},
                                        NamedExpression{day_expr, "dd"}};
  ProjectionOperator projection(1, std::make_unique<VectorSourceOperator>(std::move(batches)),
                                std::move(items));
  ExecutionContext context = make_context();
  projection.open(context);

  auto expect_row = [](const DeviceBatch& batch, std::int64_t year, std::int64_t month, std::int64_t day) {
    ASSERT_EQ(batch.row_count(), 1u);
    const std::unique_ptr<cudf::scalar> y = cudf::get_element(batch.view().column(0), 0);
    const std::unique_ptr<cudf::scalar> m = cudf::get_element(batch.view().column(1), 0);
    const std::unique_ptr<cudf::scalar> dd = cudf::get_element(batch.view().column(2), 0);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int64_t>&>(*y).value(), year);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int64_t>&>(*m).value(), month);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<std::int64_t>&>(*dd).value(), day);
  };

  std::optional<DeviceBatch> first = projection.next(context);
  ASSERT_TRUE(first.has_value());
  expect_row(*first, 2024, 1, 1);

  std::optional<DeviceBatch> second = projection.next(context);
  ASSERT_TRUE(second.has_value());
  expect_row(*second, 2024, 10, 3);

  EXPECT_FALSE(projection.next(context).has_value());
  projection.close(context);
}

// Regression coverage: compile_value()'s LIKE branch and
// materialize_like() (including the negated/NOT LIKE path) had zero
// coverage through ProjectionOperator -- LIKE was previously only ever
// tested as a WHERE-clause predicate (FilterOperator::evaluate_like) or a
// GROUP BY/aggregate-argument CASE condition, never as a plain SELECT-list
// boolean item (e.g. `SELECT c_name LIKE 'a%' FROM ...`).
class ProjectionOperatorLikeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_projection_like_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "data.parquet").string();

    arrow::StringBuilder region_builder;
    for (const std::string& region : {"banana", "apple", "cherry"}) {
      ASSERT_TRUE(region_builder.Append(region).ok());
    }
    std::shared_ptr<arrow::Array> region_array;
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    const auto schema = arrow::schema({arrow::field("region", arrow::utf8(), false)});
    const auto table = arrow::Table::Make(schema, {region_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, 3);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
};

TEST_F(ProjectionOperatorLikeTest, EvaluatesLikeAndNotLikeAsProjectedBooleans) {
  RmmEnvironment env(default_config());
  Schema schema({Field{"region", string_type(false)}});
  std::vector<PhysicalFileFragment> fragments = {PhysicalFileFragment{Uri(path_), 3, 1, {0}, {}, {}}};
  LocalObjectStore store;
  auto scan = std::make_unique<ParquetScanOperator>(1, fragments, std::vector<std::string>{"region"},
                                                    std::make_shared<const Schema>(schema), store);

  auto region = std::make_shared<ColumnExpression>("region", 0, string_type(false));
  auto starts_with_a = std::make_shared<LikeExpression>(region, "a%", /*negated=*/false);
  auto not_starts_with_a = std::make_shared<LikeExpression>(region, "a%", /*negated=*/true);
  std::vector<NamedExpression> items = {NamedExpression{starts_with_a, "starts_with_a"},
                                        NamedExpression{not_starts_with_a, "not_starts_with_a"}};
  ProjectionOperator projection(2, std::move(scan), std::move(items));
  ExecutionContext context = make_context();
  projection.open(context);

  std::optional<DeviceBatch> result = projection.next(context);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->row_count(), 3u);

  // Written order preserved (single row group, no reordering): banana,
  // apple, cherry -- only "apple" starts with 'a'.
  const std::vector<bool> expected_starts_with_a = {false, true, false};
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(expected_starts_with_a.size()); ++i) {
    const std::unique_ptr<cudf::scalar> starts = cudf::get_element(result->view().column(0), i);
    const std::unique_ptr<cudf::scalar> not_starts = cudf::get_element(result->view().column(1), i);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<bool>&>(*starts).value(), expected_starts_with_a[i]);
    EXPECT_EQ(static_cast<const cudf::numeric_scalar<bool>&>(*not_starts).value(),
              !expected_starts_with_a[i]);
  }

  EXPECT_FALSE(projection.next(context).has_value());
  projection.close(context);
}

}  // namespace
}  // namespace kernellake
