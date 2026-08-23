#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/array/builder_time.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include "kernellake/execution_gpu/arrow_bridge.hpp"
#include "kernellake/execution_gpu/device_batch.hpp"

namespace kernellake {
namespace {

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

TEST(ArrowBridge, DeviceBatchToArrowRoundTripsValues) {
  DeviceBatch batch = make_filled_batch(42, 100);
  const std::shared_ptr<arrow::RecordBatch> record_batch = to_arrow_record_batch(batch);

  ASSERT_NE(record_batch, nullptr);
  EXPECT_EQ(record_batch->num_rows(), 100);
  ASSERT_EQ(record_batch->num_columns(), 1);
  EXPECT_EQ(record_batch->schema()->field(0)->name(), "a");

  const auto int_array = std::static_pointer_cast<arrow::Int32Array>(record_batch->column(0));
  ASSERT_EQ(int_array->length(), 100);
  for (int64_t i = 0; i < int_array->length(); ++i) {
    ASSERT_FALSE(int_array->IsNull(i));
    EXPECT_EQ(int_array->Value(i), 42);
  }
}

TEST(ArrowBridge, ArrowToDeviceBatchRoundTripsValues) {
  arrow::Int32Builder builder;
  for (int i = 0; i < 50; ++i) {
    ASSERT_TRUE(builder.Append(i * 2).ok());
  }
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto arrow_schema = arrow::schema({arrow::field("a", arrow::int32(), false)});
  const auto record_batch = arrow::RecordBatch::Make(arrow_schema, 50, {array});

  const DeviceBatch batch =
      from_arrow_record_batch(*record_batch, std::make_shared<const Schema>(one_int_column_schema()));
  EXPECT_EQ(batch.row_count(), 50u);
  EXPECT_EQ(batch.column_count(), 1u);

  // Round-trip back through to_arrow_record_batch to verify the values
  // actually made it onto the GPU and back, not just the row count.
  const std::shared_ptr<arrow::RecordBatch> back = to_arrow_record_batch(batch);
  const auto int_array = std::static_pointer_cast<arrow::Int32Array>(back->column(0));
  for (int64_t i = 0; i < int_array->length(); ++i) {
    EXPECT_EQ(int_array->Value(i), i * 2);
  }
}

// cudf::from_arrow throws for Arrow types it has no matching column type
// for (cudf has no interval type at all) -- exercises
// from_arrow_record_batch's catch(...) block, which must release both C
// Data Interface structs before rethrowing (ExportRecordBatch handed this
// function ownership of them; leaking on the error path would be a real
// bug, not just a missed line).
TEST(ArrowBridge, FromArrowRecordBatchWithUnsupportedTypeThrowsAndCleansUp) {
  arrow::MonthDayNanoIntervalBuilder builder;
  ASSERT_TRUE(builder.Append(arrow::MonthDayNanoIntervalType::MonthDayNanos{1, 2, 3}).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  const auto arrow_schema = arrow::schema({arrow::field("a", arrow::month_day_nano_interval(), false)});
  const auto record_batch = arrow::RecordBatch::Make(arrow_schema, 1, {array});

  EXPECT_ANY_THROW({
    [[maybe_unused]] const DeviceBatch batch =
        from_arrow_record_batch(*record_batch, std::make_shared<const Schema>(one_int_column_schema()));
  });
}

}  // namespace
}  // namespace kernellake
