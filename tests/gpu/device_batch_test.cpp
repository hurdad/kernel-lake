#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/device_batch.hpp"

namespace kernellake {
namespace {

std::unique_ptr<cudf::table> make_table(int num_columns, cudf::size_type num_rows) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int i = 0; i < num_columns; ++i) {
    columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

Schema two_column_schema() {
  return Schema({Field{"a", int32_type(false)}, Field{"b", int32_type(false)}});
}

TEST(DeviceBatch, ReportsRowAndColumnCounts) {
  DeviceBatch batch(make_table(2, 1000), std::make_shared<const Schema>(two_column_schema()));
  EXPECT_EQ(batch.row_count(), 1000u);
  EXPECT_EQ(batch.column_count(), 2u);
  EXPECT_EQ(batch.view().num_columns(), 2);
  EXPECT_EQ(batch.view().num_rows(), 1000);
}

TEST(DeviceBatch, ExposesSchema) {
  DeviceBatch batch(make_table(2, 10), std::make_shared<const Schema>(two_column_schema()));
  EXPECT_EQ(batch.schema().field(0).name, "a");
  EXPECT_EQ(batch.schema().field(1).name, "b");
}

TEST(DeviceBatch, RejectsColumnSchemaMismatch) {
  EXPECT_THROW((void)(DeviceBatch(make_table(1, 10), std::make_shared<const Schema>(two_column_schema()))),
               ExecutionError);
}

TEST(DeviceBatch, RejectsNullTable) {
  EXPECT_THROW((void)(DeviceBatch(nullptr, std::make_shared<const Schema>(two_column_schema()))),
               ExecutionError);
}

TEST(DeviceBatch, IsMoveOnly) {
  static_assert(!std::is_copy_constructible_v<DeviceBatch>);
  static_assert(std::is_move_constructible_v<DeviceBatch>);
}

}  // namespace
}  // namespace kernellake
