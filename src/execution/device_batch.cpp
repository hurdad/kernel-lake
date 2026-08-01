#include "kernellake/execution/device_batch.hpp"

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

void validate(const cudf::table& table, const Schema& schema) {
  if (static_cast<std::size_t>(table.num_columns()) != schema.field_count()) {
    throw ExecutionError("DeviceBatch column/schema mismatch: table has " +
                         std::to_string(table.num_columns()) + " columns, schema has " +
                         std::to_string(schema.field_count()) + " fields");
  }
}

}  // namespace

DeviceBatch::DeviceBatch(std::unique_ptr<cudf::table> table, std::shared_ptr<const Schema> schema)
    : table_(std::move(table)), schema_(std::move(schema)) {
  if (table_ == nullptr) throw ExecutionError("DeviceBatch constructed with a null table");
  if (schema_ == nullptr) throw ExecutionError("DeviceBatch constructed with a null schema");
  validate(*table_, *schema_);
}

cudf::table_view DeviceBatch::view() const {
  return table_->view();
}

std::size_t DeviceBatch::row_count() const {
  return table_->num_columns() == 0 ? 0 : static_cast<std::size_t>(table_->num_rows());
}

std::size_t DeviceBatch::column_count() const {
  return static_cast<std::size_t>(table_->num_columns());
}

}  // namespace kernellake
