#include "kernellake/execution_gpu/arrow_bridge.hpp"

#include <arrow/c/bridge.h>
#include <cudf/interop.hpp>
#include <fmt/format.h>

#include <vector>

#include "kernellake/common/errors.hpp"

namespace kernellake {

std::shared_ptr<arrow::RecordBatch> to_arrow_record_batch(const DeviceBatch& batch) {
  std::vector<cudf::column_metadata> column_metadata;
  column_metadata.reserve(batch.schema().field_count());
  for (const Field& field : batch.schema().fields()) {
    column_metadata.emplace_back(field.name);
  }

  const cudf::unique_schema_t arrow_schema = cudf::to_arrow_schema(batch.view(), column_metadata);
  const cudf::unique_device_array_t device_array = cudf::to_arrow_host(batch.view());

  // ImportRecordBatch consumes (releases) both structs per the C Data
  // Interface contract; arrow_schema/device_array's own unique_ptr deleters
  // still run safely afterward since release() is idempotent (they free the
  // outer struct allocation, which Arrow's import does not own).
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> result =
      arrow::ImportRecordBatch(&device_array->array, arrow_schema.get());
  if (!result.ok()) {
    throw ExecutionError(
        fmt::format("failed to import Arrow record batch from cudf table: {}", result.status().ToString()));
  }
  return *result;
}

DeviceBatch from_arrow_record_batch(const arrow::RecordBatch& batch, std::shared_ptr<const Schema> schema) {
  ArrowArray c_array{};
  ArrowSchema c_schema{};
  const arrow::Status status = arrow::ExportRecordBatch(batch, &c_array, &c_schema);
  if (!status.ok()) {
    throw ExecutionError(fmt::format("failed to export Arrow record batch for cudf: {}", status.ToString()));
  }

  // Unlike the to-Arrow direction, cudf::from_arrow does NOT release the
  // input Array/Schema -- that is the caller's responsibility.
  std::unique_ptr<cudf::table> table;
  try {
    table = cudf::from_arrow(&c_schema, &c_array);
  } catch (...) {
    if (c_array.release != nullptr) c_array.release(&c_array);
    if (c_schema.release != nullptr) c_schema.release(&c_schema);
    throw;
  }
  if (c_array.release != nullptr) c_array.release(&c_array);
  if (c_schema.release != nullptr) c_schema.release(&c_schema);

  return DeviceBatch(std::move(table), std::move(schema));
}

}  // namespace kernellake
