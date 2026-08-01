#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>

#include "kernellake/types/schema.hpp"

namespace kernellake {

// GPU-resident batch of columnar data. It internally owns a cudf::table,
// but the rest of KernelLake depends on this abstraction rather than on
// every libcudf implementation detail directly (see docs/architecture.md).
// Throws ExecutionError if `table`'s column count doesn't match `schema`'s
// field count -- this is always a KernelLake bug (a mismatched physical
// operator), not a runtime data condition, so it fails immediately at
// construction rather than surfacing confusingly later.
class DeviceBatch {
public:
  DeviceBatch(std::unique_ptr<cudf::table> table, std::shared_ptr<const Schema> schema);

  DeviceBatch(DeviceBatch&&) noexcept = default;
  DeviceBatch& operator=(DeviceBatch&&) noexcept = default;
  DeviceBatch(const DeviceBatch&) = delete;
  DeviceBatch& operator=(const DeviceBatch&) = delete;

  [[nodiscard]] cudf::table_view view() const;
  [[nodiscard]] std::size_t row_count() const;
  [[nodiscard]] std::size_t column_count() const;
  [[nodiscard]] const Schema& schema() const noexcept { return *schema_; }
  [[nodiscard]] std::shared_ptr<const Schema> schema_ptr() const noexcept { return schema_; }

  // Releases the owned cudf::table, leaving this batch schema-only (any
  // further view()/row_count() call is invalid). Operators use this to hand
  // ownership of a table to a new DeviceBatch without an intermediate copy.
  [[nodiscard]] std::unique_ptr<cudf::table> release_table() && { return std::move(table_); }

private:
  std::unique_ptr<cudf::table> table_;
  std::shared_ptr<const Schema> schema_;
};

}  // namespace kernellake
