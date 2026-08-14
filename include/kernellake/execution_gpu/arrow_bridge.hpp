#pragma once

#include <arrow/api.h>

#include <memory>

#include "kernellake/execution_gpu/device_batch.hpp"

namespace kernellake {

// Converts a GPU-resident cudf::table_view to a host-resident Arrow
// RecordBatch. Internally: cudf::to_arrow_host() copies the column data
// device->host and wraps it in the Arrow C Data Interface (ArrowArray/
// ArrowSchema), which Arrow C++'s own bridge (arrow::ImportRecordBatch)
// then imports into a real arrow::RecordBatch. This is the one
// unavoidable device-to-host transfer boundary for materializing a query
// result; see docs/ARCHITECTURE.md. Takes a view directly (rather than
// only a DeviceBatch) so a non-owning slice -- e.g. one partition's rows
// out of cudf::slice(), as HashJoinOperator's grace-join path spills to
// host -- can be converted without an intermediate device-to-device copy
// through an owned cudf::table first.
[[nodiscard]] std::shared_ptr<arrow::RecordBatch> to_arrow_record_batch(const cudf::table_view& view,
                                                                        const Schema& schema);

// Same as above, for an already-owned DeviceBatch.
[[nodiscard]] std::shared_ptr<arrow::RecordBatch> to_arrow_record_batch(const DeviceBatch& batch);

// Converts a host-resident Arrow RecordBatch to a GPU-resident DeviceBatch
// (host->device transfer). `schema` must describe the same columns as
// `batch` (see DeviceBatch's constructor-time validation).
[[nodiscard]] DeviceBatch from_arrow_record_batch(const arrow::RecordBatch& batch,
                                                  std::shared_ptr<const Schema> schema);

}  // namespace kernellake
