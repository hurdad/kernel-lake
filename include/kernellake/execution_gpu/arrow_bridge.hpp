#pragma once

#include <arrow/api.h>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

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
//
// `stream`/`mr` default to cudf's own default stream/resource (matching
// this function's long-standing behavior) but every real engine call site
// passes its ExecutionContext's own `stream`/`memory_resource` explicitly
// -- see operator.hpp's house rules. Doing so makes the device->host copy
// properly stream-ordered against the rest of that query's work, instead
// of forcing an implicit full-device barrier through cudf's null-stream
// default every time a result batch (or a grace-join spill batch) crosses
// this boundary.
[[nodiscard]] std::shared_ptr<arrow::RecordBatch> to_arrow_record_batch(
    const cudf::table_view& view, const Schema& schema,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// Same as above, for an already-owned DeviceBatch.
[[nodiscard]] std::shared_ptr<arrow::RecordBatch> to_arrow_record_batch(
    const DeviceBatch& batch, rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// Converts a host-resident Arrow RecordBatch to a GPU-resident DeviceBatch
// (host->device transfer). `schema` must describe the same columns as
// `batch` (see DeviceBatch's constructor-time validation). Same `stream`/
// `mr` rationale as to_arrow_record_batch() above.
[[nodiscard]] DeviceBatch from_arrow_record_batch(
    const arrow::RecordBatch& batch, std::shared_ptr<const Schema> schema,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace kernellake
