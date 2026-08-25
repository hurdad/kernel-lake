#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>

#include "kernellake/execution_gpu/operator.hpp"

namespace kernellake {

// Caps every batch this operator yields at `max_rows`: an oversized batch
// pulled from `child` is split into consecutive chunks of at most
// `max_rows` rows each, returned one per next() call; a batch already
// within the cap passes through unchanged (no copy, the common case).
// Backs EngineSection::batch_rows (operator_builder.cpp wraps each
// ParquetScanNode's own operator with this) and
// EngineSection::result_batch_rows (wraps the operator feeding
// ArrowResultOperator, i.e. the whole query's final output) -- see
// build_operator_tree()'s own doc comment for where each is applied.
//
// Splitting is not free: DeviceBatch only ever owns a cudf::table outright
// (no reference-counted slice/view type -- see that class's own doc
// comment), so producing each chunk after the first materializes a real
// device-to-device copy of that chunk's column data (cudf::table's
// table_view-copying constructor), not a zero-copy view into the original
// batch. This only costs anything for a batch that actually exceeds
// max_rows in the first place; an already-small batch is returned as-is.
class BatchSizeLimitOperator final : public PhysicalOperator {
 public:
  BatchSizeLimitOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child, std::size_t max_rows);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "BatchSizeLimit"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::size_t max_rows_;
  // An oversized batch pulled from child_, not yet fully sliced out --
  // std::nullopt when there's nothing pending (the common, no-split case).
  std::optional<DeviceBatch> pending_;
  std::size_t pending_offset_ = 0;
};

}  // namespace kernellake
