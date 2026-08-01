#pragma once

#include <memory>

#include "kernellake/execution/operator.hpp"

namespace kernellake {

// Terminal operator: passes each batch from `child` through unchanged.
// Consumers of the pipeline (QueryEngine::execute()) call
// to_arrow_record_batch() on each returned DeviceBatch themselves, since
// PhysicalOperator::next() must return a DeviceBatch (GPU-resident) rather
// than an Arrow RecordBatch (host-resident) -- keeping the device-to-host
// transfer at the one point where the caller actually needs host data,
// rather than inside the operator pipeline itself.
class ArrowResultOperator final : public PhysicalOperator {
public:
  ArrowResultOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "ArrowResult"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

private:
  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
};

}  // namespace kernellake
