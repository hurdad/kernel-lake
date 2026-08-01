#pragma once

#include <optional>
#include <string_view>

#include "kernellake/common/identifiers.hpp"
#include "kernellake/execution/device_batch.hpp"
#include "kernellake/execution/execution_context.hpp"

namespace kernellake {

// Streaming, bounded-memory operator interface. Operators consume and
// produce DeviceBatch values one at a time via next() -- never materializing
// an entire dataset -- and must release batches they no longer need,
// support cancellation checks via context.cancellation, and use
// context.stream rather than the default stream so pipelines can overlap.
class PhysicalOperator {
public:
  virtual ~PhysicalOperator() = default;

  virtual void open(ExecutionContext& context) = 0;

  // Returns std::nullopt once the operator is exhausted.
  virtual std::optional<DeviceBatch> next(ExecutionContext& context) = 0;

  virtual void close(ExecutionContext& context) = 0;

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual OperatorId id() const noexcept = 0;
};

}  // namespace kernellake
