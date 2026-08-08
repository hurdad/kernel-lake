#pragma once

#include <optional>
#include <string_view>

#include "kernellake/common/identifiers.hpp"
#include "kernellake/execution_gpu/device_batch.hpp"
#include "kernellake/execution_gpu/execution_context.hpp"

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

  // Total seconds of real resource work this operator did that next()'s own
  // wall-clock self-time (see InstrumentedOperator/MetricsRegistry) may no
  // longer fully capture -- e.g. ParquetScanOperator's background decode
  // thread, whose work is deliberately *overlapped* with the consumer's own
  // next() calls rather than happening inside them (see that operator's
  // class comment on decode/compute overlap), so a plain wall-clock wrapper
  // around next() would under-report it. std::nullopt (the default) means
  // "next()'s own self-time already is this operator's real cost" -- most
  // operators never need to override this. Callers must read it only after
  // close() (see InstrumentedOperator), so it reflects the operator's final
  // total rather than a mid-execution snapshot.
  [[nodiscard]] virtual std::optional<double> resource_seconds() const { return std::nullopt; }
};

}  // namespace kernellake
