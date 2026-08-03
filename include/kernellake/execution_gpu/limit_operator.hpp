#pragma once

#include <memory>

#include "kernellake/execution_gpu/operator.hpp"

namespace kernellake {

// Caps the total number of rows returned across all batches from `child`.
// Batches are passed through unchanged until the limit would be exceeded,
// at which point the final batch is truncated (via cudf::slice + a deep
// copy into an owned table) and no further batches are pulled.
class LimitOperator final : public PhysicalOperator {
 public:
  LimitOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child, std::int64_t limit);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "Limit"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::int64_t limit_;
  std::int64_t remaining_ = 0;
};

}  // namespace kernellake
