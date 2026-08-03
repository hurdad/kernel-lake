#include "kernellake/execution_gpu/arrow_result_operator.hpp"

namespace kernellake {

ArrowResultOperator::ArrowResultOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child)
    : id_(id), child_(std::move(child)) {}

void ArrowResultOperator::open(ExecutionContext& context) {
  child_->open(context);
}

std::optional<DeviceBatch> ArrowResultOperator::next(ExecutionContext& context) {
  return child_->next(context);
}

void ArrowResultOperator::close(ExecutionContext& context) {
  child_->close(context);
}

}  // namespace kernellake
