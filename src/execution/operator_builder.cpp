#include "kernellake/execution/operator_builder.hpp"

#include "kernellake/common/errors.hpp"
#include "kernellake/execution/arrow_result_operator.hpp"
#include "kernellake/execution/filter_operator.hpp"
#include "kernellake/execution/hash_aggregate_operator.hpp"
#include "kernellake/execution/limit_operator.hpp"
#include "kernellake/execution/parquet_scan_operator.hpp"
#include "kernellake/execution/projection_operator.hpp"
#include "kernellake/execution/scalar_aggregate_operator.hpp"

namespace kernellake {

namespace {

std::unique_ptr<PhysicalOperator> build(const PhysicalPlanPtr& node, std::size_t pass_read_limit_bytes,
                                         OperatorId& next_id) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(node.get())) {
    return std::make_unique<ParquetScanOperator>(next_id++, scan->fragments(), scan->columns(),
                                                  std::make_shared<const Schema>(scan->output_schema()),
                                                  pass_read_limit_bytes);
  }
  if (const auto* filter = dynamic_cast<const FilterNode*>(node.get())) {
    return std::make_unique<FilterOperator>(next_id++, build(filter->child(), pass_read_limit_bytes, next_id),
                                             filter->predicate());
  }
  if (const auto* projection = dynamic_cast<const ProjectionNode*>(node.get())) {
    return std::make_unique<ProjectionOperator>(
        next_id++, build(projection->child(), pass_read_limit_bytes, next_id), projection->items());
  }
  if (const auto* hash_aggregate = dynamic_cast<const HashAggregateNode*>(node.get())) {
    return std::make_unique<HashAggregateOperator>(
        next_id++, build(hash_aggregate->child(), pass_read_limit_bytes, next_id), hash_aggregate->group_by(),
        hash_aggregate->aggregates());
  }
  if (const auto* scalar_aggregate = dynamic_cast<const ScalarAggregateNode*>(node.get())) {
    return std::make_unique<ScalarAggregateOperator>(
        next_id++, build(scalar_aggregate->child(), pass_read_limit_bytes, next_id),
        scalar_aggregate->aggregates());
  }
  if (const auto* limit = dynamic_cast<const LimitNode*>(node.get())) {
    return std::make_unique<LimitOperator>(next_id++, build(limit->child(), pass_read_limit_bytes, next_id),
                                            limit->limit());
  }
  if (const auto* arrow_result = dynamic_cast<const ArrowResultNode*>(node.get())) {
    return std::make_unique<ArrowResultOperator>(
        next_id++, build(arrow_result->child(), pass_read_limit_bytes, next_id));
  }
  throw PlanningError("build_operator_tree: unrecognized physical plan node");
}

}  // namespace

std::unique_ptr<PhysicalOperator> build_operator_tree(const PhysicalPlanPtr& plan,
                                                       std::size_t pass_read_limit_bytes) {
  OperatorId next_id = 1;
  return build(plan, pass_read_limit_bytes, next_id);
}

}  // namespace kernellake
