#pragma once

#include <cstddef>
#include <memory>

#include "kernellake/execution/operator.hpp"
#include "kernellake/planner/physical_plan.hpp"

namespace kernellake {

// Converts an optimized PhysicalPlan into a tree of concrete
// PhysicalOperator instances, ready for open()/next()/close().
//
// `pass_read_limit_bytes` (0 = unlimited) is forwarded to
// ParquetScanOperator to bound its per-pass decompression memory; see that
// operator's own docs for why this is a byte budget rather than an exact
// row count.
[[nodiscard]] std::unique_ptr<PhysicalOperator> build_operator_tree(const PhysicalPlanPtr& plan,
                                                                    std::size_t pass_read_limit_bytes);

}  // namespace kernellake
