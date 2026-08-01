#pragma once

#include <string>
#include <vector>

#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// The engine's pruning decision for one file: which row groups must be
// scanned, which were proven unnecessary, and why -- so pruning is always
// explainable rather than a silent count.
struct ScanDecision {
  Uri file;
  std::vector<int> selected_row_groups;
  std::vector<int> skipped_row_groups;
  std::vector<std::string> reasons;
};

// Evaluates `predicates` (extracted from the WHERE clause by the optimizer's
// predicate-pushdown rule, see LogicalScan::pushable_predicates()) against
// `file`'s per-row-group min/max statistics.
//
// A row group is only ever skipped when a predicate can be *proven*
// impossible to satisfy from its statistics; missing, partial, or
// incomparable statistics always fall back to "must scan" for that
// predicate. Correctness takes priority over aggressive pruning.
[[nodiscard]] ScanDecision evaluate_pruning(const FileMetadata& file,
                                            const std::vector<PushablePredicate>& predicates);

}  // namespace kernellake
