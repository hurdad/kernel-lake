#pragma once

#include <cudf/table/table.hpp>

#include <memory>
#include <optional>
#include <vector>

#include "kernellake/execution_gpu/expression_compiler.hpp"
#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake {

// ORDER BY: a blocking operator, unlike every other operator in this file --
// a global sort needs every row before it can produce the first one. Pulls
// `child` to exhaustion, concatenates every batch into a single table (see
// docs/ARCHITECTURE.md for why this means ORDER BY's memory footprint is the
// whole result set, not bounded like the streaming operators), sorts via
// cudf::stable_sorted_order + cudf::gather, and returns exactly one output
// batch before reporting exhausted.
class SortOperator final : public PhysicalOperator {
 public:
  SortOperator(OperatorId id, std::unique_ptr<PhysicalOperator> child, std::vector<LogicalSort::Key> keys);

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "Sort"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

 private:
  // A plain column reference (e.g. `ORDER BY region`) is referenced
  // directly rather than routed through cudf::ast::compute_column: cudf's
  // AST evaluator can only materialize fixed-width output columns, so a
  // STRING sort key would abort with "Invalid, non-fixed-width type" even
  // though no computation was requested (same issue as the other
  // operators -- see docs/ARCHITECTURE.md's "GPU operators" section).
  struct CompiledKey {
    std::optional<cudf::size_type> source_column_index;
    const cudf::ast::expression* expr = nullptr;
    bool ascending = true;
  };

  [[nodiscard]] CompiledKey compile_key(const LogicalSort::Key& key);

  OperatorId id_;
  std::unique_ptr<PhysicalOperator> child_;
  std::vector<LogicalSort::Key> keys_;
  ExpressionCompiler compiler_;
  std::vector<CompiledKey> compiled_keys_;
  bool produced_ = false;
};

}  // namespace kernellake
