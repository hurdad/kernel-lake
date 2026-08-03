#include "kernellake/execution_gpu/operator_builder.hpp"

#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <optional>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/arrow_result_operator.hpp"
#include "kernellake/execution_gpu/filter_operator.hpp"
#include "kernellake/execution_gpu/hash_aggregate_operator.hpp"
#include "kernellake/execution_gpu/hash_join_operator.hpp"
#include "kernellake/execution_gpu/limit_operator.hpp"
#include "kernellake/execution_gpu/parquet_scan_operator.hpp"
#include "kernellake/execution_gpu/projection_operator.hpp"
#include "kernellake/execution_gpu/scalar_aggregate_operator.hpp"
#include "kernellake/execution_gpu/sort_operator.hpp"

namespace kernellake {

namespace {

// Wraps every operator in the tree generically (see build() below), so no
// individual operator needs its own timing/NVTX code. Every PhysicalOperator
// subclass's name() returns a string literal (checked across every
// concrete operator in this codebase), so the std::string_view it returns
// is always null-terminated -- `.data()` is safe to pass to NVTX/use as a
// map key without a copy on the hot path, though MetricsRegistry::record()
// still takes a copy internally to own the key.
class InstrumentedOperator final : public PhysicalOperator {
 public:
  InstrumentedOperator(std::unique_ptr<PhysicalOperator> inner, bool nvtx_enabled)
      : inner_(std::move(inner)), nvtx_enabled_(nvtx_enabled) {}

  void open(ExecutionContext& context) override { inner_->open(context); }

  std::optional<DeviceBatch> next(ExecutionContext& context) override {
    std::optional<nvtx3::scoped_range> range;
    if (nvtx_enabled_) range.emplace(inner_->name().data());
    const auto start = std::chrono::steady_clock::now();
    std::optional<DeviceBatch> result = inner_->next(context);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (context.metrics != nullptr) context.metrics->record(inner_->name(), seconds);
    return result;
  }

  void close(ExecutionContext& context) override { inner_->close(context); }

  [[nodiscard]] std::string_view name() const noexcept override { return inner_->name(); }
  [[nodiscard]] OperatorId id() const noexcept override { return inner_->id(); }

 private:
  std::unique_ptr<PhysicalOperator> inner_;
  bool nvtx_enabled_;
};

std::unique_ptr<PhysicalOperator> build(const PhysicalPlanPtr& node, ObjectStore& store,
                                        std::size_t pass_read_limit_bytes, OperatorId& next_id,
                                        bool nvtx_enabled) {
  auto instrument = [nvtx_enabled](std::unique_ptr<PhysicalOperator> op) {
    return std::make_unique<InstrumentedOperator>(std::move(op), nvtx_enabled);
  };

  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(node.get())) {
    return instrument(std::make_unique<ParquetScanOperator>(
        next_id++, scan->fragments(), scan->columns(), std::make_shared<const Schema>(scan->output_schema()),
        store, pass_read_limit_bytes));
  }
  if (const auto* join = dynamic_cast<const HashJoinNode*>(node.get())) {
    // Built as two separate statements, not two arguments of the same
    // call: argument evaluation order is unspecified in C++, and both
    // recursive build() calls mutate `next_id` as a side effect, so leaving
    // it to the compiler would make operator ID assignment
    // non-deterministic across the two subtrees.
    std::unique_ptr<PhysicalOperator> left =
        build(join->left(), store, pass_read_limit_bytes, next_id, nvtx_enabled);
    std::unique_ptr<PhysicalOperator> right =
        build(join->right(), store, pass_read_limit_bytes, next_id, nvtx_enabled);
    return instrument(std::make_unique<HashJoinOperator>(
        next_id++, std::move(left), std::move(right), join->left_key_index(), join->right_key_index(),
        std::make_shared<const Schema>(join->output_schema())));
  }
  if (const auto* filter = dynamic_cast<const FilterNode*>(node.get())) {
    return instrument(std::make_unique<FilterOperator>(
        next_id++, build(filter->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled),
        filter->predicate()));
  }
  if (const auto* projection = dynamic_cast<const ProjectionNode*>(node.get())) {
    return instrument(std::make_unique<ProjectionOperator>(
        next_id++, build(projection->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled),
        projection->items()));
  }
  if (const auto* hash_aggregate = dynamic_cast<const HashAggregateNode*>(node.get())) {
    return instrument(std::make_unique<HashAggregateOperator>(
        next_id++, build(hash_aggregate->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled),
        hash_aggregate->group_by(), hash_aggregate->aggregates()));
  }
  if (const auto* scalar_aggregate = dynamic_cast<const ScalarAggregateNode*>(node.get())) {
    return instrument(std::make_unique<ScalarAggregateOperator>(
        next_id++, build(scalar_aggregate->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled),
        scalar_aggregate->aggregates()));
  }
  if (const auto* sort = dynamic_cast<const SortNode*>(node.get())) {
    return instrument(std::make_unique<SortOperator>(
        next_id++, build(sort->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled), sort->keys()));
  }
  if (const auto* limit = dynamic_cast<const LimitNode*>(node.get())) {
    return instrument(std::make_unique<LimitOperator>(
        next_id++, build(limit->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled),
        limit->limit()));
  }
  if (const auto* arrow_result = dynamic_cast<const ArrowResultNode*>(node.get())) {
    return instrument(std::make_unique<ArrowResultOperator>(
        next_id++, build(arrow_result->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled)));
  }
  throw PlanningError("build_operator_tree: unrecognized physical plan node");
}

}  // namespace

std::unique_ptr<PhysicalOperator> build_operator_tree(const PhysicalPlanPtr& plan, ObjectStore& store,
                                                      std::size_t pass_read_limit_bytes, bool nvtx_enabled) {
  OperatorId next_id = 1;
  return build(plan, store, pass_read_limit_bytes, next_id, nvtx_enabled);
}

}  // namespace kernellake
