#include "kernellake/execution_gpu/operator_builder.hpp"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <cstdio>
#include <optional>
#include <string>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/arrow_result_operator.hpp"
#include "kernellake/execution_gpu/filter_operator.hpp"
#include "kernellake/execution_gpu/hash_aggregate_operator.hpp"
#include "kernellake/execution_gpu/hash_join_operator.hpp"
#include "kernellake/execution_gpu/limit_operator.hpp"
#include "kernellake/execution_gpu/parquet_scan_operator.hpp"
#include "kernellake/execution_gpu/projection_operator.hpp"
#include "kernellake/execution_gpu/scalar_aggregate_operator.hpp"
#include "kernellake/execution_gpu/semi_anti_join_operator.hpp"
#include "kernellake/execution_gpu/sort_operator.hpp"
#include "kernellake/observability/query_tracing.hpp"

namespace kernellake {

namespace {

// Wraps every operator in the tree generically (see build() below), so no
// individual operator needs its own timing/NVTX/tracing code. Every
// PhysicalOperator subclass's name() returns a string literal (checked
// across every concrete operator in this codebase), so the std::string_view
// it returns is always null-terminated -- `.data()` is safe to pass to
// NVTX/use as a map key without a copy on the hot path, though
// MetricsRegistry::record() still takes a copy internally to own the key.
//
// Also gives every operator its own child span (see docs/OBSERVABILITY.md),
// so a real trace tool (Jaeger) shows a span tree shaped like the physical
// plan itself, not one flat whole-query span with a pile of attributes.
// Each span's lifetime is open() through close() -- see
// ExecutionContext::current_span's own comment for why the *parenting* of
// each child's span is only threaded through context for the duration of
// this operator's own recursive inner_->open() call, not held for this
// operator's whole lifetime.
class InstrumentedOperator final : public PhysicalOperator {
 public:
  InstrumentedOperator(std::unique_ptr<PhysicalOperator> inner, bool nvtx_enabled)
      : inner_(std::move(inner)), nvtx_enabled_(nvtx_enabled) {}

  void open(ExecutionContext& context) override {
    span_ = context.current_span != nullptr
                ? observability::start_client_span(inner_->name(), *context.current_span)
                : observability::start_client_span(inner_->name());

    // Only attached to context for the duration of the recursive
    // inner_->open() call immediately below (which is what invokes any
    // child operator's own InstrumentedOperator::open()) -- restored right
    // after, so a second child opened later by the same inner_->open() call
    // (e.g. HashJoinOperator opening its right side after its left) parents
    // its own span here too, as this operator's child, not as the first
    // child's child. See ExecutionContext::current_span's own comment.
    const observability::ClientSpan* const previous_span = context.current_span;
    context.current_span = &*span_;
    try {
      inner_->open(context);
    } catch (const std::exception& e) {
      context.current_span = previous_span;
      span_->finish_error(e.what());
      throw;
    }
    context.current_span = previous_span;
  }

  std::optional<DeviceBatch> next(ExecutionContext& context) override {
    std::optional<nvtx3::scoped_range> range;
    if (nvtx_enabled_) range.emplace(inner_->name().data());
    const auto start = std::chrono::steady_clock::now();
    std::optional<DeviceBatch> result;
    try {
      result = inner_->next(context);
    } catch (const std::exception& e) {
      if (context.metrics != nullptr) {
        context.metrics->record(
            inner_->name(), std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
      }
      span_->finish_error(e.what());
      throw;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (context.metrics != nullptr) context.metrics->record(inner_->name(), seconds);
    return result;
  }

  void close(ExecutionContext& context) override {
    // inner_->close() first, not after: it's what joins ParquetScanOperator's
    // background decode thread (see that class), so resource_seconds()
    // below only reflects a *final* total if this happens first.
    inner_->close(context);
    if (const std::optional<double> resource_seconds = inner_->resource_seconds()) {
      span_->set_attribute("kernellake.operator.resource_seconds", *resource_seconds);
      if (context.metrics != nullptr) {
        // A second, distinctly-keyed MetricsRegistry total alongside the
        // plain self-time one next() already records above: for an
        // operator like ParquetScanOperator whose real cost is
        // deliberately overlapped with (not included in) its own next()
        // calls' wall-clock time, next()'s self-time alone would
        // under-report -- see resource_seconds()'s own doc comment.
        // query_engine_execute_gpu.cpp reads this back via the same
        // derived key to populate QueryResult::parquet_decoding_seconds
        // accurately post-overlap, without needing a direct pointer to
        // whichever operator instance(s) in the tree provided it.
        context.metrics->record(fmt::format("{}.resource_seconds", inner_->name()), *resource_seconds);
      }
    }
    span_->finish_ok();
  }

  [[nodiscard]] std::string_view name() const noexcept override { return inner_->name(); }
  [[nodiscard]] OperatorId id() const noexcept override { return inner_->id(); }

 private:
  std::unique_ptr<PhysicalOperator> inner_;
  bool nvtx_enabled_;
  std::optional<observability::ClientSpan> span_;
};

std::unique_ptr<PhysicalOperator> build(const PhysicalPlanPtr& node, ObjectStore& store,
                                        std::size_t pass_read_limit_bytes, OperatorId& next_id,
                                        bool nvtx_enabled, std::size_t build_side_budget_bytes,
                                        const std::string& spill_directory, std::uint64_t max_distinct_keys) {
  auto instrument = [nvtx_enabled](std::unique_ptr<PhysicalOperator> op) {
    return std::make_unique<InstrumentedOperator>(std::move(op), nvtx_enabled);
  };

  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(node.get())) {
    // See acero_query_executor.cpp's translate() for the identical
    // reasoning: a scan resolved with per-table vended credentials (Unity
    // Catalog's S3/GCS/Azure) must read through that store, not the
    // caller's own default.
    ObjectStore& effective_store = scan->owned_store() != nullptr ? *scan->owned_store() : store;
    return instrument(std::make_unique<ParquetScanOperator>(
        next_id++, scan->fragments(), scan->columns(), std::make_shared<const Schema>(scan->output_schema()),
        effective_store, pass_read_limit_bytes, scan->partition_columns()));
  }
  if (const auto* join = dynamic_cast<const HashJoinNode*>(node.get())) {
    // LEFT SEMI/LEFT ANTI (see SemiAntiJoinOperator's own class comment):
    // a much simpler operator with no partitioned/spilling mode at all
    // yet, so none of HashJoinOperator's own partition-sizing/budget-
    // halving machinery below applies -- both children just get the
    // plain pass_read_limit_bytes every non-join subtree already uses.
    if (join->join_type() == JoinType::LeftSemi || join->join_type() == JoinType::LeftAnti) {
      std::unique_ptr<PhysicalOperator> semi_anti_left =
          build(join->left(), store, pass_read_limit_bytes, next_id, nvtx_enabled, build_side_budget_bytes,
                spill_directory, max_distinct_keys);
      std::unique_ptr<PhysicalOperator> semi_anti_right =
          build(join->right(), store, pass_read_limit_bytes, next_id, nvtx_enabled, build_side_budget_bytes,
                spill_directory, max_distinct_keys);
      return instrument(std::make_unique<SemiAntiJoinOperator>(
          next_id++, std::move(semi_anti_left), std::move(semi_anti_right), join->left_key_index(),
          join->right_key_index(), std::make_shared<const Schema>(join->output_schema()), join->join_type()));
    }
    // Computed *before* recursing into children (unlike every other case
    // here): choose_partition_count() only needs the plan node itself
    // (HashJoinNode::estimated_build_rows()/right()->output_schema()), not
    // a built operator, and the result changes what pass_read_limit_bytes
    // the two child subtrees below should actually use.
    const std::size_t partition_count = choose_partition_count(
        join->estimated_build_rows(), join->right()->output_schema(), build_side_budget_bytes);
    // Halved (not the plain pass_read_limit_bytes every other subtree
    // gets) only when this join is partitioned: spill_partitioned_to_disk
    // (hash_join_operator.cpp) runs cudf::hash_partition() on every batch
    // these two subtrees' own ParquetScanOperators produce, which
    // allocates a full *reordered copy* of that batch on top of the
    // decoded batch itself -- a real cost pass_read_limit_bytes' own /4
    // divisor was never sized to cover (it was tuned for a plain scan
    // pass, see that divisor's own comment). Confirmed for real: a
    // partitioned build side alone (halving build_side_budget_bytes from
    // ceiling/2 to ceiling/8) did *not* fix a real SF1000 TPC-H Q14
    // bad_alloc -- the failing allocation size stayed ~1.36 GiB,
    // unmoved, across both budgets, because the actual failure was in
    // this doubling during the *scan* itself, not bucket sizing.
    const std::size_t child_pass_read_limit_bytes =
        partition_count > 1 ? pass_read_limit_bytes / 2 : pass_read_limit_bytes;
    // Built as two separate statements, not two arguments of the same
    // call: argument evaluation order is unspecified in C++, and both
    // recursive build() calls mutate `next_id` as a side effect, so leaving
    // it to the compiler would make operator ID assignment
    // non-deterministic across the two subtrees.
    std::unique_ptr<PhysicalOperator> left =
        build(join->left(), store, child_pass_read_limit_bytes, next_id, nvtx_enabled,
              build_side_budget_bytes, spill_directory, max_distinct_keys);
    std::unique_ptr<PhysicalOperator> right =
        build(join->right(), store, child_pass_read_limit_bytes, next_id, nvtx_enabled,
              build_side_budget_bytes, spill_directory, max_distinct_keys);
    return instrument(std::make_unique<HashJoinOperator>(
        next_id++, std::move(left), std::move(right), join->left_key_index(), join->right_key_index(),
        std::make_shared<const Schema>(join->output_schema()), partition_count, spill_directory,
        join->join_type()));
  }
  if (const auto* filter = dynamic_cast<const FilterNode*>(node.get())) {
    return instrument(std::make_unique<FilterOperator>(
        next_id++,
        build(filter->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled, build_side_budget_bytes,
              spill_directory, max_distinct_keys),
        filter->predicate()));
  }
  if (const auto* projection = dynamic_cast<const ProjectionNode*>(node.get())) {
    return instrument(std::make_unique<ProjectionOperator>(
        next_id++,
        build(projection->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled,
              build_side_budget_bytes, spill_directory, max_distinct_keys),
        projection->items()));
  }
  if (const auto* hash_aggregate = dynamic_cast<const HashAggregateNode*>(node.get())) {
    std::unique_ptr<PhysicalOperator> child =
        build(hash_aggregate->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled,
              build_side_budget_bytes, spill_directory, max_distinct_keys);
    // 0 means "unset in config" -- HashAggregateOperator's own default
    // constructor argument (kDefaultMaxDistinctKeys) only applies when the
    // parameter is omitted entirely, so an explicit 0 has to be resolved
    // here instead of just forwarded.
    if (max_distinct_keys != 0) {
      return instrument(std::make_unique<HashAggregateOperator>(
          next_id++, std::move(child), hash_aggregate->group_by(), hash_aggregate->aggregates(),
          static_cast<cudf::size_type>(max_distinct_keys)));
    }
    return instrument(std::make_unique<HashAggregateOperator>(
        next_id++, std::move(child), hash_aggregate->group_by(), hash_aggregate->aggregates()));
  }
  if (const auto* scalar_aggregate = dynamic_cast<const ScalarAggregateNode*>(node.get())) {
    return instrument(std::make_unique<ScalarAggregateOperator>(
        next_id++,
        build(scalar_aggregate->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled,
              build_side_budget_bytes, spill_directory, max_distinct_keys),
        scalar_aggregate->aggregates()));
  }
  if (const auto* sort = dynamic_cast<const SortNode*>(node.get())) {
    return instrument(std::make_unique<SortOperator>(
        next_id++,
        build(sort->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled, build_side_budget_bytes,
              spill_directory, max_distinct_keys),
        sort->keys()));
  }
  if (const auto* limit = dynamic_cast<const LimitNode*>(node.get())) {
    // ORDER BY ... LIMIT N: fuse into a single SortOperator instead of a
    // separate Sort+Limit pair. Unfused, SortOperator::next() gathers every
    // sorted row (the whole result set) and then LimitOperator immediately
    // slices that down to the first N and discards the rest -- pure waste
    // for the common top-N-rows query shape (e.g. TPC-H Q10). Passing the
    // limit into SortOperator lets its own gather materialize only the top N
    // rows to begin with. Only a direct Sort child qualifies -- anything
    // else between them (a Projection, say) isn't safe to skip over, so
    // falls through to the plain LimitOperator path below.
    if (const auto* sort = dynamic_cast<const SortNode*>(limit->child().get())) {
      return instrument(std::make_unique<SortOperator>(
          next_id++,
          build(sort->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled, build_side_budget_bytes,
                spill_directory, max_distinct_keys),
          sort->keys(), limit->limit()));
    }
    return instrument(std::make_unique<LimitOperator>(
        next_id++,
        build(limit->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled, build_side_budget_bytes,
              spill_directory, max_distinct_keys),
        limit->limit()));
  }
  if (const auto* arrow_result = dynamic_cast<const ArrowResultNode*>(node.get())) {
    return instrument(std::make_unique<ArrowResultOperator>(
        next_id++, build(arrow_result->child(), store, pass_read_limit_bytes, next_id, nvtx_enabled,
                         build_side_budget_bytes, spill_directory, max_distinct_keys)));
  }
  throw PlanningError("build_operator_tree: unrecognized physical plan node");
}

}  // namespace

std::unique_ptr<PhysicalOperator> build_operator_tree(const PhysicalPlanPtr& plan, ObjectStore& store,
                                                      std::size_t pass_read_limit_bytes, bool nvtx_enabled,
                                                      std::size_t build_side_budget_bytes,
                                                      const std::string& spill_directory,
                                                      std::uint64_t max_distinct_keys) {
  OperatorId next_id = 1;
  return build(plan, store, pass_read_limit_bytes, next_id, nvtx_enabled, build_side_budget_bytes,
               spill_directory, max_distinct_keys);
}

}  // namespace kernellake
