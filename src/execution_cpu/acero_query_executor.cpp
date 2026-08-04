#include "kernellake/execution_cpu/acero_query_executor.hpp"

#include <arrow/acero/api.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/compute/initialize.h>
#include <fmt/format.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <chrono>
#include <mutex>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_cpu/expression_compiler_cpu.hpp"
#include "kernellake/types/arrow_adapter.hpp"

namespace kernellake {

namespace {

// Arrow Compute's kernel functions (sum, hash_sum, cast, greater_equal,
// sort_indices, ...) do not self-register at static-init time when linked
// statically -- arrow::compute::Initialize() must be called once, first, to
// populate the global FunctionRegistry (see arrow/compute/initialize.h).
// std::call_once makes this safe to call from every execute_physical_plan_
// cpu() invocation without redoing the work (or racing) once a future
// concurrent caller, e.g. a Flight SQL server, exists.
void ensure_compute_initialized() {
  static std::once_flag once;
  std::call_once(once, [] {
    const arrow::Status status = arrow::compute::Initialize();
    if (!status.ok()) {
      throw ExecutionError(
          fmt::format("failed to initialize the Arrow Compute function registry: {}", status.ToString()));
    }
  });
}

// Reads every fragment of `scan` into one combined arrow::Table, respecting
// the row-group pruning decisions already computed by the physical planner
// (PhysicalFileFragment::selected_row_groups) and the already-narrowed
// column list (ParquetScanNode::columns()) -- reusing the exact same
// pruning/column-pruning work the GPU path's ParquetScanOperator consumes,
// just with a different reader underneath. Deliberately simpler than the
// GPU path's bounded-memory, pass-based streaming (see ParquetScanOperator):
// this reads each fragment's selected row groups fully into memory, so this
// backend's memory footprint scales with input size rather than being
// bounded -- a real, documented MVP simplification for this phase.
std::shared_ptr<arrow::Table> read_scan_table(const ParquetScanNode& scan, ObjectStore& store) {
  const std::vector<std::string>& columns = scan.columns();
  std::vector<std::shared_ptr<arrow::RecordBatch>> all_batches;
  std::shared_ptr<arrow::Schema> table_schema;

  for (const PhysicalFileFragment& fragment : scan.fragments()) {
    // store.open() itself already throws StorageError on failure to open;
    // Open() below can still throw parquet::ParquetException separately for
    // malformed/corrupt Parquet content once bytes are flowing, same as
    // src/io/parquet_metadata.cpp's inspect_parquet_file().
    std::unique_ptr<parquet::ParquetFileReader> raw_reader;
    try {
      raw_reader = parquet::ParquetFileReader::Open(store.open(fragment.file)->as_arrow_file());
    } catch (const parquet::ParquetException& e) {
      throw StorageError(
          fmt::format("failed to open Parquet file '{}' for CPU scan: {}", fragment.file.value(), e.what()));
    }

    arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> reader_result =
        parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(raw_reader));
    if (!reader_result.ok()) {
      throw StorageError(fmt::format("failed to open Parquet file '{}' for CPU scan: {}",
                                     fragment.file.value(), reader_result.status().ToString()));
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);

    std::shared_ptr<arrow::Schema> file_schema;
    const arrow::Status schema_status = reader->GetSchema(&file_schema);
    if (!schema_status.ok()) {
      throw StorageError(
          fmt::format("failed to read Parquet schema for CPU scan: {}", schema_status.ToString()));
    }

    std::vector<int> column_indices;
    column_indices.reserve(columns.size());
    for (const std::string& name : columns) {
      const int index = file_schema->GetFieldIndex(name);
      if (index < 0) {
        throw StorageError(
            fmt::format("column '{}' not found in Parquet file '{}'", name, fragment.file.value()));
      }
      column_indices.push_back(index);
    }

    arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> rb_reader_result =
        reader->GetRecordBatchReader(fragment.selected_row_groups, column_indices);
    if (!rb_reader_result.ok()) {
      throw StorageError(fmt::format("failed to build a record batch reader for CPU scan: {}",
                                     rb_reader_result.status().ToString()));
    }
    std::unique_ptr<arrow::RecordBatchReader> rb_reader = std::move(*rb_reader_result);
    if (table_schema == nullptr) {
      table_schema = rb_reader->schema();
    }

    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      const arrow::Status next_status = rb_reader->ReadNext(&batch);
      if (!next_status.ok()) {
        throw StorageError(
            fmt::format("failed reading a Parquet batch for CPU scan: {}", next_status.ToString()));
      }
      if (batch == nullptr) {
        break;
      }
      all_batches.push_back(std::move(batch));
    }
  }

  // No fragments at all (every file/row-group was pruned away): fall back
  // to the scan's own declared (narrowed) schema so the result is a real,
  // correctly-typed empty table rather than a null schema.
  if (table_schema == nullptr) {
    table_schema = to_arrow_schema(scan.output_schema());
  }

  arrow::Result<std::shared_ptr<arrow::Table>> table_result =
      arrow::Table::FromRecordBatches(table_schema, all_batches);
  if (!table_result.ok()) {
    throw StorageError(
        fmt::format("failed to assemble the CPU scan's result table: {}", table_result.status().ToString()));
  }
  return *table_result;
}

// Resolves a NamedExpression that must be a plain column reference (a
// GROUP BY key or ORDER BY key) to its field name -- Acero's
// AggregateNodeOptions::keys and SortKey both take a FieldRef, which this
// backend always resolves by name (not by the GPU path's index-based
// remapping: Acero's Declaration tree is built purely out of real Arrow
// schemas with real field names at every stage, so there is no equivalent
// of the GPU path's remap_columns()/find_scan_schema() problem to solve
// here at all). Throws PlanningError for anything else (e.g. a computed
// CASE-derived GROUP BY alias) -- not yet supported by this backend.
const std::string& require_plain_column_name(const ExpressionPtr& expr, const char* context) {
  const auto* column = dynamic_cast<const ColumnExpression*>(expr.get());
  if (column == nullptr) {
    throw PlanningError(
        fmt::format("{} by a computed expression is not yet supported by the CPU execution backend "
                    "(only a plain column reference is) -- see docs/ARCHITECTURE.md",
                    context));
  }
  return column->name();
}

std::shared_ptr<arrow::compute::FunctionOptions> count_options() {
  return std::make_shared<arrow::compute::CountOptions>(arrow::compute::CountOptions::ONLY_VALID);
}

// Acero's AggregateNodeOptions can only target an already-existing column
// by FieldRef -- it has no way to evaluate an arbitrary expression itself
// (unlike GROUP BY/ORDER BY keys, which really are always meant to be a
// plain column in this engine's own grammar). A computed aggregate
// argument (e.g. `SUM(l_extendedprice * l_discount)`, both of TPC-H's own
// Q1/Q6) therefore needs a real column computed for it *before* the
// aggregate node -- via an implicit "project" Declaration inserted between
// the aggregate's child and the aggregate node itself, gathering every
// GROUP BY key (pass-through) and aggregate argument (pass-through if
// already a plain column, computed under a synthetic name otherwise) that
// the aggregate node will need to reference afterward. `project_names`
// staying empty means nothing needs it (e.g. bare COUNT(*), which has no
// argument() column at all) -- callers skip inserting the projection node
// entirely in that case, unchanged from before this existed.
struct AggregateInputPlan {
  std::vector<arrow::compute::Expression> project_expressions;
  std::vector<std::string> project_names;
  int next_synthetic_id = 0;
};

void ensure_column_projected(AggregateInputPlan& plan, const std::string& name) {
  if (std::find(plan.project_names.begin(), plan.project_names.end(), name) != plan.project_names.end()) {
    return;
  }
  plan.project_expressions.push_back(arrow::compute::field_ref(name));
  plan.project_names.push_back(name);
}

std::string resolve_aggregate_target(AggregateInputPlan& plan, const ExpressionPtr& expr) {
  if (const auto* column = dynamic_cast<const ColumnExpression*>(expr.get())) {
    ensure_column_projected(plan, column->name());
    return column->name();
  }
  const std::string synthetic_name = fmt::format("__kernellake_agg_arg_{}", plan.next_synthetic_id++);
  plan.project_expressions.push_back(compile_expression_cpu(*expr));
  plan.project_names.push_back(synthetic_name);
  return synthetic_name;
}

// `grouped` selects the "hash_"-prefixed Arrow Compute aggregate functions
// (used when there is a non-empty `keys` list) vs. the plain scalar
// versions (used for a whole-input, no-GROUP-BY aggregate) -- Acero
// requires this distinction explicitly (AggregateNodeOptions's own docs:
// "If keys is non-empty then each aggregate must be a HashAggregate
// function").
arrow::compute::Aggregate translate_aggregate(AggregateInputPlan& plan, const AggregateExpression& aggregate,
                                              const std::string& output_name, bool grouped) {
  std::vector<arrow::FieldRef> target;
  if (aggregate.argument() != nullptr) {
    target.emplace_back(resolve_aggregate_target(plan, aggregate.argument()));
  }
  switch (aggregate.function()) {
    case AggregateFunction::Sum:
      return arrow::compute::Aggregate{grouped ? "hash_sum" : "sum", nullptr, target, output_name};
    case AggregateFunction::Count:
      return arrow::compute::Aggregate{grouped ? "hash_count" : "count", count_options(), target,
                                       output_name};
    case AggregateFunction::CountStar:
      // "count"/"hash_count" require a real value column (arity 1/2); Acero
      // has dedicated arity-0/1 "count_all"/"hash_count_all" functions
      // specifically for COUNT(*), which needs no column input at all --
      // `target` is empty here (COUNT(*) has no argument() column, matching
      // the GPU path's convention), which is exactly what these two
      // functions expect (any FunctionOptions here would be rejected, since
      // there is no value column whose nulls could be filtered).
      return arrow::compute::Aggregate{grouped ? "hash_count_all" : "count_all", nullptr, target,
                                       output_name};
    case AggregateFunction::Min:
      return arrow::compute::Aggregate{grouped ? "hash_min" : "min", nullptr, target, output_name};
    case AggregateFunction::Max:
      return arrow::compute::Aggregate{grouped ? "hash_max" : "max", nullptr, target, output_name};
    case AggregateFunction::Avg:
      return arrow::compute::Aggregate{grouped ? "hash_mean" : "mean", nullptr, target, output_name};
  }
  throw ExecutionError("unreachable: unknown AggregateFunction in CPU execution backend");
}

arrow::acero::Declaration translate(const PhysicalPlanPtr& node, ObjectStore& store) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(node.get())) {
    return arrow::acero::Declaration{"table_source",
                                     arrow::acero::TableSourceNodeOptions{read_scan_table(*scan, store)}};
  }
  if (const auto* filter = dynamic_cast<const FilterNode*>(node.get())) {
    return arrow::acero::Declaration{
        "filter",
        {translate(filter->child(), store)},
        arrow::acero::FilterNodeOptions{compile_expression_cpu(*filter->predicate())}};
  }
  if (const auto* projection = dynamic_cast<const ProjectionNode*>(node.get())) {
    std::vector<arrow::compute::Expression> expressions;
    std::vector<std::string> names;
    expressions.reserve(projection->items().size());
    names.reserve(projection->items().size());
    for (const NamedExpression& item : projection->items()) {
      expressions.push_back(compile_expression_cpu(*item.expr));
      names.push_back(item.name);
    }
    return arrow::acero::Declaration{
        "project",
        {translate(projection->child(), store)},
        arrow::acero::ProjectNodeOptions{std::move(expressions), std::move(names)}};
  }
  if (const auto* hash_aggregate = dynamic_cast<const HashAggregateNode*>(node.get())) {
    AggregateInputPlan plan;
    std::vector<arrow::FieldRef> keys;
    keys.reserve(hash_aggregate->group_by().size());
    for (const NamedExpression& item : hash_aggregate->group_by()) {
      const std::string& key_name = require_plain_column_name(item.expr, "GROUP BY");
      ensure_column_projected(plan, key_name);
      keys.emplace_back(key_name);
    }
    std::vector<arrow::compute::Aggregate> aggregates;
    aggregates.reserve(hash_aggregate->aggregates().size());
    for (const NamedExpression& item : hash_aggregate->aggregates()) {
      const auto* aggregate = dynamic_cast<const AggregateExpression*>(item.expr.get());
      if (aggregate == nullptr) {
        throw ExecutionError(
            fmt::format("HashAggregateNode item '{}' is not an AggregateExpression", item.name));
      }
      aggregates.push_back(translate_aggregate(plan, *aggregate, item.name, /*grouped=*/true));
    }
    arrow::acero::Declaration child = translate(hash_aggregate->child(), store);
    arrow::acero::Declaration input =
        plan.project_names.empty()
            ? std::move(child)
            : arrow::acero::Declaration{"project",
                                        {std::move(child)},
                                        arrow::acero::ProjectNodeOptions{std::move(plan.project_expressions),
                                                                         std::move(plan.project_names)}};
    return arrow::acero::Declaration{
        "aggregate",
        {std::move(input)},
        arrow::acero::AggregateNodeOptions{std::move(aggregates), std::move(keys)}};
  }
  if (const auto* scalar_aggregate = dynamic_cast<const ScalarAggregateNode*>(node.get())) {
    AggregateInputPlan plan;
    std::vector<arrow::compute::Aggregate> aggregates;
    aggregates.reserve(scalar_aggregate->aggregates().size());
    for (const NamedExpression& item : scalar_aggregate->aggregates()) {
      const auto* aggregate = dynamic_cast<const AggregateExpression*>(item.expr.get());
      if (aggregate == nullptr) {
        throw ExecutionError(
            fmt::format("ScalarAggregateNode item '{}' is not an AggregateExpression", item.name));
      }
      aggregates.push_back(translate_aggregate(plan, *aggregate, item.name, /*grouped=*/false));
    }
    arrow::acero::Declaration child = translate(scalar_aggregate->child(), store);
    arrow::acero::Declaration input =
        plan.project_names.empty()
            ? std::move(child)
            : arrow::acero::Declaration{"project",
                                        {std::move(child)},
                                        arrow::acero::ProjectNodeOptions{std::move(plan.project_expressions),
                                                                         std::move(plan.project_names)}};
    return arrow::acero::Declaration{
        "aggregate", {std::move(input)}, arrow::acero::AggregateNodeOptions{std::move(aggregates), {}}};
  }
  if (const auto* sort = dynamic_cast<const SortNode*>(node.get())) {
    std::vector<arrow::compute::SortKey> keys;
    keys.reserve(sort->keys().size());
    for (const LogicalSort::Key& key : sort->keys()) {
      const std::string& name = require_plain_column_name(key.expr, "ORDER BY");
      const arrow::compute::SortOrder order =
          key.ascending ? arrow::compute::SortOrder::Ascending : arrow::compute::SortOrder::Descending;
      // Matches the GPU SortOperator's convention (and standard SQL
      // behavior): NULLs sort last in ASC order, first in DESC order.
      const arrow::compute::NullPlacement null_placement =
          key.ascending ? arrow::compute::NullPlacement::AtEnd : arrow::compute::NullPlacement::AtStart;
      keys.emplace_back(arrow::FieldRef(name), order, null_placement);
    }
    return arrow::acero::Declaration{
        "order_by",
        {translate(sort->child(), store)},
        arrow::acero::OrderByNodeOptions{arrow::compute::Ordering{std::move(keys)}}};
  }
  if (const auto* limit = dynamic_cast<const LimitNode*>(node.get())) {
    return arrow::acero::Declaration{"fetch",
                                     {translate(limit->child(), store)},
                                     arrow::acero::FetchNodeOptions{/*offset=*/0, limit->limit()}};
  }
  if (const auto* arrow_result = dynamic_cast<const ArrowResultNode*>(node.get())) {
    return translate(arrow_result->child(), store);
  }
  throw PlanningError(
      fmt::format("physical plan node '{}' is not yet supported by the CPU execution backend (e.g. HashJoin "
                  "isn't yet) -- "
                  "see docs/ARCHITECTURE.md",
                  node->node_name()));
}

}  // namespace

CpuQueryExecutionResult execute_physical_plan_cpu(const PhysicalPlanPtr& physical, ObjectStore& store) {
  ensure_compute_initialized();
  const auto start = std::chrono::steady_clock::now();

  const arrow::acero::Declaration declaration = translate(physical, store);
  const arrow::Result<std::shared_ptr<arrow::Table>> table_result =
      arrow::acero::DeclarationToTable(declaration);
  if (!table_result.ok()) {
    throw ExecutionError(fmt::format("CPU execution backend failed: {}", table_result.status().ToString()));
  }

  CpuQueryExecutionResult result;
  result.table = *table_result;
  result.execution_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace kernellake
