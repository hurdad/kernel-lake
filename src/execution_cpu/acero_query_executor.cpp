#include "kernellake/execution_cpu/acero_query_executor.hpp"

#include <arrow/acero/api.h>
#include <arrow/array/util.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/compute/initialize.h>
#include <arrow/util/iterator.h>
#include <fmt/format.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

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

// parquet::arrow::FileReader::GetRecordBatchReader()'s own docs are explicit:
// "FileReaders must outlive their RecordBatchReaders." Bundled together here
// so nothing can accidentally drop the FileReader while its RecordBatchReader
// is still in use (a real use-after-free bug this fix's first version hit:
// an earlier draft returned only the RecordBatchReader from a helper
// function, letting the FileReader local variable be destroyed at that
// function's return -- crashed every CPU-backend test with a segfault).
struct OpenFragment {
  std::unique_ptr<parquet::arrow::FileReader> file_reader;
  std::unique_ptr<arrow::RecordBatchReader> batch_reader;
};

// Opens one fragment's selected row groups/narrowed columns -- store.open()
// itself already throws StorageError on failure to open; Open() below can
// still throw parquet::ParquetException separately for malformed/corrupt
// Parquet content once bytes are flowing, same as
// src/io/parquet_metadata.cpp's inspect_parquet_file().
OpenFragment open_fragment_reader(const PhysicalFileFragment& fragment,
                                  const std::vector<std::string>& columns, ObjectStore& store) {
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
    throw StorageError(fmt::format("failed to open Parquet file '{}' for CPU scan: {}", fragment.file.value(),
                                   reader_result.status().ToString()));
  }
  OpenFragment opened;
  opened.file_reader = std::move(*reader_result);

  std::shared_ptr<arrow::Schema> file_schema;
  const arrow::Status schema_status = opened.file_reader->GetSchema(&file_schema);
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
      opened.file_reader->GetRecordBatchReader(fragment.selected_row_groups, column_indices);
  if (!rb_reader_result.ok()) {
    throw StorageError(fmt::format("failed to build a record batch reader for CPU scan: {}",
                                   rb_reader_result.status().ToString()));
  }
  opened.batch_reader = std::move(*rb_reader_result);
  return opened;
}

// Builds a length-`length` constant-value array for one Hive partition
// column, reusing expression_compiler_cpu.hpp's literal-to-Arrow-scalar
// conversion (the same type mapping the AST literal-node path already uses)
// rather than duplicating a second DataType-to-Arrow-scalar table here.
std::shared_ptr<arrow::Array> make_partition_constant_array(const LiteralStorage& value, const DataType& type,
                                                            std::int64_t length) {
  const LiteralExpression literal(value, type);
  const arrow::Datum datum = literal_to_arrow_datum(literal);
  const arrow::Result<std::shared_ptr<arrow::Array>> result =
      arrow::MakeArrayFromScalar(*datum.scalar(), length);
  if (!result.ok()) {
    throw StorageError(
        fmt::format("failed to materialize Hive partition column constant: {}", result.status().ToString()));
  }
  return *result;
}

// Appends one constant-value column per entry in `partition_columns` (using
// the parallel `partition_values`, see PhysicalFileFragment's own doc
// comment) to `batch` -- these columns are derived from the fragment's file
// location (e.g. "region=US/part-0.parquet"), never physically present in
// the Parquet file itself, so open_fragment_reader() never reads them; they
// only exist from here on. A no-op (returns `batch` unchanged) for a plain,
// non-partitioned scan.
std::shared_ptr<arrow::RecordBatch> append_partition_columns(
    std::shared_ptr<arrow::RecordBatch> batch, const std::vector<PartitionColumn>& partition_columns,
    const std::vector<LiteralStorage>& partition_values) {
  std::shared_ptr<arrow::RecordBatch> result = std::move(batch);
  for (std::size_t i = 0; i < partition_columns.size(); ++i) {
    const std::shared_ptr<arrow::Array> column =
        make_partition_constant_array(partition_values[i], partition_columns[i].type, result->num_rows());
    const std::shared_ptr<arrow::Field> field =
        arrow::field(partition_columns[i].name, to_arrow_type(partition_columns[i].type), /*nullable=*/false);
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> appended =
        result->AddColumn(result->num_columns(), field, column);
    if (!appended.ok()) {
      throw StorageError(fmt::format("failed to append Hive partition column '{}': {}",
                                     partition_columns[i].name, appended.status().ToString()));
    }
    result = *appended;
  }
  return result;
}

// Iteration state for make_streaming_scan_reader()'s lazy, cross-fragment
// RecordBatchReader -- shared_ptr since Acero's "record_batch_reader_source"
// node (see its own doc comment) runs each ReadNext() call as a task on its
// own I/O thread pool, not synchronously inline like the rest of this
// backend's translate()/execute_physical_plan_cpu() -- the reader itself
// (and this state) must outlive that asynchronous execution, not just the
// synchronous scope that constructs it. `mutex` guards every mutable field
// below: nothing in Acero's own documented contract for
// RecordBatchReaderSourceNodeOptions ("each iteration...run on a new
// thread task") promises those tasks are serialized rather than pipelined,
// and this state's mutation (advancing `next_fragment_index`, replacing
// `current`) is not safe under concurrent access either way -- cheap
// insurance against a scheduling assumption this code doesn't actually
// need to rely on, not a response to an observed race.
struct ScanIterationState {
  std::mutex mutex;
  const ParquetScanNode* scan;
  ObjectStore* store;
  std::size_t next_fragment_index = 0;
  OpenFragment current;
};

// Builds one RecordBatchReader spanning every fragment of `scan`, opening
// and reading each fragment lazily -- one at a time, one batch at a time --
// instead of read_scan_table()'s previous approach of draining every
// fragment's every batch into one in-memory vector before Acero's pipeline
// even starts. Handed to Acero via RecordBatchReaderSourceNodeOptions
// ("record_batch_reader_source"), this backend's memory footprint no longer
// scales with total input size the way TableSourceNodeOptions's
// full-materialization did -- only whatever's actually in flight through
// the rest of the pipeline at once, matching (in spirit, not exact
// mechanism) the GPU path's own pass-based, bounded-memory scan.
std::shared_ptr<arrow::RecordBatchReader> make_streaming_scan_reader(const ParquetScanNode& scan,
                                                                     ObjectStore& store) {
  const std::vector<PhysicalFileFragment>& fragments = scan.fragments();

  // No fragments at all (every file/row-group was pruned away): the scan's
  // own declared (narrowed) schema is the only schema available, and there
  // are zero batches to stream -- a real, correctly-typed empty result,
  // not a null schema.
  if (fragments.empty()) {
    arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> empty_reader_result =
        arrow::RecordBatchReader::Make({}, to_arrow_schema(scan.output_schema()));
    if (!empty_reader_result.ok()) {
      throw StorageError(fmt::format("failed to build an empty CPU scan reader: {}",
                                     empty_reader_result.status().ToString()));
    }
    return *empty_reader_result;
  }

  // Opening a fragment's reader only reads Parquet *metadata* (schema, row
  // group offsets) -- ReadNext() is what actually reads row data, and nothing
  // calls it here. Eagerly opening exactly the first fragment (synchronously,
  // before this function returns) to determine the real schema matches
  // read_scan_table()'s previous behavior exactly (using the first fragment's
  // actual reader-derived schema, not scan.output_schema(), which can differ
  // in nullability once real Parquet file metadata is consulted) while still
  // reading zero rows of data up front.
  auto state = std::make_shared<ScanIterationState>();
  state->scan = &scan;
  state->store = &store;
  state->current = open_fragment_reader(fragments.front(), scan.columns(), store);
  state->next_fragment_index = 1;
  std::shared_ptr<arrow::Schema> schema = state->current.batch_reader->schema();
  // Hive partition columns (see LogicalScan::partition_columns()) are never
  // physically present in the file, so open_fragment_reader() never reads
  // them -- append their fields here (always non-nullable: a partition
  // value is either present from the file's own path, or the file wouldn't
  // have matched this scan's fragment list at all) so the reader's declared
  // schema matches what append_partition_columns() below actually adds to
  // every batch.
  if (!scan.partition_columns().empty()) {
    arrow::FieldVector fields = schema->fields();
    for (const PartitionColumn& column : scan.partition_columns()) {
      fields.push_back(arrow::field(column.name, to_arrow_type(column.type), /*nullable=*/false));
    }
    schema = arrow::schema(std::move(fields));
  }

  arrow::Iterator<std::shared_ptr<arrow::RecordBatch>> iterator =
      arrow::MakeFunctionIterator([state]() -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
        // Every exception this callback's own call chain can throw
        // (StorageError from open_fragment_reader(), any other
        // KernelLakeError) must be caught *here*, inside this single
        // invocation -- it runs on Acero's I/O thread pool (see this
        // function's own doc comment), where an uncaught C++ exception
        // would escape a thread-pool worker rather than propagate back to
        // execute_physical_plan_cpu()'s try/catch, crashing the process
        // instead of failing the query cleanly.
        const std::lock_guard<std::mutex> lock(state->mutex);
        try {
          while (true) {
            if (state->current.batch_reader == nullptr) {
              const std::vector<PhysicalFileFragment>& fragments = state->scan->fragments();
              if (state->next_fragment_index >= fragments.size()) {
                return std::shared_ptr<arrow::RecordBatch>();  // end of stream
              }
              state->current = open_fragment_reader(fragments[state->next_fragment_index],
                                                    state->scan->columns(), *state->store);
              ++state->next_fragment_index;
            }
            std::shared_ptr<arrow::RecordBatch> batch;
            ARROW_RETURN_NOT_OK(state->current.batch_reader->ReadNext(&batch));
            if (batch == nullptr) {
              state->current = OpenFragment{};
              continue;  // this fragment is exhausted; try the next one
            }
            if (!state->scan->partition_columns().empty()) {
              const std::vector<PhysicalFileFragment>& fragments = state->scan->fragments();
              const PhysicalFileFragment& current_fragment = fragments[state->next_fragment_index - 1];
              batch = append_partition_columns(std::move(batch), state->scan->partition_columns(),
                                               current_fragment.partition_values);
            }
            return batch;
          }
        } catch (const KernelLakeError& e) {
          return arrow::Status::IOError(e.what());
        } catch (const std::exception& e) {
          return arrow::Status::UnknownError(e.what());
        }
      });

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> reader_result =
      arrow::RecordBatchReader::MakeFromIterator(std::move(iterator), schema);
  if (!reader_result.ok()) {
    throw StorageError(fmt::format("failed to build the CPU scan's streaming reader: {}",
                                   reader_result.status().ToString()));
  }
  return *reader_result;
}

// Resolves a NamedExpression that must be a plain column reference (a
// GROUP BY key or ORDER BY key) to its column_index() -- Acero's
// AggregateNodeOptions::keys and SortKey both take a FieldRef, referenced
// here *by position* (see compile_expression_cpu's identical reasoning):
// a JOIN's combined physical schema can have two same-named columns from
// opposite sides, which a by-name FieldRef can't disambiguate (Acero
// itself throws "Multiple matches" for it) even though column_index()
// already unambiguously identifies the right one. Throws PlanningError for
// anything else (e.g. a computed CASE-derived GROUP BY alias) -- not yet
// supported by this backend.
std::size_t require_plain_column_index(const ExpressionPtr& expr, const char* context) {
  const auto* column = dynamic_cast<const ColumnExpression*>(expr.get());
  if (column == nullptr) {
    throw PlanningError(
        fmt::format("{} by a computed expression is not yet supported by the CPU execution backend "
                    "(only a plain column reference is) -- see docs/ARCHITECTURE.md",
                    context));
  }
  return column->column_index();
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
  // Keyed by source *position* rather than name, so a repeated reference
  // to the same original column (e.g. used as both a GROUP BY key and an
  // aggregate argument) reuses one projected slot instead of projecting it
  // twice.
  std::unordered_map<std::size_t, std::string> projected_index_to_name;
  // Every *original* bare name already claimed by a projected column, so a
  // genuine collision -- two different source positions sharing a bare
  // name, only possible after a JOIN -- gets a synthetic name instead of
  // colliding. The common (non-colliding) case keeps its original name
  // unchanged: HashAggregateNode's own group_by/aggregate output field
  // names are decided at the logical layer (logical_planner.cpp) and
  // assumed to already match 1:1 with whatever Acero actually names its
  // output columns once the redundant final re-projection is elided (see
  // is_identity_projection()'s comment in physical_planner.cpp) --
  // renaming every pass-through column unconditionally broke that
  // assumption even for a plain, non-JOIN GROUP BY (confirmed by a real
  // test failure before this comment was added).
  //
  // Residual gap, not fixed here: if two GROUP BY keys sharing a bare name
  // from opposite JOIN sides are *both* selected together (e.g. `SELECT
  // l.x, r.x, COUNT(*) ... GROUP BY l.x, r.x`), the second one's synthetic
  // name here won't match its own HashAggregateNode::output_schema() field
  // name (which is unaware of any physical collision) -- a real caller
  // would hit a GetColumnByName() mismatch downstream. Aggregate
  // arguments and any non-aggregate SELECT (both handled elsewhere) are
  // unaffected. Closing this would need Field-level qualification, out of
  // scope here.
  std::unordered_set<std::string> claimed_names;
  int next_synthetic_id = 0;
};

std::string next_synthetic_name(AggregateInputPlan& plan) {
  return fmt::format("__kernellake_agg_input_{}", plan.next_synthetic_id++);
}

// Projects `column` (by position, see compile_expression_cpu's identical
// reasoning) under its own original name, unless that name is already
// claimed by a *different* source position projected earlier in this same
// plan, in which case it gets a synthetic name instead -- see
// AggregateInputPlan::claimed_names's own comment for why.
const std::string& ensure_column_projected(AggregateInputPlan& plan, const ColumnExpression& column) {
  const auto it = plan.projected_index_to_name.find(column.column_index());
  if (it != plan.projected_index_to_name.end()) {
    return it->second;
  }
  std::string name = plan.claimed_names.find(column.name()) == plan.claimed_names.end()
                         ? column.name()
                         : next_synthetic_name(plan);
  plan.claimed_names.insert(name);
  plan.project_expressions.push_back(arrow::compute::field_ref(static_cast<int>(column.column_index())));
  plan.project_names.push_back(name);
  return plan.projected_index_to_name.emplace(column.column_index(), std::move(name)).first->second;
}

std::string resolve_aggregate_target(AggregateInputPlan& plan, const ExpressionPtr& expr) {
  if (const auto* column = dynamic_cast<const ColumnExpression*>(expr.get())) {
    return ensure_column_projected(plan, *column);
  }
  const std::string synthetic_name = next_synthetic_name(plan);
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
    // Most scans read through the caller's own default `store`; a scan
    // resolved with per-table vended credentials (see
    // ParquetScanNode::owned_store()'s own doc comment -- currently only
    // Unity Catalog's S3/GCS/Azure vended credentials) must read through
    // that store instead, or it would silently fall back to whatever
    // static access the engine's own default config happens to have (or
    // fail outright, if it has none).
    ObjectStore& effective_store = scan->owned_store() != nullptr ? *scan->owned_store() : store;
    return arrow::acero::Declaration{
        "record_batch_reader_source",
        arrow::acero::RecordBatchReaderSourceNodeOptions{make_streaming_scan_reader(*scan, effective_store)}};
  }
  // Acero's own "hashjoin" node (HashJoinNodeOptions) implements exactly the
  // two-table INNER/LEFT OUTER equi-join HashJoinNode describes; output_all
  // defaults to true (all columns from both sides, left fields then right),
  // matching HashJoinNode::build_schema()'s convention exactly (including
  // its own right-side nullable widening for LEFT OUTER, which Acero's own
  // LEFT_OUTER null-extension naturally produces), so no left_output/
  // right_output list needs to be built here.
  if (const auto* hash_join = dynamic_cast<const HashJoinNode*>(node.get())) {
    // By position (see compile_expression_cpu's identical reasoning): if
    // `hash_join->left()`/`right()` is itself a nested HashJoinNode (a 3+
    // -way join), its own combined schema can already have two same-named
    // columns from its own two children, making a by-name FieldRef here
    // ambiguous too, not just at the outer join.
    const arrow::acero::JoinType acero_join_type = hash_join->join_type() == JoinType::LeftOuter
                                                       ? arrow::acero::JoinType::LEFT_OUTER
                                                       : arrow::acero::JoinType::INNER;
    return arrow::acero::Declaration{
        "hashjoin",
        {translate(hash_join->left(), store), translate(hash_join->right(), store)},
        arrow::acero::HashJoinNodeOptions{acero_join_type,
                                          {arrow::FieldRef(static_cast<int>(hash_join->left_key_index()))},
                                          {arrow::FieldRef(static_cast<int>(hash_join->right_key_index()))}}};
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
      const auto* column = dynamic_cast<const ColumnExpression*>(item.expr.get());
      if (column != nullptr) {
        keys.emplace_back(ensure_column_projected(plan, *column));
        continue;
      }
      // A computed GROUP BY key (e.g. an EXTRACT- or CASE-derived alias)
      // has no existing column to reference -- project it under its own
      // logical name (item.name) rather than resolve_aggregate_target's
      // throwaway synthetic name below: unlike an aggregate argument (whose
      // projected name is a pure implementation detail, since
      // translate_aggregate always passes its own explicit output_name),
      // Acero's `keys` FieldRef has no separate output-name mechanism --
      // whatever name this key is projected under becomes the actual output
      // column name, which must match HashAggregateNode::output_schema()'s
      // own field name (item.name) for the same reason
      // AggregateInputPlan::claimed_names's comment already documents for
      // plain pass-through columns.
      if (plan.claimed_names.find(item.name) != plan.claimed_names.end()) {
        throw PlanningError(fmt::format(
            "GROUP BY key '{}' collides with another projected column name -- not supported by the "
            "CPU execution backend",
            item.name));
      }
      plan.claimed_names.insert(item.name);
      plan.project_expressions.push_back(compile_expression_cpu(*item.expr));
      plan.project_names.push_back(item.name);
      keys.emplace_back(item.name);
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
      const std::size_t index = require_plain_column_index(key.expr, "ORDER BY");
      const arrow::compute::SortOrder order =
          key.ascending ? arrow::compute::SortOrder::Ascending : arrow::compute::SortOrder::Descending;
      // Matches the GPU SortOperator's convention (and standard SQL
      // behavior): NULLs sort last in ASC order, first in DESC order.
      const arrow::compute::NullPlacement null_placement =
          key.ascending ? arrow::compute::NullPlacement::AtEnd : arrow::compute::NullPlacement::AtStart;
      keys.emplace_back(arrow::FieldRef(static_cast<int>(index)), order, null_placement);
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
  throw PlanningError(fmt::format(
      "physical plan node '{}' is not yet supported by the CPU execution backend -- see docs/ARCHITECTURE.md",
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
