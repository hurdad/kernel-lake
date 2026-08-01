#include "kernellake/io/physical_planner.hpp"

#include <algorithm>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/io/parquet_pruning.hpp"
#include "kernellake/storage/file_discovery.hpp"

namespace kernellake {

namespace {

// LogicalScan's own output_schema() always keeps every original column (the
// binder resolves every ColumnExpression in the whole query against that one
// full schema -- see docs/ARCHITECTURE.md and the comment on
// LogicalScan::required_columns()). convert_scan() below narrows the
// *physical* scan to only the columns actually referenced, in doing so
// reindexing them (a column no longer at its original ordinal position once
// earlier columns are dropped). Every expression built by the binder still
// carries the *original* index, so anything sitting above the scan --
// filter predicates, projection items, group-by keys, aggregate arguments --
// must have its ColumnExpression indices rewritten to match the narrowed
// scan's actual column order before it can run against the real batches the
// scan operator produces.
ExpressionPtr remap_columns(const ExpressionPtr& expr, const Schema& scan_schema) {
  if (const auto* column = dynamic_cast<const ColumnExpression*>(expr.get())) {
    const std::optional<std::size_t> index = scan_schema.find_field(column->name());
    if (!index) {
      throw PlanningError("physical planner: column '" + column->name() +
                          "' referenced above the scan but missing from its pruned column list "
                          "(internal error)");
    }
    return std::make_shared<ColumnExpression>(column->name(), *index, column->result_type());
  }
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(expr.get())) {
    return std::make_shared<BinaryExpression>(binary->op(), remap_columns(binary->left(), scan_schema),
                                              remap_columns(binary->right(), scan_schema),
                                              binary->result_type());
  }
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(expr.get())) {
    return std::make_shared<UnaryExpression>(unary->op(), remap_columns(unary->operand(), scan_schema),
                                             unary->result_type());
  }
  if (const auto* cast = dynamic_cast<const CastExpression*>(expr.get())) {
    return std::make_shared<CastExpression>(remap_columns(cast->operand(), scan_schema), cast->result_type());
  }
  if (const auto* between = dynamic_cast<const BetweenExpression*>(expr.get())) {
    return std::make_shared<BetweenExpression>(remap_columns(between->value(), scan_schema),
                                               remap_columns(between->lower(), scan_schema),
                                               remap_columns(between->upper(), scan_schema));
  }
  if (const auto* aggregate = dynamic_cast<const AggregateExpression*>(expr.get())) {
    ExpressionPtr argument =
        aggregate->argument() ? remap_columns(aggregate->argument(), scan_schema) : nullptr;
    return std::make_shared<AggregateExpression>(aggregate->function(), std::move(argument),
                                                 aggregate->result_type());
  }
  return expr;  // LiteralExpression: no column reference to remap.
}

std::vector<NamedExpression> remap_named(const std::vector<NamedExpression>& items,
                                         const Schema& scan_schema) {
  std::vector<NamedExpression> result;
  result.reserve(items.size());
  for (const NamedExpression& item : items) {
    result.push_back(NamedExpression{remap_columns(item.expr, scan_schema), item.name});
  }
  return result;
}

// Single-source queries only (no joins yet -- see docs/ROADMAP.md), so
// exactly one ParquetScanNode exists per physical plan; find it to recover
// the narrowed schema every expression above it must be remapped against.
const Schema* find_scan_schema(const PhysicalPlanNode& node) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(&node)) return &scan->output_schema();
  for (const PhysicalPlanPtr& child : node.children()) {
    if (const Schema* found = find_scan_schema(*child)) return found;
  }
  return nullptr;
}

PhysicalPlanPtr convert_scan(const LogicalScan& scan, ObjectStore& store) {
  const std::vector<ObjectInfo> files = discover_parquet_files(store, scan.source_paths());
  std::vector<FileMetadata> metadata;
  metadata.reserve(files.size());
  for (const ObjectInfo& file : files) metadata.push_back(inspect_parquet_file(store, file.uri));
  validate_schema_compatibility(metadata);

  std::vector<PhysicalFileFragment> fragments;
  for (const FileMetadata& meta : metadata) {
    const ScanDecision decision = evaluate_pruning(meta, scan.pushable_predicates());
    if (decision.selected_row_groups.empty() && !meta.row_groups.empty()) {
      continue;  // Every row group was proven unnecessary: skip the file entirely.
    }
    fragments.push_back(
        PhysicalFileFragment{meta.path, meta.row_count, static_cast<int>(meta.row_groups.size()),
                             decision.selected_row_groups, decision.skipped_row_groups, decision.reasons});
  }

  // Narrow the schema (and the matching column list) to required_columns(),
  // preserving the scan's original field order rather than
  // required_columns()'s alphabetical order (see LogicalScan/optimizer.cpp).
  const std::vector<std::string>& required = scan.required_columns();
  std::vector<Field> narrowed_fields;
  std::vector<std::string> narrowed_columns;
  for (const Field& field : scan.output_schema().fields()) {
    if (std::find(required.begin(), required.end(), field.name) != required.end()) {
      narrowed_fields.push_back(field);
      narrowed_columns.push_back(field.name);
    }
  }

  return std::make_shared<ParquetScanNode>(std::move(fragments), std::move(narrowed_columns),
                                           Schema(std::move(narrowed_fields)),
                                           static_cast<int>(metadata.size()));
}

PhysicalPlanPtr convert(const LogicalPlanPtr& node, ObjectStore& store) {
  if (const auto* scan = dynamic_cast<const LogicalScan*>(node.get())) {
    return convert_scan(*scan, store);
  }
  if (const auto* filter = dynamic_cast<const LogicalFilter*>(node.get())) {
    PhysicalPlanPtr child = convert(filter->child(), store);
    const Schema* scan_schema = find_scan_schema(*child);
    ExpressionPtr predicate =
        scan_schema ? remap_columns(filter->predicate(), *scan_schema) : filter->predicate();
    return std::make_shared<FilterNode>(std::move(child), std::move(predicate));
  }
  if (const auto* projection = dynamic_cast<const LogicalProjection*>(node.get())) {
    PhysicalPlanPtr child = convert(projection->child(), store);
    // A LogicalProjection is the one node with two structurally different
    // roles: the non-aggregate path's final projection, sitting directly
    // on LogicalFilter/LogicalScan with items that reference the
    // base-table schema (needs remapping, same as a Filter predicate); and
    // the aggregate path's reprojection, sitting directly on
    // LogicalAggregate with items that already reference *that* node's
    // own output schema one-for-one (e.g. an AggregateExpression alias
    // like "total" -- remapping that against the scan would fail, since
    // "total" was never a scanned column). Only the first case needs
    // remapping -- see the near-identical LogicalSort discriminator below
    // for why this checks positively for Filter/Scan rather than
    // negatively for "not Aggregate".
    const bool items_reference_scan_schema =
        dynamic_cast<const LogicalFilter*>(projection->child().get()) != nullptr ||
        dynamic_cast<const LogicalScan*>(projection->child().get()) != nullptr;
    std::vector<NamedExpression> items = projection->items();
    if (items_reference_scan_schema) {
      if (const Schema* scan_schema = find_scan_schema(*child)) items = remap_named(items, *scan_schema);
    }
    return std::make_shared<ProjectionNode>(std::move(child), std::move(items));
  }
  if (const auto* aggregate = dynamic_cast<const LogicalAggregate*>(node.get())) {
    PhysicalPlanPtr child = convert(aggregate->child(), store);
    const Schema* scan_schema = find_scan_schema(*child);
    std::vector<NamedExpression> aggregates =
        scan_schema ? remap_named(aggregate->aggregates(), *scan_schema) : aggregate->aggregates();
    if (aggregate->group_by().empty()) {
      return std::make_shared<ScalarAggregateNode>(std::move(child), std::move(aggregates));
    }
    std::vector<NamedExpression> group_by =
        scan_schema ? remap_named(aggregate->group_by(), *scan_schema) : aggregate->group_by();
    return std::make_shared<HashAggregateNode>(std::move(child), std::move(group_by), std::move(aggregates));
  }
  if (const auto* sort = dynamic_cast<const LogicalSort*>(node.get())) {
    PhysicalPlanPtr child = convert(sort->child(), store);
    std::vector<LogicalSort::Key> keys = sort->keys();
    // logical_planner.cpp places a non-aggregate ORDER BY's Sort directly
    // on the scan/filter chain -- its child is *always* exactly a
    // LogicalFilter or LogicalScan, by construction -- with keys that
    // reference the base-table schema, same as a Filter predicate. An
    // aggregate query's Sort sits above the aggregate's output instead,
    // with keys already referencing that output schema one-for-one (see
    // binder.cpp's ORDER BY handling), needing no remap.
    //
    // Checking for "child is LogicalFilter/LogicalScan" (a positive match
    // on the one case that needs remapping) rather than "child is not
    // LogicalProjection" (a negative match on the other case) matters: the
    // optimizer's redundant-projection-removal rule can delete the
    // aggregate-path's LogicalProjection when the SELECT list already
    // matches the aggregate's natural column order, leaving Sort sitting
    // directly on LogicalAggregate -- a LogicalProjection child is not a
    // reliable "aggregate path" signal, but LogicalFilter/LogicalScan *is*
    // a reliable "non-aggregate path" signal.
    const bool keys_reference_scan_schema =
        dynamic_cast<const LogicalFilter*>(sort->child().get()) != nullptr ||
        dynamic_cast<const LogicalScan*>(sort->child().get()) != nullptr;
    if (keys_reference_scan_schema) {
      if (const Schema* scan_schema = find_scan_schema(*child)) {
        for (LogicalSort::Key& key : keys) key.expr = remap_columns(key.expr, *scan_schema);
      }
    }
    return std::make_shared<SortNode>(std::move(child), std::move(keys));
  }
  if (const auto* limit = dynamic_cast<const LogicalLimit*>(node.get())) {
    return std::make_shared<LimitNode>(convert(limit->child(), store), limit->limit());
  }
  throw PlanningError("physical planner encountered an unrecognized logical plan node");
}

}  // namespace

PhysicalPlanPtr build_physical_plan(const LogicalPlanPtr& logical_plan, ObjectStore& store) {
  return std::make_shared<ArrowResultNode>(convert(logical_plan, store));
}

}  // namespace kernellake
