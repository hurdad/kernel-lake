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
// full schema -- see docs/architecture.md and the comment on
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

std::vector<NamedExpression> remap_named(const std::vector<NamedExpression>& items, const Schema& scan_schema) {
  std::vector<NamedExpression> result;
  result.reserve(items.size());
  for (const NamedExpression& item : items) {
    result.push_back(NamedExpression{remap_columns(item.expr, scan_schema), item.name});
  }
  return result;
}

// Single-source queries only (no joins yet -- see docs/roadmap.md), so
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
    fragments.push_back(PhysicalFileFragment{
        meta.path, meta.row_count, static_cast<int>(meta.row_groups.size()),
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
    const Schema* scan_schema = find_scan_schema(*child);
    std::vector<NamedExpression> items =
        scan_schema ? remap_named(projection->items(), *scan_schema) : projection->items();
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
  if (dynamic_cast<const LogicalSort*>(node.get()) != nullptr) {
    throw PlanningError(
        "ORDER BY has no physical execution path yet (Sort is a prepared interface only "
        "pending a physical sort operator)");
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
