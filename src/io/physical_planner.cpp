#include "kernellake/io/physical_planner.hpp"

#include <algorithm>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/io/parquet_pruning.hpp"
#include "kernellake/storage/file_discovery.hpp"

namespace kernellake {

namespace {

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
    return std::make_shared<FilterNode>(convert(filter->child(), store), filter->predicate());
  }
  if (const auto* projection = dynamic_cast<const LogicalProjection*>(node.get())) {
    return std::make_shared<ProjectionNode>(convert(projection->child(), store),
                                             projection->items());
  }
  if (const auto* aggregate = dynamic_cast<const LogicalAggregate*>(node.get())) {
    PhysicalPlanPtr child = convert(aggregate->child(), store);
    if (aggregate->group_by().empty()) {
      return std::make_shared<ScalarAggregateNode>(std::move(child), aggregate->aggregates());
    }
    return std::make_shared<HashAggregateNode>(std::move(child), aggregate->group_by(),
                                                aggregate->aggregates());
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
