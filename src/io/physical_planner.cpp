#include "kernellake/io/physical_planner.hpp"

#include <fmt/format.h>

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
      throw PlanningError(fmt::format(
          "physical planner: column '{}' referenced above the scan but missing from its pruned column list "
          "(internal error)",
          column->name()));
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
  if (const auto* like = dynamic_cast<const LikeExpression*>(expr.get())) {
    return std::make_shared<LikeExpression>(remap_columns(like->value(), scan_schema), like->pattern(),
                                            like->negated());
  }
  if (const auto* case_expr = dynamic_cast<const CaseExpression*>(expr.get())) {
    std::vector<CaseExpression::WhenThen> branches;
    branches.reserve(case_expr->when_then().size());
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      branches.push_back(CaseExpression::WhenThen{remap_columns(branch.condition, scan_schema),
                                                  remap_columns(branch.result, scan_schema)});
    }
    ExpressionPtr else_branch =
        case_expr->else_branch() ? remap_columns(case_expr->else_branch(), scan_schema) : nullptr;
    return std::make_shared<CaseExpression>(std::move(branches), std::move(else_branch),
                                            case_expr->result_type());
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

// Finds the schema every expression sitting directly above a scan (or, for
// a JOIN query, directly above the HashJoinNode) must be remapped against.
// A HashJoinNode is a schema boundary exactly like a ParquetScanNode is --
// its own already-narrowed, already-concatenated output_schema() is what a
// Filter/Projection/Aggregate/Sort sitting on top of a join needs, so the
// search stops there rather than continuing into the join's two children
// (which have two separate, incompatible narrowed schemas).
//
// Known limitation: remapping above a join matches by column *name* against
// this combined schema (same as the single-table case always has), so if
// both JOIN sides happen to have a column with the same name, an unqualified
// reference to it above the join could resolve to the wrong side. Not a
// concern for the join *condition* itself (bound with each side's own
// index, see binder.cpp), only for additional WHERE/SELECT/GROUP BY
// references after the join -- avoid colliding column names across joined
// tables, or select/rename them distinctly, until this is tightened.
// Estimates a physical plan node's output row count, for choosing a hash
// join's build side (HashJoinOperator always materializes its *right*
// child into device memory -- see that class's own doc comment -- so the
// smaller side should end up there, not whichever side a query happened
// to write first). This is a rough, pre-filter/pre-aggregation size
// estimate, not real cardinality estimation: a ParquetScanNode reports the
// sum of its scanned files' whole-file row counts (ignoring row-group
// pruning and any filter selectivity), and every single-child node above
// it (Filter, Projection, aggregates, Sort, Limit, ...) just passes that
// estimate through unchanged via its generic children() accessor, rather
// than needing a case for every node type. Returns nullopt when no
// estimate is available (never guesses in that case).
std::optional<std::int64_t> estimate_row_count(const PhysicalPlanPtr& node) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(node.get())) {
    std::int64_t total = 0;
    for (const PhysicalFileFragment& fragment : scan->fragments()) {
      total += fragment.file_row_count;
    }
    return total;
  }
  if (const auto* join = dynamic_cast<const HashJoinNode*>(node.get())) {
    const std::optional<std::int64_t> left_estimate = estimate_row_count(join->left());
    const std::optional<std::int64_t> right_estimate = estimate_row_count(join->right());
    if (!left_estimate || !right_estimate) {
      return std::nullopt;
    }
    // A real inner join's output can exceed either input (duplicate
    // keys), but "smaller of the two inputs" is the usual fallback real
    // planners use absent join-selectivity statistics, and is enough to
    // let an N-way chain's outer join estimate a nested join child's
    // size too.
    return std::min(*left_estimate, *right_estimate);
  }
  const std::vector<PhysicalPlanPtr> children = node->children();
  if (children.size() == 1) {
    return estimate_row_count(children.front());
  }
  return std::nullopt;
}

const Schema* find_scan_schema(const PhysicalPlanNode& node) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(&node)) {
    return &scan->output_schema();
  }
  if (const auto* join = dynamic_cast<const HashJoinNode*>(&node)) {
    return &join->output_schema();
  }
  for (const PhysicalPlanPtr& child : node.children()) {
    if (const Schema* found = find_scan_schema(*child)) {
      return found;
    }
  }
  return nullptr;
}

PhysicalPlanPtr convert_scan(const LogicalScan& scan, ObjectStore& store) {
  const std::vector<ObjectInfo> files = discover_parquet_files(store, scan.source_paths());
  std::vector<FileMetadata> metadata;
  metadata.reserve(files.size());
  for (const ObjectInfo& file : files) {
    metadata.push_back(inspect_parquet_file(store, file.uri));
  }
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

  // A bare `COUNT(*)` (no other column referenced anywhere in the query --
  // no WHERE/GROUP BY/join) legitimately produces an empty required_columns()
  // here: COUNT(*) needs no column data, only a row count. But cudf::table
  // has no way to represent "N rows, 0 columns" (unlike arrow::RecordBatch,
  // which tracks row count independently of its columns) -- a cudf::table
  // built from zero selected columns reports num_rows() == 0 regardless of
  // how many rows the underlying row groups actually contain, so the GPU
  // scan operator would silently produce no batches at all. Keeping one
  // arbitrary real column (the schema's first field) selected in this case
  // preserves row-count fidelity through cudf::table; nothing above the
  // scan references it (that's exactly why required_columns() was empty),
  // so it's inert for every consumer except row counting. Real Parquet
  // files always have at least one column, so the guard below is only for
  // a pathological zero-field schema, not the common case.
  if (narrowed_columns.empty() && !scan.output_schema().fields().empty()) {
    const Field& fallback = scan.output_schema().fields().front();
    narrowed_fields.push_back(fallback);
    narrowed_columns.push_back(fallback.name);
  }

  return std::make_shared<ParquetScanNode>(std::move(fragments), std::move(narrowed_columns),
                                           Schema(std::move(narrowed_fields)),
                                           static_cast<int>(metadata.size()));
}

PhysicalPlanPtr convert(const LogicalPlanPtr& node, ObjectStore& store) {
  if (const auto* scan = dynamic_cast<const LogicalScan*>(node.get())) {
    return convert_scan(*scan, store);
  }
  if (const auto* join = dynamic_cast<const LogicalJoin*>(node.get())) {
    PhysicalPlanPtr left_child = convert(join->left(), store);
    PhysicalPlanPtr right_child = convert(join->right(), store);
    // The join key's index was assigned against each side's *original*
    // (pre-narrowing) schema; translate by name to that side's actual
    // narrowed physical schema, exactly like remap_columns does for any
    // other column reference above a scan.
    const std::string& left_key_name = join->left()->output_schema().field(join->left_key_index()).name;
    const std::string& right_key_name = join->right()->output_schema().field(join->right_key_index()).name;
    std::optional<std::size_t> left_key_narrowed = left_child->output_schema().find_field(left_key_name);
    std::optional<std::size_t> right_key_narrowed = right_child->output_schema().find_field(right_key_name);
    if (!left_key_narrowed || !right_key_narrowed) {
      throw PlanningError(
          "physical planner: JOIN key column missing from its pruned column list (internal "
          "error)");
    }
    // HashJoinOperator always builds its hash table on the *right* child
    // (see its own doc comment) -- swap here if the left side's
    // estimated row count is smaller, so the smaller table ends up
    // materialized into device memory instead of whichever side a query
    // happened to write first in its FROM/JOIN clause. Every expression
    // above this node already resolves columns by *name* against this
    // node's own output_schema() (see find_scan_schema()'s own comment),
    // never by fixed position, so reordering left/right here is
    // transparent to everything above it. Only swaps when both sides'
    // sizes are actually known -- never guesses.
    const std::optional<std::int64_t> left_estimate = estimate_row_count(left_child);
    const std::optional<std::int64_t> right_estimate = estimate_row_count(right_child);
    if (left_estimate && right_estimate && *left_estimate < *right_estimate) {
      std::swap(left_child, right_child);
      std::swap(left_key_narrowed, right_key_narrowed);
    }
    return std::make_shared<HashJoinNode>(std::move(left_child), std::move(right_child), *left_key_narrowed,
                                          *right_key_narrowed);
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
    // LogicalJoin joins this list alongside Filter/Scan: a Projection can
    // sit directly on a JOIN with no intervening WHERE clause beyond the ON
    // condition (e.g. `SELECT a.x, b.y FROM ... JOIN ... ON ...`), and its
    // items then reference the join's combined pre-narrowing schema just
    // like they'd reference a bare scan's.
    const bool items_reference_scan_schema =
        dynamic_cast<const LogicalFilter*>(projection->child().get()) != nullptr ||
        dynamic_cast<const LogicalScan*>(projection->child().get()) != nullptr ||
        dynamic_cast<const LogicalJoin*>(projection->child().get()) != nullptr;
    std::vector<NamedExpression> items = projection->items();
    if (items_reference_scan_schema) {
      if (const Schema* scan_schema = find_scan_schema(*child)) {
        items = remap_named(items, *scan_schema);
      }
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
        dynamic_cast<const LogicalScan*>(sort->child().get()) != nullptr ||
        dynamic_cast<const LogicalJoin*>(sort->child().get()) != nullptr;
    if (keys_reference_scan_schema) {
      if (const Schema* scan_schema = find_scan_schema(*child)) {
        for (LogicalSort::Key& key : keys) {
          key.expr = remap_columns(key.expr, *scan_schema);
        }
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
