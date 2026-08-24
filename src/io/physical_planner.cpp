#include "kernellake/io/physical_planner.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/io/parquet_pruning.hpp"
#include "kernellake/io/table_resolution.hpp"

namespace kernellake {

namespace {

// The schema/column-identity a Filter/Projection/Aggregate/Sort sitting
// directly above a scan or JOIN needs to remap its ColumnExpressions
// against -- see find_scan_boundary() below for how this is found, and
// ParquetScanNode::original_column_map()'s own comment for why
// `original_to_narrowed` exists instead of relying on Schema::find_field().
struct ScanBoundary {
  const Schema* schema;
  const std::vector<std::optional<std::size_t>>* original_to_narrowed;
};

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
//
// Rewritten by index (`column->column_index()` looked up in
// `boundary.original_to_narrowed`), not by name against `boundary.schema` --
// a name-based lookup is ambiguous once a JOIN puts two same-named columns
// from different sides into one combined schema, silently resolving a
// reference to the wrong side's column (see ParquetScanNode's and
// HashJoinNode's own original_column_map() comments in physical_plan.hpp).
ExpressionPtr remap_columns(const ExpressionPtr& expr, const ScanBoundary& boundary) {
  if (const auto* column = dynamic_cast<const ColumnExpression*>(expr.get())) {
    const std::size_t original_index = column->column_index();
    const std::optional<std::size_t> narrowed_index = original_index < boundary.original_to_narrowed->size()
                                                          ? (*boundary.original_to_narrowed)[original_index]
                                                          : std::nullopt;
    if (!narrowed_index) {
      throw PlanningError(fmt::format(
          "physical planner: column '{}' referenced above the scan but missing from its pruned column list "
          "(internal error)",
          column->name()));
    }
    return std::make_shared<ColumnExpression>(column->name(), *narrowed_index, column->result_type());
  }
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(expr.get())) {
    return std::make_shared<BinaryExpression>(binary->op(), remap_columns(binary->left(), boundary),
                                              remap_columns(binary->right(), boundary),
                                              binary->result_type());
  }
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(expr.get())) {
    return std::make_shared<UnaryExpression>(unary->op(), remap_columns(unary->operand(), boundary),
                                             unary->result_type());
  }
  if (const auto* cast = dynamic_cast<const CastExpression*>(expr.get())) {
    return std::make_shared<CastExpression>(remap_columns(cast->operand(), boundary), cast->result_type());
  }
  if (const auto* between = dynamic_cast<const BetweenExpression*>(expr.get())) {
    return std::make_shared<BetweenExpression>(remap_columns(between->value(), boundary),
                                               remap_columns(between->lower(), boundary),
                                               remap_columns(between->upper(), boundary));
  }
  if (const auto* aggregate = dynamic_cast<const AggregateExpression*>(expr.get())) {
    ExpressionPtr argument = aggregate->argument() ? remap_columns(aggregate->argument(), boundary) : nullptr;
    return std::make_shared<AggregateExpression>(aggregate->function(), std::move(argument),
                                                 aggregate->result_type());
  }
  if (const auto* like = dynamic_cast<const LikeExpression*>(expr.get())) {
    return std::make_shared<LikeExpression>(remap_columns(like->value(), boundary), like->pattern(),
                                            like->negated());
  }
  if (const auto* case_expr = dynamic_cast<const CaseExpression*>(expr.get())) {
    std::vector<CaseExpression::WhenThen> branches;
    branches.reserve(case_expr->when_then().size());
    for (const CaseExpression::WhenThen& branch : case_expr->when_then()) {
      branches.push_back(CaseExpression::WhenThen{remap_columns(branch.condition, boundary),
                                                  remap_columns(branch.result, boundary)});
    }
    ExpressionPtr else_branch =
        case_expr->else_branch() ? remap_columns(case_expr->else_branch(), boundary) : nullptr;
    return std::make_shared<CaseExpression>(std::move(branches), std::move(else_branch),
                                            case_expr->result_type());
  }
  if (const auto* extract = dynamic_cast<const ExtractExpression*>(expr.get())) {
    return std::make_shared<ExtractExpression>(extract->part(), remap_columns(extract->operand(), boundary),
                                               extract->result_type());
  }
  return expr;  // LiteralExpression: no column reference to remap.
}

std::vector<NamedExpression> remap_named(const std::vector<NamedExpression>& items,
                                         const ScanBoundary& boundary) {
  std::vector<NamedExpression> result;
  result.reserve(items.size());
  for (const NamedExpression& item : items) {
    result.push_back(NamedExpression{remap_columns(item.expr, boundary), item.name});
  }
  return result;
}

// True if `items` is a plain, order-preserving pass-through of `child_schema`
// (every item is a bare column reference at its own position, under its own
// name) -- i.e. materializing a ProjectionNode for it would do nothing but
// copy the child's output unchanged. Checked here, against `items` and
// `child`'s schema *after* remap_named() above has already run (so this
// necessarily runs after column pruning has already used the projection's
// original items in optimizer.cpp's annotate_scan()) -- eliding the
// equivalent identity LogicalProjection any earlier than this, before pruning
// sees it, is what caused a real bug: `SELECT id, amount FROM t WHERE id < 3`
// silently returned only `id`, since nothing was left in the logical tree to
// tell annotate_scan the query's actual output needed `amount` too. See
// optimizer.cpp's rewrite_plan() LogicalProjection case for the full story.
bool is_identity_projection(const std::vector<NamedExpression>& items, const Schema& child_schema) {
  if (items.size() != child_schema.field_count()) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto* column = dynamic_cast<const ColumnExpression*>(items[i].expr.get());
    if (column == nullptr) {
      return false;
    }
    if (column->column_index() != i) {
      return false;
    }
    if (items[i].name != child_schema.field(i).name) {
      return false;
    }
  }
  return true;
}

// Rough default-selectivity heuristic for estimate_row_count()'s FilterNode
// case below, in the same spirit as classic optimizers that fall back to
// fixed per-predicate-shape defaults absent real column statistics/
// histograms (this project has none -- Parquet row-group min/max stats
// drive evaluate_pruning()'s exact, provable row-group elimination
// elsewhere, but say nothing about selectivity *within* a kept row group).
// Never a correctness concern either way estimate_row_count() only feeds
// the hash join build-side choice (see that function's own comment), a
// pure performance heuristic -- an inner join's result is identical
// regardless of which side builds, so a wrong selectivity guess costs
// performance, never correctness. Walks top-level AND/OR structure
// recursively (mirroring optimizer.cpp's own flatten_and_conjuncts(), same
// idea applied to estimation instead of pushdown) so a WHERE clause's
// several independent predicates each contribute their own discount rather
// than the whole clause being treated as one opaque unit.
//
// Real motivating case: TPC-H Q12 (orders JOIN lineitem, WHERE clause
// entirely on lineitem columns) -- orders has no predicate of its own (the
// query's own semantics need it in full), while lineitem's WHERE clause
// (l_shipmode IN (...), a shipdate/commitdate/receiptdate ordering
// constraint, a one-year receiptdate range) is genuinely selective. Without
// this, both sides were compared by whole-file row count alone, orders'
// raw count came out smaller than lineitem's raw count at SF1000, and
// HashJoinOperator built its hash table on the *larger* (post-filter)
// side -- confirmed as the real, reproducible cause of Q12 running slower
// on KernelLake than on PySpark in two separate real AWS SF1000 runs
// (2026-08-16).
double estimate_selectivity(const Expression& predicate) {
  if (const auto* binary = dynamic_cast<const BinaryExpression*>(&predicate)) {
    switch (binary->op()) {
      case BinaryOperator::And:
        return estimate_selectivity(*binary->left()) * estimate_selectivity(*binary->right());
      case BinaryOperator::Or: {
        // Independence-assumption union, same as a classic optimizer's
        // default absent real correlation statistics: P(A or B) = P(A) +
        // P(B) - P(A)*P(B), clamped to 1.0 so a long OR chain (e.g. an
        // IN-list desugared into ORed equalities) can't estimate above
        // "every row passes."
        const double left = estimate_selectivity(*binary->left());
        const double right = estimate_selectivity(*binary->right());
        return std::min(1.0, left + right - left * right);
      }
      case BinaryOperator::Equal:
        return 0.1;
      case BinaryOperator::NotEqual:
        return 0.9;
      case BinaryOperator::Less:
      case BinaryOperator::LessEqual:
      case BinaryOperator::Greater:
      case BinaryOperator::GreaterEqual:
        return 0.33;
      default:
        // Arithmetic operators (Add/Subtract/Multiply/Divide) never appear
        // as a filter's own top-level boolean shape; no discount if one
        // somehow does.
        return 1.0;
    }
  }
  if (dynamic_cast<const BetweenExpression*>(&predicate)) {
    return 0.25;
  }
  if (const auto* unary = dynamic_cast<const UnaryExpression*>(&predicate)) {
    switch (unary->op()) {
      case UnaryOperator::Not:
        return 1.0 - estimate_selectivity(*unary->operand());
      case UnaryOperator::IsNull:
        return 0.05;
      case UnaryOperator::IsNotNull:
        return 0.95;
      default:
        return 1.0;
    }
  }
  if (dynamic_cast<const LikeExpression*>(&predicate)) {
    return 0.25;
  }
  // Anything else (a bare boolean column, CASE, a function call, ...): no
  // discount rather than a guess -- estimate_row_count() would rather
  // slightly overestimate a filtered scan's size (worst case: a missed
  // build-side swap) than underestimate it (worst case: swapping onto a
  // side that turns out not to be smaller after all).
  return 1.0;
}

// Estimates a physical plan node's output row count, for choosing a hash
// join's build side (HashJoinOperator always materializes its *right*
// child into device memory -- see that class's own doc comment -- so the
// smaller side should end up there, not whichever side a query happened
// to write first). Not real cardinality estimation -- no histograms, no
// join-selectivity modeling -- but no longer purely pre-filter either: a
// ParquetScanNode reports the sum of its scanned files' whole-file row
// counts (still ignoring row-group pruning, which evaluate_pruning() only
// applies when a predicate provably eliminates whole row groups -- most
// TPC-H-derived predicates here don't, since generate_tpch.py assigns
// values uniformly at random per row, uncorrelated with row-group layout);
// a FilterNode discounts its child's estimate via estimate_selectivity()
// above; every other single-child node (Projection, aggregates, Sort,
// Limit, ...) still just passes its child's estimate through unchanged via
// the generic children() accessor. Returns nullopt when no estimate is
// available (never guesses in that case).
std::optional<std::int64_t> estimate_row_count(const PhysicalPlanPtr& node) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(node.get())) {
    std::int64_t total = 0;
    for (const PhysicalFileFragment& fragment : scan->fragments()) {
      total += fragment.file_row_count;
    }
    return total;
  }
  if (const auto* filter = dynamic_cast<const FilterNode*>(node.get())) {
    const std::optional<std::int64_t> child_estimate = estimate_row_count(filter->child());
    if (!child_estimate) {
      return std::nullopt;
    }
    const double selectivity = estimate_selectivity(*filter->predicate());
    return static_cast<std::int64_t>(std::llround(static_cast<double>(*child_estimate) * selectivity));
  }
  if (const auto* join = dynamic_cast<const HashJoinNode*>(node.get())) {
    const std::optional<std::int64_t> left_estimate = estimate_row_count(join->left());
    const std::optional<std::int64_t> right_estimate = estimate_row_count(join->right());
    if (!left_estimate || !right_estimate) {
      return std::nullopt;
    }
    // max(), not min() (fixed 2026-08-16 -- see below): this estimate only
    // matters when this HashJoinNode is itself the left or right child of
    // an *outer* join in an N-way chain (the only caller, this function's
    // own recursion from the outer join's own left_estimate/right_estimate
    // above -- see convert()'s JOIN case, where the identical
    // left_estimate/right_estimate pattern also becomes that outer join's
    // own estimated_build_rows). For every real TPC-H-derived N-way chain
    // this engine runs, that inner join is a foreign-key join (e.g. Q3's
    // customer JOIN orders feeding the outer join against lineitem): most
    // rows on the *larger* side survive (each has a matching key on the
    // smaller side, by construction of real referential integrity), so the
    // join's true output tracks the larger input's size, not the smaller
    // one -- min() was a real, confirmed bug, not just a theoretical
    // under-estimate. Real SF1000 TPC-H Q3 GPU OOM root-caused to exactly
    // this (2026-08-16): min(customer_filtered, orders_filtered) badly
    // undersized the customer-JOIN-orders inner join's real output (an FK
    // join, output size tracks orders_filtered, the larger side, not
    // customer_filtered), so the outer join's estimated_build_rows stayed
    // under build_side_budget_bytes (query_engine_execute_gpu.cpp) and
    // choose_partition_count() (hash_join_operator.cpp) decided
    // partitioning wasn't needed -- when the join's *actual* build side, at
    // real SF1000 scale, was large enough that HashJoinOperator::open()'s
    // unbounded, unpartitioned cudf::concatenate() OOM'd. max() is the
    // conservative direction to be wrong in: it can only trigger
    // partitioning/build-side swaps *more* readily than strictly necessary
    // (cheap -- see choose_partition_count()'s own comment on why erring
    // toward more partitions is safe), never less, unlike min() silently
    // under-provisioning toward the exact failure this was meant to guard
    // against.
    return std::max(*left_estimate, *right_estimate);
  }
  const std::vector<PhysicalPlanPtr> children = node->children();
  if (children.size() == 1) {
    return estimate_row_count(children.front());
  }
  return std::nullopt;
}

// Finds the schema/column-map every expression sitting directly above a
// scan (or, for a JOIN query, directly above the HashJoinNode) must be
// remapped against. A HashJoinNode is a schema boundary exactly like a
// ParquetScanNode is -- its own already-narrowed, already-concatenated
// output_schema() (and its own combined original_column_map(), built while
// converting the LogicalJoin -- see convert()'s JOIN case) is what a
// Filter/Projection/Aggregate/Sort sitting on top of a join needs, so the
// search stops there rather than continuing into the join's two children
// (which have two separate, incompatible narrowed schemas).
std::optional<ScanBoundary> find_scan_boundary(const PhysicalPlanNode& node) {
  if (const auto* scan = dynamic_cast<const ParquetScanNode*>(&node)) {
    return ScanBoundary{&scan->output_schema(), &scan->original_column_map()};
  }
  if (const auto* join = dynamic_cast<const HashJoinNode*>(&node)) {
    return ScanBoundary{&join->output_schema(), &join->original_column_map()};
  }
  for (const PhysicalPlanPtr& child : node.children()) {
    if (const std::optional<ScanBoundary> found = find_scan_boundary(*child)) {
      return found;
    }
  }
  return std::nullopt;
}

// Whether a node's ColumnExpressions reference the base-table (pre-pruning)
// scan schema, looking through any LogicalSort/LogicalLimit sitting in
// between -- both are pass-through nodes (same schema as their child) that
// can end up directly under a LogicalProjection: LogicalSort, whenever a
// non-aggregate query has an ORDER BY (binder.cpp places it directly on the
// Filter/Scan chain); LogicalLimit, whenever the optimizer's insert_limit()
// pushes a LIMIT down through a LogicalProjection to sit just above the
// scan/filter/join it can actually benefit from (optimizer.cpp). Missing
// either case here previously left a Projection's items/Sort's keys
// unremapped whenever one of these sat in between -- a real bug: the scan
// beneath had already been narrowed to required_columns(), but the
// expressions above it still carried original (pre-narrowing) column
// indices, causing an out-of-bounds vector access at execution time for any
// narrowed (non-*, non-aggregate) SELECT combined with ORDER BY and/or
// LIMIT. See PhysicalPlannerTest.SurvivingPlainProjectionRemapsThroughAn
// InterposedSort/...InterposedLimit.
//
// A `LogicalFilter` is *not* always a positive match: a HAVING filter (see
// logical_planner.cpp's finish_logical_plan()) is also a LogicalFilter, but
// sits directly on a LogicalAggregate and its ColumnExpressions reference
// *that* node's output schema, not the scan's -- the same distinction this
// function already draws for LogicalScan vs. everything else. Real bug
// this exact gap caused: `GROUP BY ... HAVING ... ORDER BY ...`, once the
// optimizer's redundant-projection-removal rule elides the aggregate
// path's re-projection (leaving LogicalSort sitting directly on the HAVING
// LogicalFilter), wrongly remapped the Sort's already-correctly-indexed
// keys as if they were scan indices -- an out-of-range FieldRef at
// execution time on any query combining a JOIN with GROUP BY/HAVING.
bool references_scan_schema(const LogicalPlanNode* node) {
  while (true) {
    if (const auto* sort = dynamic_cast<const LogicalSort*>(node)) {
      node = sort->child().get();
      continue;
    }
    if (const auto* limit = dynamic_cast<const LogicalLimit*>(node)) {
      node = limit->child().get();
      continue;
    }
    if (const auto* filter = dynamic_cast<const LogicalFilter*>(node)) {
      return dynamic_cast<const LogicalAggregate*>(filter->child().get()) == nullptr;
    }
    return dynamic_cast<const LogicalScan*>(node) != nullptr ||
           dynamic_cast<const LogicalJoin*>(node) != nullptr;
  }
}

PhysicalPlanPtr convert_scan(const LogicalScan& scan, ObjectStore& store,
                             TableSourceResolver* extra_resolver) {
  const ResolvedTable resolved =
      resolve_table_or_delegate(store, scan.source_paths(), extra_resolver, scan.pushable_predicates());

  // Narrow the schema (and the matching column list) to required_columns(),
  // preserving the scan's original field order rather than
  // required_columns()'s alphabetical order (see LogicalScan/optimizer.cpp).
  // A required *partition* column (never physically present in the file --
  // see LogicalScan::partition_columns()) goes into narrowed_partitions
  // instead of narrowed_columns, so it's never handed to cudf's/Arrow's
  // Parquet reader as a column to read; the scan operator materializes it
  // as a per-fragment constant from PhysicalFileFragment::partition_values
  // instead. Computed *before* building fragments below (not after, as an
  // earlier version of this function did) specifically so
  // kept_partition_original_indices is available to narrow each fragment's
  // own partition_values in lockstep with narrowed_partitions -- otherwise
  // ParquetScanNode::partition_columns() (narrowed) and
  // PhysicalFileFragment::partition_values (previously left at the full,
  // unnarrowed per-file list from resolve_table()) silently fall out of
  // the positional-parallel relationship PhysicalFileFragment's own doc
  // comment documents, and the scan operator pairs each narrowed partition
  // column's type with the wrong original column's value for every row.
  const std::vector<std::string>& required = scan.required_columns();
  std::vector<Field> narrowed_fields;
  std::vector<std::string> narrowed_columns;
  std::vector<PartitionColumn> narrowed_partitions;
  std::vector<std::size_t> kept_partition_original_indices;
  // original_column_map[i] records where this LogicalScan's original field
  // i ended up in narrowed_fields (nullopt if pruned) -- see
  // ParquetScanNode::original_column_map()'s own comment for why this
  // exists (index-based remapping above the scan, not by-name).
  std::vector<std::optional<std::size_t>> original_column_map(scan.output_schema().field_count(),
                                                              std::nullopt);
  for (std::size_t original_index = 0; original_index < scan.output_schema().field_count();
       ++original_index) {
    const Field& field = scan.output_schema().field(original_index);
    if (std::find(required.begin(), required.end(), field.name) == required.end()) {
      continue;
    }
    original_column_map[original_index] = narrowed_fields.size();
    narrowed_fields.push_back(field);
    bool matched_partition = false;
    for (std::size_t partition_index = 0; partition_index < scan.partition_columns().size();
         ++partition_index) {
      const PartitionColumn& column = scan.partition_columns()[partition_index];
      if (column.name == field.name) {
        narrowed_partitions.push_back(column);
        kept_partition_original_indices.push_back(partition_index);
        matched_partition = true;
        break;
      }
    }
    if (!matched_partition) {
      narrowed_columns.push_back(field.name);
    }
  }

  std::vector<PhysicalFileFragment> fragments;
  fragments.reserve(resolved.files.size());
  for (const ResolvedFile& file : resolved.files) {
    const FileMetadata& meta = file.metadata;
    const ScanDecision decision = evaluate_pruning(meta, scan.pushable_predicates());
    if (decision.selected_row_groups.empty() && !meta.row_groups.empty()) {
      continue;  // Every row group was proven unnecessary: skip the file entirely.
    }
    // Narrowed to the same kept partition columns, in the same order, as
    // narrowed_partitions above -- see this function's own comment on why.
    std::vector<LiteralStorage> narrowed_partition_values;
    narrowed_partition_values.reserve(kept_partition_original_indices.size());
    for (const std::size_t partition_index : kept_partition_original_indices) {
      narrowed_partition_values.push_back(file.partition_values[partition_index]);
    }
    fragments.push_back(PhysicalFileFragment{
        meta.path, meta.row_count, static_cast<int>(meta.row_groups.size()), decision.selected_row_groups,
        decision.skipped_row_groups, decision.reasons, std::move(narrowed_partition_values)});
  }

  // A bare `COUNT(*)` (no other column referenced anywhere in the query --
  // no WHERE/GROUP BY/join) legitimately produces an empty required_columns()
  // here: COUNT(*) needs no column data, only a row count. But cudf::table
  // has no way to represent "N rows, 0 columns" (unlike arrow::RecordBatch,
  // which tracks row count independently of its columns) -- a cudf::table
  // built from zero selected columns reports num_rows() == 0 regardless of
  // how many rows the underlying row groups actually contain, so the GPU
  // scan operator would silently produce no batches at all. Keeping one
  // arbitrary real *physical* column (the schema's first field -- always
  // physical, never a partition column, since resolve_table() always
  // appends partition columns after every physical one) selected in this
  // case preserves row-count fidelity through cudf::table; nothing above
  // the scan references it (that's exactly why required_columns() was
  // empty), so it's inert for every consumer except row counting. Real
  // Parquet files always have at least one column, so the guard below is
  // only for a pathological zero-field schema, not the common case.
  if (narrowed_columns.empty() && narrowed_partitions.empty() && !scan.output_schema().fields().empty()) {
    const Field& fallback = scan.output_schema().fields().front();
    original_column_map[0] = narrowed_fields.size();
    narrowed_fields.push_back(fallback);
    narrowed_columns.push_back(fallback.name);
  }

  return std::make_shared<ParquetScanNode>(
      std::move(fragments), std::move(narrowed_columns), Schema(std::move(narrowed_fields)),
      static_cast<int>(resolved.files.size()), std::move(narrowed_partitions), std::move(original_column_map),
      resolved.owned_store);
}

PhysicalPlanPtr convert(const LogicalPlanPtr& node, ObjectStore& store, TableSourceResolver* extra_resolver) {
  if (const auto* scan = dynamic_cast<const LogicalScan*>(node.get())) {
    return convert_scan(*scan, store, extra_resolver);
  }
  if (const auto* join = dynamic_cast<const LogicalJoin*>(node.get())) {
    PhysicalPlanPtr left_child = convert(join->left(), store, extra_resolver);
    PhysicalPlanPtr right_child = convert(join->right(), store, extra_resolver);
    const std::optional<ScanBoundary> left_boundary = find_scan_boundary(*left_child);
    const std::optional<ScanBoundary> right_boundary = find_scan_boundary(*right_child);
    if (!left_boundary || !right_boundary) {
      throw PlanningError(
          "physical planner: JOIN child has no identifiable scan/join schema (internal error)");
    }
    // The join key's index was assigned against each side's *original*
    // (pre-narrowing) schema (see LogicalJoin's own doc comment) --
    // translate via that side's own original_column_map(), exactly like
    // remap_columns does for any other column reference above a scan.
    const std::vector<std::optional<std::size_t>>& left_original_map = *left_boundary->original_to_narrowed;
    const std::vector<std::optional<std::size_t>>& right_original_map = *right_boundary->original_to_narrowed;
    std::optional<std::size_t> left_key_narrowed = join->left_key_index() < left_original_map.size()
                                                       ? left_original_map[join->left_key_index()]
                                                       : std::nullopt;
    std::optional<std::size_t> right_key_narrowed = join->right_key_index() < right_original_map.size()
                                                        ? right_original_map[join->right_key_index()]
                                                        : std::nullopt;
    if (!left_key_narrowed || !right_key_narrowed) {
      throw PlanningError(
          "physical planner: JOIN key column missing from its pruned column list (internal "
          "error)");
    }

    // The *original* combined-schema column-index domain used by anything
    // sitting above this join (WHERE/SELECT/GROUP BY/ORDER BY, and a
    // nested outer LogicalJoin's own left_key_index/right_key_index) is
    // fixed at [0, left_original_count + right_original_count) in original
    // left-then-right order, per LogicalJoin's own doc comment --
    // regardless of which side HashJoinOperator ends up materializing
    // below. Compute where each original side's narrowed columns will
    // physically land *before* the build-side swap changes which
    // PhysicalPlanPtr variable holds which side, so the offsets below stay
    // correct either way.
    const std::optional<std::int64_t> left_estimate = estimate_row_count(left_child);
    const std::optional<std::int64_t> right_estimate = estimate_row_count(right_child);
    // Only safe for INNER JOIN, which is symmetric (swapping which side is
    // "build" vs "probe" never changes the result set, just which table
    // gets materialized into device memory). A LEFT OUTER JOIN is not
    // symmetric: HashJoinOperator's own LEFT JOIN implementation always
    // treats its "left" (probe) side as the preserved side and "right"
    // (build) side as the one that gets NULL-extended on no match --
    // swapping which SQL-level side lands in each physical slot here would
    // silently invert that, preserving the wrong side. See
    // docs/ARCHITECTURE.md's "Hash joins" section.
    const bool swap_for_build_side = join->join_type() == JoinType::Inner && left_estimate && right_estimate &&
                                     *left_estimate < *right_estimate;
    // Whichever side ends up in the *build* (right) slot after the swap
    // decision below -- persisted on HashJoinNode so HashJoinOperator can
    // size its partition count against it (see choose_partition_count() in
    // hash_join_operator.cpp). Requires both estimates so it's exactly the
    // one actually used for the swap decision above, not a fallback guess.
    const std::optional<std::int64_t> estimated_build_rows =
        left_estimate && right_estimate ? (swap_for_build_side ? left_estimate : right_estimate)
                                        : std::nullopt;
    const std::size_t left_narrowed_count = left_child->output_schema().field_count();
    const std::size_t right_narrowed_count = right_child->output_schema().field_count();
    const std::size_t original_left_physical_offset = swap_for_build_side ? right_narrowed_count : 0;
    const std::size_t original_right_physical_offset = swap_for_build_side ? 0 : left_narrowed_count;
    std::vector<std::optional<std::size_t>> combined_column_map;
    combined_column_map.reserve(left_original_map.size() + right_original_map.size());
    for (const std::optional<std::size_t>& index : left_original_map) {
      combined_column_map.push_back(index ? std::optional<std::size_t>(original_left_physical_offset + *index)
                                          : std::nullopt);
    }
    for (const std::optional<std::size_t>& index : right_original_map) {
      combined_column_map.push_back(
          index ? std::optional<std::size_t>(original_right_physical_offset + *index) : std::nullopt);
    }

    // HashJoinOperator always builds its hash table on the *right* child
    // (see its own doc comment) -- swap here if the left side's
    // estimated row count is smaller, so the smaller table ends up
    // materialized into device memory instead of whichever side a query
    // happened to write first in its FROM/JOIN clause. combined_column_map
    // above was already computed to match whichever physical order results
    // from this swap decision, so nothing else needs adjusting here.
    if (swap_for_build_side) {
      std::swap(left_child, right_child);
      std::swap(left_key_narrowed, right_key_narrowed);
    }
    return std::make_shared<HashJoinNode>(std::move(left_child), std::move(right_child), *left_key_narrowed,
                                          *right_key_narrowed, std::move(combined_column_map),
                                          estimated_build_rows, join->join_type());
  }
  if (const auto* filter = dynamic_cast<const LogicalFilter*>(node.get())) {
    PhysicalPlanPtr child = convert(filter->child(), store, extra_resolver);
    // A HAVING filter sits directly on LogicalAggregate (see
    // logical_planner.cpp's finish_logical_plan()) -- its predicate
    // already references *that* node's own output schema one-for-one, the
    // same "already correctly indexed, no remap needed" situation the
    // LogicalProjection case below documents for its own aggregate-
    // reprojection shape. Only remap when this filter's own predicate
    // actually references scan schema (a WHERE filter, sitting on
    // Scan/Filter/Join, possibly through an interposed Sort/Limit) --
    // reusing the exact same references_scan_schema() discriminator the
    // Projection/Sort cases below already rely on for the identical
    // problem.
    const bool predicate_references_scan_schema = references_scan_schema(filter->child().get());
    const std::optional<ScanBoundary> boundary =
        predicate_references_scan_schema ? find_scan_boundary(*child) : std::nullopt;
    ExpressionPtr predicate = boundary ? remap_columns(filter->predicate(), *boundary) : filter->predicate();
    return std::make_shared<FilterNode>(std::move(child), std::move(predicate));
  }
  if (const auto* projection = dynamic_cast<const LogicalProjection*>(node.get())) {
    PhysicalPlanPtr child = convert(projection->child(), store, extra_resolver);
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
    // like they'd reference a bare scan's. references_scan_schema() also
    // looks through any LogicalSort/LogicalLimit interposed here -- see its
    // own comment.
    const bool items_reference_scan_schema = references_scan_schema(projection->child().get());
    std::vector<NamedExpression> items = projection->items();
    if (items_reference_scan_schema) {
      if (const std::optional<ScanBoundary> boundary = find_scan_boundary(*child)) {
        items = remap_named(items, *boundary);
      }
    }
    // See is_identity_projection()'s own comment: safe to elide here (unlike
    // in optimizer.cpp) since column pruning has already run and items has
    // already been remapped against the real (possibly narrowed) child
    // schema above.
    if (is_identity_projection(items, child->output_schema())) {
      return child;
    }
    return std::make_shared<ProjectionNode>(std::move(child), std::move(items));
  }
  if (const auto* aggregate = dynamic_cast<const LogicalAggregate*>(node.get())) {
    PhysicalPlanPtr child = convert(aggregate->child(), store, extra_resolver);
    const std::optional<ScanBoundary> boundary = find_scan_boundary(*child);
    std::vector<NamedExpression> aggregates =
        boundary ? remap_named(aggregate->aggregates(), *boundary) : aggregate->aggregates();
    if (aggregate->group_by().empty()) {
      return std::make_shared<ScalarAggregateNode>(std::move(child), std::move(aggregates));
    }
    std::vector<NamedExpression> group_by =
        boundary ? remap_named(aggregate->group_by(), *boundary) : aggregate->group_by();
    return std::make_shared<HashAggregateNode>(std::move(child), std::move(group_by), std::move(aggregates));
  }
  if (const auto* sort = dynamic_cast<const LogicalSort*>(node.get())) {
    PhysicalPlanPtr child = convert(sort->child(), store, extra_resolver);
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
    // a reliable "non-aggregate path" signal. references_scan_schema() also
    // looks through any LogicalLimit interposed here -- see its own comment
    // (not currently reachable for Sort's own child via insert_limit(), which
    // only recurses through LogicalProjection, but kept consistent with the
    // identical check above rather than assuming that never changes).
    const bool keys_reference_scan_schema = references_scan_schema(sort->child().get());
    if (keys_reference_scan_schema) {
      if (const std::optional<ScanBoundary> boundary = find_scan_boundary(*child)) {
        for (LogicalSort::Key& key : keys) {
          key.expr = remap_columns(key.expr, *boundary);
        }
      }
    }
    return std::make_shared<SortNode>(std::move(child), std::move(keys));
  }
  if (const auto* limit = dynamic_cast<const LogicalLimit*>(node.get())) {
    return std::make_shared<LimitNode>(convert(limit->child(), store, extra_resolver), limit->limit());
  }
  throw PlanningError("physical planner encountered an unrecognized logical plan node");
}

}  // namespace

PhysicalPlanPtr build_physical_plan(const LogicalPlanPtr& logical_plan, ObjectStore& store,
                                    TableSourceResolver* extra_resolver) {
  return std::make_shared<ArrowResultNode>(convert(logical_plan, store, extra_resolver));
}

}  // namespace kernellake
