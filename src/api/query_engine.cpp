#include "kernellake/api/query_engine.hpp"

#include <arrow/scalar.h>
#include <arrow/table.h>
#include <fmt/format.h>

#include <chrono>

#include "kernellake/common/errors.hpp"
#include "kernellake/delta/delta_source_resolver.hpp"
#include "kernellake/iceberg/iceberg_source_resolver.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/io/table_resolution.hpp"
#include "kernellake/optimizer/optimizer.hpp"
#include "kernellake/planner/logical_planner.hpp"
#include "kernellake/sql/parser.hpp"
#include "kernellake/sql/subquery_resolver.hpp"
#include "kernellake/storage/nvme_cache_otel.hpp"
#include "kernellake/unitycatalog/unity_catalog_source_resolver.hpp"

#include "composite_source_resolver.hpp"

namespace kernellake {

QueryEngine::QueryEngine(EngineConfig config, int device_id)
    : config_(std::move(config)), device_id_(device_id), store_(config_.storage) {}

namespace {

// DOUBLE/INT64/STRING cover both use cases this function serves: a HAVING
// scalar subquery's result is always DOUBLE/INT64 (SUM/AVG -> DOUBLE,
// COUNT/CountStar -> INT64; MIN/MAX pass their argument's own type
// through, which for every currently-generated TPC-H column is itself
// DOUBLE or INT64) -- see docs/ARCHITECTURE.md's HAVING section. An
// IN-subquery's result, though, is typically a plain (non-aggregated)
// grouped/selected column, which is just as often STRING (e.g.
// `region IN (SELECT region FROM ...)`) as numeric. Shared by
// evaluate_scalar_subquery()/evaluate_list_subquery() below, since both
// convert one Arrow scalar to one AstLiteral, just a different number of
// times.
sql::AstLiteral scalar_to_literal(const std::shared_ptr<arrow::Scalar>& scalar) {
  if (!scalar->is_valid) {
    return sql::AstLiteral{sql::AstLiteralKind::Null, 0, 0.0, {}, false};
  }
  if (scalar->type->id() == arrow::Type::DOUBLE) {
    return sql::AstLiteral{sql::AstLiteralKind::Float,
                           0,
                           std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value,
                           {},
                           false};
  }
  if (scalar->type->id() == arrow::Type::INT64) {
    return sql::AstLiteral{sql::AstLiteralKind::Integer,
                           std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value,
                           0.0,
                           {},
                           false};
  }
  if (scalar->type->id() == arrow::Type::STRING) {
    return sql::AstLiteral{sql::AstLiteralKind::String, 0, 0.0,
                           std::static_pointer_cast<arrow::StringScalar>(scalar)->value->ToString(), false};
  }
  throw ExecutionError(
      fmt::format("a subquery returned a {} value, which isn't supported (only DOUBLE/INT64/STRING -- the "
                  "result types SUM/COUNT/MIN/MAX/AVG can actually produce, or a plain grouped INT64/DOUBLE/"
                  "STRING column -- are)",
                  scalar->type->ToString()));
}

ResolvedTable inspect_source(ObjectStore& store, const std::vector<std::string>& paths,
                             TableSourceResolver* extra_resolver, double* elapsed_seconds_out) {
  const auto start = std::chrono::steady_clock::now();
  // Empty predicates: this call is schema/partition-column discovery for
  // binding, run before the optimizer has computed any pushable predicates
  // (the WHERE clause isn't bound against a schema yet at this point) --
  // see TableSourceResolver::resolve()'s own doc comment. Physical
  // planning's own resolve (convert_scan(), src/io/physical_planner.cpp)
  // passes the real ones once they exist.
  ResolvedTable resolved = resolve_table_or_delegate(store, paths, extra_resolver, {});
  if (elapsed_seconds_out != nullptr) {
    *elapsed_seconds_out += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
  return resolved;
}
}  // namespace

QueryResult QueryEngine::run_subquery(const sql::AstSelectStatement& subquery_ast) const {
  // Same per-call resolver construction plan_logical()/explain() already
  // use -- see their own comments on why this stays fresh per call today.
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(
      config_.unity_catalog, config_.delta, config_.storage.s3, config_.storage.gcs, config_.storage.azure,
      &unity_catalog_token_cache_);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver, unity_catalog_resolver);

  // nullptr: this subquery's own metadata-inspection time isn't folded
  // into the outer query's QueryResult::metadata_inspection_seconds --
  // it's a genuinely separate nested execution, not part of the outer
  // query's own resolve() work. plan_logical_unoptimized() itself already
  // handles every FROM shape (single table, JOIN chain, derived table)
  // plus this subquery's own nested HAVING/WHERE-IN/EXISTS resolution --
  // see this method's own header comment (query_engine.hpp).
  const LogicalPlanPtr logical =
      optimize(plan_logical_unoptimized(subquery_ast, resolver, /*metadata_inspection_seconds_out=*/nullptr));
  const PhysicalPlanPtr physical = build_physical_plan(logical, store_, &resolver);
  // Always the CPU backend, regardless of config_.engine.backend -- see
  // this method's own header comment (query_engine.hpp) for why nesting a
  // second GPU/RmmEnvironment lifecycle inside plan_logical() isn't worth
  // the risk for what's ultimately one scalar (or one small column) of
  // values.
  return execute_cpu(physical);
}

sql::AstLiteral QueryEngine::evaluate_scalar_subquery(const sql::AstSelectStatement& subquery_ast) const {
  const QueryResult result = run_subquery(subquery_ast);

  if (result.schema == nullptr || result.schema->num_fields() != 1) {
    throw ExecutionError(fmt::format("a HAVING subquery must return exactly one column, got {}",
                                     result.schema != nullptr ? result.schema->num_fields() : 0));
  }
  std::int64_t total_rows = 0;
  std::shared_ptr<arrow::RecordBatch> non_empty_batch;
  for (const std::shared_ptr<arrow::RecordBatch>& batch : result.batches) {
    total_rows += batch->num_rows();
    if (batch->num_rows() > 0 && non_empty_batch == nullptr) {
      non_empty_batch = batch;
    }
  }
  if (total_rows != 1) {
    throw ExecutionError(fmt::format("a HAVING subquery must return exactly one row, got {}", total_rows));
  }

  const arrow::Result<std::shared_ptr<arrow::Scalar>> scalar_result =
      non_empty_batch->column(0)->GetScalar(0);
  if (!scalar_result.ok()) {
    throw ExecutionError(fmt::format("failed to read a HAVING subquery's scalar result: {}",
                                     scalar_result.status().ToString()));
  }
  return scalar_to_literal(*scalar_result);
}

std::vector<sql::AstLiteral> QueryEngine::evaluate_list_subquery(
    const sql::AstSelectStatement& subquery_ast) const {
  const QueryResult result = run_subquery(subquery_ast);

  if (result.schema == nullptr || result.schema->num_fields() != 1) {
    throw ExecutionError(fmt::format("an IN (SELECT ...) subquery must return exactly one column, got {}",
                                     result.schema != nullptr ? result.schema->num_fields() : 0));
  }
  std::vector<sql::AstLiteral> literals;
  for (const std::shared_ptr<arrow::RecordBatch>& batch : result.batches) {
    const std::shared_ptr<arrow::Array>& column = batch->column(0);
    literals.reserve(literals.size() + static_cast<std::size_t>(batch->num_rows()));
    for (std::int64_t row = 0; row < batch->num_rows(); ++row) {
      const arrow::Result<std::shared_ptr<arrow::Scalar>> scalar_result = column->GetScalar(row);
      if (!scalar_result.ok()) {
        throw ExecutionError(fmt::format("failed to read an IN (SELECT ...) subquery's result: {}",
                                         scalar_result.status().ToString()));
      }
      literals.push_back(scalar_to_literal(*scalar_result));
    }
  }
  // An empty result is legitimate (`x IN ()` is always false) -- handled
  // by sql::resolve_in_subqueries() itself, not an error here.
  return literals;
}

LogicalPlanPtr QueryEngine::plan_logical(std::string_view sql,
                                         double* metadata_inspection_seconds_out) const {
  sql::AstSelectStatement ast = sql::parse_sql(sql);
  // Constructed fresh per call -- see IcebergSourceResolver's/
  // DeltaSourceResolver's/UnityCatalogSourceResolver's own comments on why
  // none of the three caches its backing client(s) across queries yet.
  // Cheap regardless: just copies of config_.iceberg's/config_.unity_catalog's
  // catalog maps and config_.delta's/config_.storage.s3's single sections.
  // Shared across every recursive plan_logical_unoptimized() call this one
  // top-level call makes (a derived table's own inner query included), not
  // rebuilt per level.
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(
      config_.unity_catalog, config_.delta, config_.storage.s3, config_.storage.gcs, config_.storage.azure,
      &unity_catalog_token_cache_);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver, unity_catalog_resolver);

  // optimize() runs exactly once, here, over the whole assembled tree --
  // see plan_logical_unoptimized()'s own comment for why.
  return optimize(plan_logical_unoptimized(std::move(ast), resolver, metadata_inspection_seconds_out));
}

LogicalPlanPtr QueryEngine::plan_logical_unoptimized(sql::AstSelectStatement ast,
                                                     TableSourceResolver& resolver,
                                                     double* metadata_inspection_seconds_out) const {
  // Structural, not a resolution-via-execution pass like the two below --
  // see sql::rewrite_exists_subqueries()'s own doc comment. Runs first
  // since it can change ast.where's own top-level shape (extracting
  // EXISTS/NOT EXISTS conjuncts out into ast.join entirely); order
  // relative to the HAVING/IN resolution below doesn't otherwise matter
  // (neither touches AstExists nodes), this is just the more natural
  // "structural rewrite before inline resolution" sequence.
  ast = sql::rewrite_exists_subqueries(std::move(ast));

  if (ast.having != nullptr) {
    // Resolved before binding: the binder has no I/O capability of its
    // own (by design, see ast.hpp's own header comment) and can't run a
    // nested query itself -- see resolve_subqueries()'s own doc comment
    // for why this lives here instead. Any AstSubquery surviving this
    // (i.e. one that wasn't inside HAVING) reaches Binder::bind_node(const
    // AstSubquery&, bool) instead, which rejects it with a clear error.
    ast.having = sql::resolve_subqueries(
        ast.having, [this](const sql::AstSelectStatement& sub) { return evaluate_scalar_subquery(sub); });
  }
  if (ast.where != nullptr) {
    // Same rationale as the HAVING resolution above, run separately since
    // an IN-subquery can legitimately return many rows (unlike HAVING's
    // exactly-one-row/one-column contract) -- see
    // sql::resolve_in_subqueries()'s own doc comment. Any AstIn surviving
    // this with `subquery` still set (i.e. one that wasn't inside WHERE)
    // reaches Binder::bind_node(const AstIn&, bool) instead, which rejects
    // it with a clear error.
    ast.where = sql::resolve_in_subqueries(
        ast.where, [this](const sql::AstSelectStatement& sub) { return evaluate_list_subquery(sub); });
  }

  if (ast.join.has_value()) {
    std::vector<Schema> join_schemas;
    std::vector<std::vector<PartitionColumn>> partition_columns_per_source;
    join_schemas.reserve(ast.join->steps.size() + 1);
    partition_columns_per_source.reserve(ast.join->steps.size() + 1);
    const ResolvedTable first =
        inspect_source(store_, ast.join->first.paths, &resolver, metadata_inspection_seconds_out);
    join_schemas.push_back(first.schema);
    partition_columns_per_source.push_back(first.partition_columns);
    for (const sql::AstJoinStep& step : ast.join->steps) {
      const ResolvedTable resolved =
          inspect_source(store_, step.source.paths, &resolver, metadata_inspection_seconds_out);
      join_schemas.push_back(resolved.schema);
      partition_columns_per_source.push_back(resolved.partition_columns);
    }
    const BoundQuery bound = bind_query(ast, join_schemas);
    return build_logical_plan(bound, join_schemas, std::move(partition_columns_per_source));
  }

  if (ast.from_subquery != nullptr) {
    LogicalPlanPtr inner =
        plan_logical_unoptimized(*ast.from_subquery, resolver, metadata_inspection_seconds_out);
    const BoundQuery bound = bind_query(ast, inner->output_schema());
    return build_logical_plan(bound, std::move(inner));
  }

  const ResolvedTable resolved =
      inspect_source(store_, ast.from.paths, &resolver, metadata_inspection_seconds_out);
  const BoundQuery bound = bind_query(ast, resolved.schema);
  return build_logical_plan(bound, resolved.schema, resolved.partition_columns);
}

LogicalPlanPtr QueryEngine::explain_logical(std::string_view sql) const {
  return plan_logical(sql);
}

PhysicalPlanPtr QueryEngine::explain(std::string_view sql) const {
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(
      config_.unity_catalog, config_.delta, config_.storage.s3, config_.storage.gcs, config_.storage.azure,
      &unity_catalog_token_cache_);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver, unity_catalog_resolver);
  return build_physical_plan(plan_logical(sql), store_, &resolver);
}

std::optional<NvmeCacheMetricsSnapshot> QueryEngine::cache_metrics() const {
  return store_.cache_metrics();
}

void QueryEngine::register_cache_otel_instruments() const {
  register_nvme_cache_otel_instruments(store_);
}

}  // namespace kernellake
