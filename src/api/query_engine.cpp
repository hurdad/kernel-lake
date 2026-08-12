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

QueryEngine::QueryEngine(EngineConfig config) : config_(std::move(config)), store_(config_.storage) {}

namespace {
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

sql::AstLiteral QueryEngine::evaluate_scalar_subquery(const sql::AstSelectStatement& subquery_ast) const {
  sql::AstSelectStatement resolved = subquery_ast;
  if (resolved.having != nullptr) {
    resolved.having = sql::resolve_subqueries(resolved.having, [this](const sql::AstSelectStatement& nested) {
      return evaluate_scalar_subquery(nested);
    });
  }

  // Same per-call resolver construction plan_logical()/explain() already
  // use -- see their own comments on why this stays fresh per call today.
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(
      config_.unity_catalog, config_.delta, config_.storage.s3, config_.storage.gcs, config_.storage.azure,
      &unity_catalog_token_cache_);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver, unity_catalog_resolver);

  LogicalPlanPtr logical;
  if (resolved.join.has_value()) {
    std::vector<Schema> join_schemas;
    std::vector<std::vector<PartitionColumn>> partition_columns_per_source;
    join_schemas.reserve(resolved.join->steps.size() + 1);
    partition_columns_per_source.reserve(resolved.join->steps.size() + 1);
    // nullptr: this subquery's own metadata-inspection time isn't folded
    // into the outer query's QueryResult::metadata_inspection_seconds --
    // it's a genuinely separate nested execution, not part of the outer
    // query's own resolve() work.
    const ResolvedTable first = inspect_source(store_, resolved.join->first.paths, &resolver, nullptr);
    join_schemas.push_back(first.schema);
    partition_columns_per_source.push_back(first.partition_columns);
    for (const sql::AstJoinStep& step : resolved.join->steps) {
      const ResolvedTable joined = inspect_source(store_, step.source.paths, &resolver, nullptr);
      join_schemas.push_back(joined.schema);
      partition_columns_per_source.push_back(joined.partition_columns);
    }
    const BoundQuery bound = bind_query(resolved, join_schemas);
    logical = optimize(build_logical_plan(bound, join_schemas, std::move(partition_columns_per_source)));
  } else {
    const ResolvedTable table = inspect_source(store_, resolved.from.paths, &resolver, nullptr);
    const BoundQuery bound = bind_query(resolved, table.schema);
    logical = optimize(build_logical_plan(bound, table.schema, table.partition_columns));
  }

  const PhysicalPlanPtr physical = build_physical_plan(logical, store_, &resolver);
  // Always the CPU backend, regardless of config_.engine.backend -- see
  // this method's own header comment (query_engine.hpp) for why nesting a
  // second GPU/RmmEnvironment lifecycle inside plan_logical() isn't worth
  // the risk for what's ultimately one scalar number.
  const QueryResult result = execute_cpu(physical);

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
  const std::shared_ptr<arrow::Scalar>& scalar = *scalar_result;
  if (!scalar->is_valid) {
    return sql::AstLiteral{sql::AstLiteralKind::Null, 0, 0.0, {}, false};
  }
  // DOUBLE/INT64 are the only two Arrow types this project's own
  // aggregates can actually produce (SUM/AVG -> DOUBLE, COUNT/CountStar
  // -> INT64; MIN/MAX pass their argument's own type through, which for
  // every currently-generated TPC-H column is itself DOUBLE or INT64) --
  // see docs/ARCHITECTURE.md's HAVING section.
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
  throw ExecutionError(fmt::format(
      "a HAVING subquery returned a {} value, which isn't supported (only DOUBLE/INT64 -- the result "
      "types SUM/COUNT/MIN/MAX/AVG can actually produce here -- are)",
      scalar->type->ToString()));
}

LogicalPlanPtr QueryEngine::plan_logical(std::string_view sql,
                                         double* metadata_inspection_seconds_out) const {
  sql::AstSelectStatement ast = sql::parse_sql(sql);
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
  // Constructed fresh per call -- see IcebergSourceResolver's/
  // DeltaSourceResolver's/UnityCatalogSourceResolver's own comments on why
  // none of the three caches its backing client(s) across queries yet.
  // Cheap regardless: just copies of config_.iceberg's/config_.unity_catalog's
  // catalog maps and config_.delta's/config_.storage.s3's single sections.
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(
      config_.unity_catalog, config_.delta, config_.storage.s3, config_.storage.gcs, config_.storage.azure,
      &unity_catalog_token_cache_);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver, unity_catalog_resolver);

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
    LogicalPlanPtr logical = build_logical_plan(bound, join_schemas, std::move(partition_columns_per_source));
    return optimize(logical);
  }

  const ResolvedTable resolved =
      inspect_source(store_, ast.from.paths, &resolver, metadata_inspection_seconds_out);
  const BoundQuery bound = bind_query(ast, resolved.schema);
  LogicalPlanPtr logical = build_logical_plan(bound, resolved.schema, resolved.partition_columns);
  return optimize(logical);
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
