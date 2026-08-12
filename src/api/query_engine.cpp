#include "kernellake/api/query_engine.hpp"

#include <chrono>

#include "kernellake/common/errors.hpp"
#include "kernellake/delta/delta_source_resolver.hpp"
#include "kernellake/iceberg/iceberg_source_resolver.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/io/table_resolution.hpp"
#include "kernellake/optimizer/optimizer.hpp"
#include "kernellake/planner/logical_planner.hpp"
#include "kernellake/sql/parser.hpp"
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

LogicalPlanPtr QueryEngine::plan_logical(std::string_view sql,
                                         double* metadata_inspection_seconds_out) const {
  const sql::AstSelectStatement ast = sql::parse_sql(sql);
  // Constructed fresh per call -- see IcebergSourceResolver's/
  // DeltaSourceResolver's/UnityCatalogSourceResolver's own comments on why
  // none of the three caches its backing client(s) across queries yet.
  // Cheap regardless: just copies of config_.iceberg's/config_.unity_catalog's
  // catalog maps and config_.delta's/config_.storage.s3's single sections.
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(config_.unity_catalog, config_.delta,
                                                                  config_.storage.s3);
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
  unitycatalog::UnityCatalogSourceResolver unity_catalog_resolver(config_.unity_catalog, config_.delta,
                                                                  config_.storage.s3);
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
