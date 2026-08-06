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

#include "composite_source_resolver.hpp"

namespace kernellake {

QueryEngine::QueryEngine(EngineConfig config) : config_(std::move(config)), store_(config_.storage) {}

namespace {
ResolvedTable inspect_source(ObjectStore& store, const std::vector<std::string>& paths,
                             TableSourceResolver* extra_resolver, double* elapsed_seconds_out) {
  const auto start = std::chrono::steady_clock::now();
  ResolvedTable resolved = resolve_table_or_delegate(store, paths, extra_resolver);
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
  // DeltaSourceResolver's own comments on why neither caches its backing
  // client across queries yet. Cheap regardless: just a copy of
  // config_.iceberg's catalog map and config_.delta's single section.
  iceberg::IcebergSourceResolver iceberg_resolver(config_.iceberg);
  delta::DeltaSourceResolver delta_resolver(config_.delta);
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver);

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
  CompositeSourceResolver resolver(iceberg_resolver, delta_resolver);
  return build_physical_plan(plan_logical(sql), store_, &resolver);
}

}  // namespace kernellake
