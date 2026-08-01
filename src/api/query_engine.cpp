#include "kernellake/api/query_engine.hpp"

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/optimizer/optimizer.hpp"
#include "kernellake/planner/logical_planner.hpp"
#include "kernellake/sql/parser.hpp"
#include "kernellake/storage/file_discovery.hpp"

namespace kernellake {

QueryEngine::QueryEngine(EngineConfig config) : config_(std::move(config)) {}

LogicalPlanPtr QueryEngine::plan_logical(std::string_view sql) const {
  const sql::AstSelectStatement ast = sql::parse_sql(sql);

  const std::vector<ObjectInfo> files = discover_parquet_files(store_, ast.from.paths);
  std::vector<FileMetadata> metadata;
  metadata.reserve(files.size());
  for (const ObjectInfo& file : files) metadata.push_back(inspect_parquet_file(store_, file.uri));
  validate_schema_compatibility(metadata);
  const Schema source_schema = metadata.front().schema;

  const BoundQuery bound = bind_query(ast, source_schema);
  LogicalPlanPtr logical = build_logical_plan(bound, source_schema);
  return optimize(std::move(logical));
}

LogicalPlanPtr QueryEngine::explain_logical(std::string_view sql) const { return plan_logical(sql); }

PhysicalPlanPtr QueryEngine::explain(std::string_view sql) const {
  return build_physical_plan(plan_logical(sql), store_);
}

}  // namespace kernellake
