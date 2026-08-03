#include "kernellake/api/query_engine.hpp"

#include <chrono>

#include "kernellake/common/errors.hpp"
#include "kernellake/io/parquet_metadata.hpp"
#include "kernellake/io/physical_planner.hpp"
#include "kernellake/optimizer/optimizer.hpp"
#include "kernellake/planner/logical_planner.hpp"
#include "kernellake/sql/parser.hpp"
#include "kernellake/storage/file_discovery.hpp"

namespace kernellake {

QueryEngine::QueryEngine(EngineConfig config) : config_(std::move(config)), store_(config_.storage) {}

namespace {
Schema inspect_source_schema(ObjectStore& store, const std::vector<std::string>& paths,
                             double* elapsed_seconds_out) {
  const auto start = std::chrono::steady_clock::now();
  const std::vector<ObjectInfo> files = discover_parquet_files(store, paths);
  std::vector<FileMetadata> metadata;
  metadata.reserve(files.size());
  for (const ObjectInfo& file : files) metadata.push_back(inspect_parquet_file(store, file.uri));
  validate_schema_compatibility(metadata);
  if (elapsed_seconds_out != nullptr) {
    *elapsed_seconds_out += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
  return metadata.front().schema;
}
}  // namespace

LogicalPlanPtr QueryEngine::plan_logical(std::string_view sql,
                                         double* metadata_inspection_seconds_out) const {
  const sql::AstSelectStatement ast = sql::parse_sql(sql);

  if (ast.join.has_value()) {
    const Schema left_schema =
        inspect_source_schema(store_, ast.join->left.paths, metadata_inspection_seconds_out);
    const Schema right_schema =
        inspect_source_schema(store_, ast.join->right.paths, metadata_inspection_seconds_out);
    const BoundQuery bound = bind_query(ast, left_schema, right_schema);
    LogicalPlanPtr logical = build_logical_plan(bound, left_schema, &right_schema);
    return optimize(std::move(logical));
  }

  const Schema source_schema = inspect_source_schema(store_, ast.from.paths, metadata_inspection_seconds_out);
  const BoundQuery bound = bind_query(ast, source_schema);
  LogicalPlanPtr logical = build_logical_plan(bound, source_schema);
  return optimize(std::move(logical));
}

LogicalPlanPtr QueryEngine::explain_logical(std::string_view sql) const {
  return plan_logical(sql);
}

PhysicalPlanPtr QueryEngine::explain(std::string_view sql) const {
  return build_physical_plan(plan_logical(sql), store_);
}

}  // namespace kernellake
