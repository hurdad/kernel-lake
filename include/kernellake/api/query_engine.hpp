#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/planner/logical_plan.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/sql/ast.hpp"
#include "kernellake/storage/object_store_registry.hpp"
#include "kernellake/types/schema.hpp"
#include "kernellake/unitycatalog/unity_catalog_token_cache.hpp"

namespace arrow {
class Schema;
class RecordBatch;
}  // namespace arrow

namespace kernellake {

class RmmEnvironment;
class TableSourceResolver;

// Execution and I/O metrics for one query. Every metric KernelLake cannot
// yet measure (because execution requires GPU/libcudf, not yet built --
// see docs/ARCHITECTURE.md) stays std::nullopt rather than being guessed at,
// per the spec's "documented null value, never an invented measurement"
// rule.
struct QueryResult {
  std::shared_ptr<arrow::Schema> schema;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;

  std::optional<double> elapsed_wall_seconds;
  std::optional<std::int64_t> rows_scanned;
  std::optional<std::int64_t> rows_returned;
  std::optional<std::int64_t> files_considered;
  std::optional<std::int64_t> files_scanned;
  std::optional<std::int64_t> row_groups_considered;
  std::optional<std::int64_t> row_groups_scanned;
  std::optional<std::int64_t> compressed_bytes_read;
  std::optional<std::int64_t> estimated_uncompressed_bytes;
  std::optional<double> metadata_inspection_seconds;
  std::optional<double> parquet_decoding_seconds;
  std::optional<double> gpu_execution_seconds;
  std::optional<double> host_to_device_seconds;
  std::optional<double> device_to_host_seconds;
  std::optional<std::int64_t> peak_gpu_memory_bytes;
  // Populated only by the CPU (Acero) execution backend -- kept as its own
  // field rather than overloading gpu_execution_seconds, since the two
  // backends measure genuinely different work (a single Acero
  // DeclarationToTable() call vs. GPU operator pull-loop + device-to-host
  // transfer) and conflating them under one name would misrepresent which
  // backend actually ran.
  std::optional<double> cpu_execution_seconds;
};

// The top-level entry point described in the spec: SQL in, either a plan
// (for explain_logical/explain) or a result (for execute) out.
//
// explain_logical() and explain() bind against the real Parquet schema of
// the query's FROM read_parquet(...) source (discovered and inspected via
// LocalObjectStore), so they fully exercise parsing, binding, logical
// planning, optimization, file discovery, and pruning -- everything short
// of actually decoding column data and running GPU operators.
//
// execute() actually runs the query on the GPU when built with
// KERNELLAKE_WITH_CUDA=ON (see query_engine_execute_gpu.cpp): it builds the
// physical-plan operator tree (kernellake/execution/operator_builder.hpp),
// pulls DeviceBatch results through it, and converts each to a host-side
// Arrow RecordBatch. In a CPU-only build (KERNELLAKE_WITH_CUDA=OFF, the
// `dev` preset), execute() throws ExecutionError instead
// (query_engine_execute_stub.cpp) -- KernelLake never substitutes a CPU
// implementation for GPU execution without saying so explicitly.
class QueryEngine {
 public:
  // `device_id` is which CUDA device the execute(sql) one-shot overload's
  // own RmmEnvironment targets -- not part of EngineConfig (see its own
  // comment: which device is a runtime dispatch parameter, not a shared
  // config concern). Defaults to 0 so every existing caller that just
  // wants "an engine" (tests, and any caller that never calls execute(sql)
  // at all -- e.g. kernellake-server, which always goes through
  // explain()+execute(physical, rmm_environment)/execute_cpu() instead)
  // needs no changes. The CLI passes its own CliConfig::device_id here.
  explicit QueryEngine(EngineConfig config, int device_id = 0);

  // Returns the optimized logical plan (i.e. the plan the physical planner
  // would actually receive), not the pre-optimization one -- use this to
  // see what KernelLake decided the query means after its rewrite rules.
  [[nodiscard]] LogicalPlanPtr explain_logical(std::string_view sql) const;

  [[nodiscard]] PhysicalPlanPtr explain(std::string_view sql) const;

  // Convenience entry point for one-shot callers (the CLI): plans and
  // executes `sql` end to end, building and tearing down its own
  // RmmEnvironment for the duration of this call. Fine for a one-query-
  // per-process model; a long-lived caller (e.g. a server handling many
  // requests) should prefer explain() + the execute(physical, rmm_environment)
  // overload below instead, reusing one RmmEnvironment across requests --
  // see docs/ARCHITECTURE.md's Concurrency notes for why rebuilding the RMM
  // pool per request is both wrong under concurrency and wasteful even
  // single-threaded.
  [[nodiscard]] QueryResult execute(std::string_view sql) const;

  // Runs an already-built physical plan's GPU execution against an
  // *externally owned* RmmEnvironment (constructed and torn down by the
  // caller, once, not per call). `result.metadata_inspection_seconds` stays
  // nullopt here -- planning (where that time is actually spent) already
  // happened before this call, in whatever produced `physical` (e.g.
  // explain()); a caller that wants it should time its own explain() call.
  [[nodiscard]] QueryResult execute(const PhysicalPlanPtr& physical, RmmEnvironment& rmm_environment) const;

  // Runs an already-built physical plan on the Apache Arrow Acero CPU
  // execution backend (see docs/ARCHITECTURE.md's CPU backend section).
  // Always available, in both the `dev` and `gpu-dev` presets -- unlike the
  // execute(physical, rmm_environment) overload above, this needs no CUDA
  // and takes no external resource, since Acero owns its own thread pool
  // internally. Throws PlanningError/ExecutionError for physical plan nodes
  // this backend doesn't yet support (e.g. HashJoin). Like the GPU overload,
  // leaves metadata_inspection_seconds null (planning already happened
  // before this call).
  [[nodiscard]] QueryResult execute_cpu(const PhysicalPlanPtr& physical) const;

  // nullopt when storage.cache.enabled is false. See NvmeObjectCache::
  // snapshot() (kernellake/storage/nvme_object_cache.hpp) for field
  // semantics -- cumulative hits/misses/evictions since this QueryEngine
  // was constructed, live current_bytes/current_entries gauges.
  [[nodiscard]] std::optional<NvmeCacheMetricsSnapshot> cache_metrics() const;

  // Registers this engine's cache metrics as OTel instruments under
  // "kernellake.storage.cache.*" -- a no-op unless built with
  // KERNELLAKE_ENABLE_OTEL and storage.cache.enabled. Call at most once,
  // from a caller whose own lifetime already outlives `this`
  // (kernellake-server's constructor, right after constructing its
  // QueryEngine member -- see docs/ARCHITECTURE.md's "NVMe cache tier"
  // section for why the CLI doesn't call this).
  void register_cache_otel_instruments() const;

 private:
  // `metadata_inspection_seconds_out`, when non-null, accumulates the time
  // spent discovering/inspecting each FROM source's Parquet metadata (the
  // JOIN case inspects two sources, hence "accumulates" rather than
  // "assigns"). Left null by explain_logical()/explain(), which don't
  // return a QueryResult to put it in.
  [[nodiscard]] LogicalPlanPtr plan_logical(std::string_view sql,
                                            double* metadata_inspection_seconds_out = nullptr) const;

  // The recursive body plan_logical() wraps with a single optimize() call
  // at the very top. Resolves `ast`'s own HAVING/WHERE-IN subqueries, then
  // binds+builds (but does not optimize) its logical plan, dispatching on
  // whether `ast.from` is a plain source, a JOIN chain, or a derived table
  // (`FROM (SELECT ...) AS alias`) -- the last case recurses into this same
  // method for the inner query first, whose own (unoptimized)
  // LogicalPlanPtr becomes the outer query's source and whose own
  // BoundQuery::output_schema becomes the outer query's "table" to bind
  // against (see logical_planner.cpp's build_logical_plan(query,
  // source_plan) overload and binder.hpp's single-table bind_query()
  // overload, both reused as-is -- a derived table's output looks exactly
  // like a real source's schema to either). `resolver` is threaded through
  // rather than constructed fresh per recursive call, so a derived table's
  // own inner query shares plan_logical()'s one Iceberg/Delta/Unity
  // Catalog resolver instance instead of building a redundant copy.
  // Optimization is deliberately deferred to plan_logical() itself (called
  // exactly once, over the fully assembled tree) rather than run here per
  // recursive call, both to avoid the question of whether optimize() is
  // safe to run twice over an already-optimized subtree, and so passes
  // like predicate pushdown see the whole tree, inner and outer query
  // alike, at once.
  [[nodiscard]] LogicalPlanPtr plan_logical_unoptimized(sql::AstSelectStatement ast,
                                                        TableSourceResolver& resolver,
                                                        double* metadata_inspection_seconds_out) const;

  // Runs `subquery_ast` as a genuinely separate, complete query (bind,
  // logical-plan, optimize, physical-plan, execute) on the CPU (Acero)
  // backend, always -- regardless of the outer query's own configured
  // backend, see docs/ARCHITECTURE.md's HAVING section for why. Delegates
  // straight to plan_logical_unoptimized() (the same recursive planner a
  // real top-level query goes through) rather than reimplementing its own
  // narrower join-or-single-table planning, so a subquery whose own FROM
  // is itself a derived table (TPC-H Q15's shape: `HAVING total_revenue =
  // (SELECT MAX(total_revenue) FROM (SELECT ... GROUP BY l_suppkey) AS
  // r2)`) works the same way a top-level derived-table query already
  // does -- plan_logical_unoptimized() also recursively resolves this
  // subquery's own nested HAVING/WHERE-IN subqueries and any EXISTS/NOT
  // EXISTS along the way, so a subquery nested inside a subquery works for
  // free regardless of which kind it is. (Fixed 2026-08-24: the previous
  // hand-rolled version had no case for `from_subquery` at all, silently
  // falling through to the single-table branch with an empty path list --
  // "no data source given" at execution time for exactly this shape.)
  [[nodiscard]] QueryResult run_subquery(const sql::AstSelectStatement& subquery_ast) const;

  // Returns `subquery_ast`'s single scalar result as an AstLiteral -- see
  // docs/ARCHITECTURE.md's HAVING section for the full scope
  // (non-correlated, single-row/single-column only). Called by
  // plan_logical() (via sql::resolve_subqueries()) to resolve every
  // AstSubquery reachable from a HAVING clause before the outer query is
  // bound -- the physical plan's own HAVING filter needs a real literal
  // threshold, not a placeholder. Throws ExecutionError if the result
  // isn't exactly one row and one column, or if that column's Arrow type
  // isn't one of the few this project's aggregates can actually produce.
  [[nodiscard]] sql::AstLiteral evaluate_scalar_subquery(const sql::AstSelectStatement& subquery_ast) const;

  // Returns `subquery_ast`'s single-column result as a list of AstLiterals
  // (possibly empty -- see sql::resolve_in_subqueries()'s own handling of
  // that case). Called by plan_logical() (via sql::resolve_in_subqueries())
  // to resolve every IN-subquery reachable from a WHERE clause before the
  // outer query is bound -- TPC-H Q18's shape
  // (`o_orderkey IN (SELECT l_orderkey FROM ... HAVING SUM(...) > 300)`).
  // Throws ExecutionError if the result isn't exactly one column, or if
  // that column's Arrow type isn't one this project can convert to a
  // literal. Deliberately narrow: the returned list becomes an OR-chain of
  // equality comparisons at bind time (the same desugar a literal IN list
  // already gets), so this is scoped to a subquery expected to return a
  // modest number of rows, not a general-purpose semi-join -- see
  // sql::resolve_in_subqueries()'s own doc comment.
  [[nodiscard]] std::vector<sql::AstLiteral> evaluate_list_subquery(
      const sql::AstSelectStatement& subquery_ast) const;

  EngineConfig config_;
  // Which device execute(sql)'s one-shot RmmEnvironment targets -- see the
  // constructor's own comment.
  int device_id_ = 0;
  // Declared after config_ (member init order follows declaration order):
  // ObjectStoreRegistry keeps a reference to config_.storage, valid for
  // QueryEngine's whole lifetime since both are members of the same object.
  mutable ObjectStoreRegistry store_;
  // Shared across every UnityCatalogSourceResolver this QueryEngine
  // constructs (plan_logical()/explain()/execute(sql), each of which
  // builds its own resolver instance today -- see those methods' own
  // comments) so a Unity Catalog instance's OAuth2 token, once fetched,
  // survives past the one resolve() call that fetched it -- both across
  // the multiple resolves a single query already makes and across
  // separate queries against the same QueryEngine. Not `mutable`: every
  // public method of UnityCatalogTokenCache is const and internally
  // synchronized (see its own class comment), so this member needs no
  // extra help staying safely touchable from QueryEngine's own const
  // methods, the same way `store_` above already relies on
  // ObjectStoreRegistry's own internal synchronization rather than being
  // plain-const itself.
  unitycatalog::UnityCatalogTokenCache unity_catalog_token_cache_;
};

}  // namespace kernellake
