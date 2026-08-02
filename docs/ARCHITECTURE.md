# KernelLake architecture

KernelLake is a compute and query layer over Apache Iceberg/Parquet data
lakes: it executes analytical SQL using Apache Arrow-compatible columnar
data as its in-memory representation and NVIDIA GPUs for accelerated
execution. It is not a storage database.

## Pipeline

```
SQL text
  -> SQL parser              (kernellake::sql)
  -> binder / type checker   (kernellake::bind_query)
  -> logical plan            (kernellake::LogicalPlanNode)
  -> rule-based optimizer    (kernellake::optimize)
  -> file discovery          (kernellake::ObjectStore, discover_parquet_files)
  -> Parquet metadata + pruning (kernellake::inspect_parquet_file, evaluate_pruning)
  -> physical plan           (kernellake::PhysicalPlanNode, build_physical_plan)
  -> GPU execution           (kernellake::PhysicalOperator / DeviceBatch, kernellake_execution)
  -> Arrow result
```

Every stage is implemented and covered by tests: parsing through physical
planning by `tests/unit/`, GPU execution by `tests/gpu/` (see "CPU/GPU build
split" below for why GPU execution lives in a separate test binary and
CMake preset rather than `tests/unit/`).

## Module layout

| Module | Contents |
| --- | --- |
| `common` | Error hierarchy, identifiers, config loading, logging, date parsing |
| `types` | Internal `TypeId`/`DataType`/`Schema`, Arrow adapters |
| `expression` | Typed expression tree (`Expression` and subclasses) |
| `sql` | Parser-independent AST (`kernellake::sql::AstSelectStatement`) and the parser adapter around the vendored `hyrise/sql-parser` |
| `planner` | Binder, logical plan + logical planner, physical plan node definitions |
| `optimizer` | Rule-based logical plan rewriting |
| `storage` | `ObjectStore`/`LocalObjectStore`, file discovery |
| `io` | Parquet metadata inspection, row-group pruning, the physical planner (ties `planner` + `storage` + `io` together) |
| `memory` | RAII CUDA device/stream wrappers, RMM memory-pool/statistics/limit configuration (`gpu-dev` preset only) |
| `execution` | `PhysicalOperator`/`DeviceBatch`/`ExecutionContext`, the Arrow<->cudf bridge, the AST expression compiler, and every concrete GPU operator (`gpu-dev` preset only) |
| `generator` | `kernellake generate-data`'s deterministic synthetic dataset generator (CPU-only, no CUDA dependency) |
| `api` | `QueryEngine`, the top-level entry point |
| `cli` | The `kernellake` executable and its subcommands |

Dependency direction: `storage` has no dependency on `planner`; `planner`
depends on `storage` only for the `Uri` type used in scan fragments; `io`
depends on both `planner` and `storage`. This keeps the object-store
abstraction independent of the query compiler while letting the physical
planner (which lives in `io`) consume both.

## Why the SQL parser needs a `read_parquet(...)` adapter

KernelLake vendors `hyrise/sql-parser` (MIT license, pinned commit) via
CMake `FetchContent` rather than hand-writing a parser. That grammar's
`FROM` clause accepts table names, joins, and subqueries -- but has no
table-valued-function syntax. `kernellake::sql::parse_sql()` therefore
finds every occurrence of `read_parquet('path' [, 'path2', ...])` in the
query text with a regex, extracts each one's path arguments, and
substitutes a distinct placeholder identifier for each occurrence, leaving
the surrounding syntax (`JOIN`/`ON`/aliases/commas) completely untouched --
hsql's own grammar still parses the real table-reference/join structure
around those placeholders. `parse_sql()` then walks the resulting
`fromTable` and only accepts two shapes: a single placeholder (the
single-table MVP case) or exactly one `INNER JOIN ... ON` between two
placeholders, both aliased (see "Hash joins" below); anything else (a real
table name, a subquery, `LEFT`/`RIGHT`/`FULL`/`CROSS` JOIN, a comma-style
join, 3+ `read_parquet(...)` sources) fails with a clear `SqlError` rather
than being silently reinterpreted. This is a narrow, deliberately limited
syntax adapter, not general SQL-string rewriting -- optimizer rules always
operate on the structured plan/expression trees, never on SQL text.

## Supported SQL grammar (current)

```
SELECT <items> FROM read_parquet('path' [, 'path2', ...])
  [WHERE <expr>] [GROUP BY <cols>] [ORDER BY <cols>] [LIMIT <n>]

SELECT <items> FROM read_parquet(...) AS a JOIN read_parquet(...) AS b ON <a.col = b.col>
  [WHERE <expr>] [GROUP BY <cols>] [ORDER BY <cols>] [LIMIT <n>]
```

- Column references, aliases, `*`
- Numeric, string, boolean, date (`DATE 'YYYY-MM-DD'`), and `NULL` literals
- Arithmetic (`+ - * /`), comparisons, `AND`/`OR`/`NOT`, `BETWEEN`,
  `IS [NOT] NULL`
- `LIKE`/`NOT LIKE` (SQL `%`/`_` wildcards; scoped to top-level `WHERE`
  AND-conjuncts -- not inside `OR`, the `SELECT` list, or aggregate
  arguments)
- `IN (literal, ...)`/`NOT IN (...)` (desugared at bind time into an
  equivalent `OR`/`AND` chain of equality comparisons -- no new GPU
  execution support needed, and no scalar-subquery form `IN (SELECT ...)`)
- `CASE WHEN ... THEN ... [WHEN ...] [ELSE ...] END`, both simple
  (`CASE x WHEN ...`) and searched forms (scoped to the `SELECT` list and
  `GROUP BY` keys -- not `WHERE` or aggregate arguments)
- `CAST(expr AS type)` (`INT`/`BIGINT`/`DOUBLE`/`VARCHAR(n)`/`DECIMAL(p, s)`;
  see "LIKE/IN/CASE/CAST implementation notes" below for the
  truncate-vs-round caveat on numeric-to-integer casts, and "DECIMAL
  support" below for `DECIMAL`'s own scope)
- `DECIMAL(p, s)` columns and literals -- see "DECIMAL support" below
- Aggregates: `SUM`, `COUNT`, `COUNT(*)`, `MIN`, `MAX`, `AVG` (`AVG` does
  not support a `DECIMAL` argument; see "DECIMAL support")
- A two-table `INNER JOIN ... ON` with a single equality key (see "Hash
  joins" below for the full scope)

Not yet supported (fails clearly rather than being silently reinterpreted):
`DISTINCT`, `HAVING`, set operations (`UNION`/etc.), `WITH`/CTEs,
subqueries, `OFFSET`, window functions, `LIKE` outside a top-level `WHERE`
AND-conjunct, `CASE` in `WHERE`/aggregate arguments, any function other than
the five aggregates above, comma-style joins, `LEFT`/`RIGHT`/`FULL`/`CROSS`
JOIN, multi-key or non-equality join conditions, and 3+-table joins.

`GROUP BY <name>` resolves `<name>` against the base-table schema first,
then falls back to matching a `SELECT`-list output alias -- this is what
lets you `GROUP BY` a computed expression like `CASE ... AS bucket` (which
has no column name of its own): `SELECT CASE ... AS bucket, COUNT(*) FROM
... GROUP BY bucket` groups by the alias, not a base column named `bucket`.
If both exist, the base column wins (standard SQL alias-shadowing
behavior). The alias-defining `SELECT` item is exempted from the
ungrouped-column check that normally rejects non-aggregated,
non-grouped-by expressions in an aggregate query's `SELECT` list.

`ORDER BY` executes for real via `SortOperator` (`cudf::stable_sorted_order`
+ `cudf::gather`, blocking -- it consumes the whole input before producing
its one output batch, so its memory footprint is the whole result set, not
bounded like the streaming operators). On a non-aggregate query it can
reference any column, bound against the base-table schema. After `GROUP
BY` it is scoped to a plain SELECT-list output name (e.g. `ORDER BY total`)
-- resolved against the query's final output schema rather than the base
table, since an aggregate alias doesn't exist as a column until after
aggregation -- not an arbitrary re-derived expression; `ORDER BY
<expression not in the SELECT list>` after `GROUP BY` fails with a clear
`BindingError` rather than being silently reinterpreted.

## Logical plan and the optimizer

Logical plan nodes: `LogicalScan`, `LogicalJoin`, `LogicalFilter`,
`LogicalProjection`, `LogicalAggregate`, `LogicalSort`, `LogicalLimit`.
`LogicalJoin` is the only binary node (see "Hash joins" above); every other
node, including `LogicalScan`, is a leaf or unary. `LogicalScan` always
starts with every column and predicate the schema/query could reference;
`kernellake::optimize()` then:

- folds constants and simplifies boolean expressions bottom-up
- rewrites `BETWEEN a AND b` into `(>= a) AND (<= b)`
- combines adjacent filters and removes filters that fold to `TRUE`
  (a filter that folds to `FALSE` is kept but annotated
  `estimated_rows = 0`, since there is no dedicated empty-result plan node)
- removes an identity `LogicalProjection` (one that just passes its child's
  columns through unchanged)
- pushes `LIMIT` down through any chain of pass-through projections
- annotates `LogicalScan::required_columns()` with the minimal column set
  the plan actually references, **without** reindexing the scan's schema or
  the expressions above it -- narrowing column *positions* would require
  rewriting every `ColumnExpression` in the tree. Instead the physical
  planner (in `io`) builds the narrowed physical schema and column list
  when it converts `LogicalScan` into `ParquetScanNode`, which is the
  natural point where physical batch layout is decided anyway.
- annotates `LogicalScan::pushable_predicates()` with `column OP literal`
  conjuncts extracted from the `WHERE` clause, for Parquet pruning to
  consume without re-walking the filter expression tree

An aggregate query's `LogicalAggregate` always emits columns in
`[group_by..., aggregates...]` order; a `LogicalProjection` is added on top
to reorder/rename to whatever order the `SELECT` list actually requested
(removed by the redundant-projection rule when it happens to already match).

## Parquet metadata, pruning, and the physical plan

`inspect_parquet_file()` reads a file's footer (schema, row count, row
groups, per-row-group column statistics) without decoding any column data.
`evaluate_pruning()` compares `LogicalScan::pushable_predicates()` against
each row group's min/max statistics and only ever skips a row group when a
predicate is *proven* impossible to satisfy -- missing, partial, or
type-incomparable statistics always fall back to "must scan". Correctness
takes priority over aggressive pruning; see `ScanDecision::reasons` for the
human-readable explanation of every skip.

`build_physical_plan()` (in `io/physical_planner.cpp`) resolves a
`LogicalScan`'s source-path specs into concrete files via `ObjectStore`,
inspects and prunes each one, and maps every other logical node onto its
physical equivalent (`LogicalAggregate` becomes `HashAggregate` when it has
a `GROUP BY`, or `ScalarAggregate` when it does not; `LogicalSort` becomes
`SortNode`, see `SortOperator` below), always wrapping the result in
`ArrowResultNode`.

## CPU/GPU build split

`KERNELLAKE_WITH_CUDA` (off by default) gates everything that needs CUDA,
libcudf, and RMM: `DeviceBatch` (the GPU-resident batch wrapping
`cudf::table`), `ExecutionContext`, the `PhysicalOperator` streaming
interface, and the concrete GPU operators (`kernellake_execution`,
`kernellake_memory`; see `tests/gpu/`). This gate is about the GPU path
specifically, not about whether queries can execute at all on a CPU-only
build -- see "CPU execution backend" below for the real, always-available
CPU path (`kernellake_execution_cpu`), which needs none of this.

`QueryEngine::execute()` is built from one of two mutually exclusive
translation units selected by a CMake generator expression on
`KERNELLAKE_WITH_CUDA` (`src/api/query_engine_execute_gpu.cpp` vs.
`_stub.cpp`), plus a third, **always-built** translation unit
(`query_engine_execute_cpu.cpp`) providing `execute_cpu()`. The GPU build's
`execute(sql)` runs the real GPU operator pipeline by default
(`kernellake/execution/operator_builder.hpp` turns a `PhysicalPlanPtr` into
a `PhysicalOperator` tree, pulled to exhaustion inside
`RmmEnvironment::track_query()` for per-query memory accounting, with each
resulting `DeviceBatch` converted to an Arrow `RecordBatch` via
`kernellake/execution/arrow_bridge.hpp`); the CPU-only `dev` preset's stub
throws a clear `ExecutionError` instead -- **unless** `engine.backend` (or
`kernellake query --backend`) is set to `"cpu"`, in which case both builds
instead dispatch to `execute_cpu()`. The `kernellake query` CLI command
(`src/cli/query_command.cpp`) is unconditionally built and calls
`QueryEngine::execute()`, so which of these three paths actually runs
depends on both which preset built it and the `backend` setting.

Everything else in this document (parsing through physical planning and
pruning), plus `kernellake generate-data`, is CPU-only and builds/tests
with the `dev` CMake preset alone.

### Concurrency: `RmmEnvironment`'s lifetime and the split execution API

`QueryEngine::execute(std::string_view sql)` is a convenience wrapper: it
plans, constructs its own `RmmEnvironment` (installing itself as the
process's current CUDA device memory resource for the call's duration),
executes, and tears that `RmmEnvironment` down again -- all inside one
call. This is correct for a one-query-per-process model (the CLI) but not
for a long-lived process serving many requests: two threads racing
`RmmEnvironment` construction/destruction against the single process-wide
current-device-resource slot is a real use-after-free risk (the same bug
*class*, for a different trigger, as the sequential-test-runs race already
found and fixed -- see "Done" in `docs/ROADMAP.md`), and rebuilding an
entire RMM pool per request is wasteful even single-threaded.

For that reason `QueryEngine` also exposes a split entry point:
`explain(sql) -> PhysicalPlanPtr` (already reentrant -- parsing, binding,
logical planning, optimization, and Parquet metadata inspection touch no
shared mutable state) followed by
`execute(const PhysicalPlanPtr&, RmmEnvironment&) -> QueryResult`, which
takes an **externally owned** `RmmEnvironment` instead of building its own.
A long-lived caller should construct exactly one `RmmEnvironment` at
startup, reuse it across every request, and serialize calls to this split
`execute()` through a single-flight queue (only the GPU-touching phase
needs serializing; planning stays fully concurrent). `execute(sql)` itself
is implemented in terms of this split pair -- it is not two independent
code paths to keep in sync.

The split entry point cannot honestly populate
`QueryResult::metadata_inspection_seconds` (planning already happened
before it was called, in whatever produced the `PhysicalPlanPtr`) -- it
stays `nullopt` there, matching the "documented null, never an invented
measurement" rule. `execute(sql)` measures it itself around its own
`explain()`-equivalent call.

### Real-time instrumentation: `MetricsRegistry` and NVTX

Every node `operator_builder.cpp`'s `build_operator_tree()` returns is
wrapped in `InstrumentedOperator`, a generic decorator that times each
`next()` call and (when `ExecutionContext::metrics` is non-null) records it
into a `MetricsRegistry` keyed by `PhysicalOperator::name()` -- no
individual operator implements its own timing. Recorded totals are
*inclusive* of children (a parent's `next()` call naturally invokes its
already-separately-instrumented child's `next()` internally), so only a
leaf operator's (e.g. `ParquetScanOperator`'s) total is true self time;
`QueryResult::parquet_decoding_seconds` uses exactly that leaf total.
`QueryResult::gpu_execution_seconds`/`device_to_host_seconds` are measured
independently as plain wrapping timers around the operator-tree pull loop
and each `to_arrow_record_batch()` call, respectively, rather than via the
registry -- there is no meaningful "device to host" or "whole tree"
*operator* to attribute those to.

`QueryResult::host_to_device_seconds` stays `nullopt`: there is no separate
host-to-device transfer phase to time in the current architecture (cudf's
chunked Parquet reader decodes host file bytes directly into device memory
in one call), not a gap that was simply missed.

`ProfilingSection::nvtx` (in `EngineConfig`) makes the same
`InstrumentedOperator` wrapper additionally emit an `nvtx3::scoped_range`
per `next()` call, visible in `nsys`/Nsight Systems timelines -- one
instrumentation point serves both the internal `MetricsRegistry` and
external NVTX profiling.

### GPU operators

| Operator | Notes |
| --- | --- |
| `ParquetScanOperator` | `cudf::io::chunked_parquet_reader`, bounded by `pass_read_limit_bytes` (not row count) |
| `FilterOperator` | `cudf::compute_column` + `cudf::apply_boolean_mask` over a compiled AST predicate |
| `ProjectionOperator` | Compiled AST per computed item; a bare column reference is copied directly instead (see below) |
| `ScalarAggregateOperator` | No `GROUP BY`: `cudf::reduce` with its `init` parameter folds each batch into a running scalar (SUM/MIN/MAX/AVG numerator); COUNT/AVG's denominator is a host-side counter. Empty input produces NULL, not zero, except `COUNT(*)`/`COUNT(x)` which produce 0. |
| `HashAggregateOperator` | `GROUP BY`: `cudf::groupby::streaming_groupby` accumulates partial groups across batches within `max_distinct_keys` (default 10M), `finalize()`d on last `next()` |
| `LimitOperator` | `cudf::slice` + the `cudf::table` copy constructor to truncate the final batch |
| `SortOperator` | `ORDER BY`: **blocking**, unlike every operator above -- consumes `child` to exhaustion, concatenates every batch (`cudf::concatenate`) into one table, then `cudf::stable_sorted_order` + `cudf::gather`. Memory footprint is the whole result set, not bounded like the streaming operators. |
| `HashJoinOperator` | Two-table `INNER JOIN`: its *build* (right) side is **blocking** like `SortOperator` (consumed to exhaustion, concatenated into one table, then wraps a single `cudf::hash_join`); its *probe* (left) side streams normally, one `inner_join()` + double-`cudf::gather` per batch. See "Hash joins" above. |
| `ArrowResultOperator` | Trivial passthrough; the actual `DeviceBatch` -> `arrow::RecordBatch` conversion happens in `QueryEngine::execute()` via `to_arrow_record_batch()`, since `PhysicalOperator::next()` must return a `DeviceBatch` |

Two correctness details worth knowing if you touch these operators:

- **Plain column references never go through `cudf::ast::compute_column`.**
  cudf's AST evaluator can only materialize fixed-width output columns, so
  routing a bare STRING column reference through it (e.g. `GROUP BY region`,
  or `SELECT region`) aborts with "Invalid, non-fixed-width type" even
  though no computation was requested. `ProjectionOperator`,
  `HashAggregateOperator` (group-by keys and `COUNT(*)`'s materialized
  argument), and `ScalarAggregateOperator` (aggregate arguments) all detect
  a plain `ColumnExpression` and copy that column directly instead.
- **The physical planner remaps `ColumnExpression` indices above the scan.**
  The binder resolves every column reference against the query's one full
  (unpruned) schema; `build_physical_plan()`'s `convert_scan()` then
  narrows the *physical* scan to only the columns actually referenced,
  which shifts column positions whenever a non-trailing column is dropped.
  `physical_planner.cpp`'s `remap_columns()`/`remap_named()` rewrite every
  filter predicate, projection item, and aggregate/group-by expression
  above the scan to the narrowed schema's actual layout -- without this, a
  query like `SELECT SUM(amount) ...` (where an earlier column got pruned
  away) would index past the end of the batch the scan operator actually
  produces.
- `cudf::groupby`'s COUNT aggregations always produce `INT32` regardless of
  requested type, but KernelLake's binder declares `COUNT`/`COUNT(*)` as
  `INT64`; `HashAggregateOperator` casts those result columns after
  `finalize()` so the output `DeviceBatch`'s actual column types match its
  declared schema. `ScalarAggregateOperator` doesn't need this -- `cudf::reduce`
  honors the requested output type for `COUNT`.
- **`LogicalProjection`/`LogicalSort` conversion must not always remap
  against the scan.** Most physical nodes (`Filter`, `Aggregate`) sit at a
  fixed structural position, always directly above the scan, so their
  expressions always reference the base-table schema and always need
  `remap_columns()`. `Projection` and `Sort` are different: each can sit
  either directly on `Filter`/`Scan` (references the base-table schema,
  needs remapping) *or* directly on `Aggregate` (its reprojection's/sort
  key's expressions already reference the aggregate's own output schema
  one-for-one, and remapping them against the scan fails outright, since
  an aggregate alias like `total` was never a real scanned column).
  `physical_planner.cpp`'s `convert()` discriminates by checking whether
  the node's child is specifically `LogicalFilter`/`LogicalScan` (a
  positive match on the case that needs remapping) rather than checking
  "child is not `LogicalAggregate`" -- the latter looked equivalent but
  isn't: the optimizer's redundant-projection-removal rule can delete an
  aggregate query's reprojection when the SELECT list already matches the
  aggregate's natural column order, which made an earlier, negative-check
  version of this discriminator (checking only for `LogicalProjection`)
  wrongly treat a `Sort` sitting directly on `LogicalAggregate` as if it
  needed scan-schema remapping.

### CPU execution backend (Apache Arrow Acero)

Unlike the GPU path, the CPU backend is not a set of hand-rolled operators
pulled batch-by-batch: `src/execution_cpu/acero_query_executor.cpp`
translates a `PhysicalPlanPtr` directly into an `arrow::acero::Declaration`
tree (Arrow's own CPU-native streaming execution engine) and runs it in one
shot via `arrow::acero::DeclarationToTable()`, which returns an Arrow
`arrow::Table` with no separate device-to-host bridging step needed. This
module (`kernellake_execution_cpu`) is **always built**, in both the `dev`
and `gpu-dev` presets, and depends on nothing CUDA-related -- selecting it
is a runtime choice (`engine.backend: "cpu"`, or `kernellake query --backend
cpu`), not a build-time one, and it works even on the CPU-only `dev` build,
which cannot run the GPU path at all.

**Scope for this phase**, matching the GPU engine's own original MVP order:
`ParquetScan` -> `Filter` -> `Projection` (arithmetic/comparisons/`BETWEEN`/
numeric `CAST` only) -> `ScalarAggregate`/`HashAggregate` (`SUM`/`COUNT`/
`MIN`/`MAX`/`AVG`, grouping only by a plain column) -> `Sort` (by a plain
column only) -> `Limit`. `LIKE`/`IN`/`CASE`/`CAST`-to-`DECIMAL`-or-`STRING`/
`HashJoin` are not yet supported here -- `compile_expression_cpu()` and
`acero_query_executor.cpp`'s `translate()` throw `ExecutionError`/
`PlanningError` naming the specific unsupported construct rather than
silently miscompiling. Arrow Compute's function registry is actually more
complete than `cudf::ast` turned out to be for some of these (no "cannot
output STRING" restriction, for instance), so extending this list later is
likely less work than the GPU equivalents were -- just not free.

**No index-based column remapping.** Acero resolves every `FieldRef` by
name against real Arrow schemas at each stage of its `Declaration` tree, so
this backend needs none of `physical_planner.cpp`'s
`remap_columns()`/`find_scan_schema()` machinery the GPU path requires
(see above) -- `ColumnExpression::name()` is sufficient and correct
throughout `acero_query_executor.cpp`'s translator.

**`arrow::compute::Initialize()` must be called before running any
query.** Arrow Compute's kernel functions (`sum`, `hash_sum`,
`greater_equal`, `sort_indices`, ...) do not self-register at static-init
time when statically linked -- an omitted call leaves the global
`FunctionRegistry` empty and every query fails with "No function
registered with name: ...", regardless of which libraries are linked in.
`execute_physical_plan_cpu()` calls it once via `std::call_once`
(`ensure_compute_initialized()`) before building any `Declaration`.

**`COUNT(*)` needs its own Arrow Compute functions.** `count`/`hash_count`
require a real value-column argument (arity 1/2) and reject `COUNT(*)`'s
empty target outright; Acero has dedicated arity-0/1 `count_all`/
`hash_count_all` functions specifically for counting rows with no column
input, which `translate_aggregate()` uses instead for
`AggregateFunction::CountStar` (`Count`, i.e. `COUNT(column)`, still uses
`count`/`hash_count` with `CountOptions::ONLY_VALID`, matching the GPU
path's null-exclusion semantics).

**Scan pruning is reused, not rediscovered.** `read_scan_table()` reads
each `ParquetScanNode` fragment via `parquet::arrow::FileReader::
GetRecordBatchReader(selected_row_groups, column_indices)`, honoring the
exact row-group and column pruning decisions the physical planner already
computed -- deliberately not routed through `arrow::dataset`, which would
have no way to know about pruning this project's own physical planner
already did. Unlike the GPU path's bounded-memory, pass-based streaming
(`ParquetScanOperator`), this reads each fragment's selected row groups
fully into memory: a real, documented MVP simplification, not a bounded
streaming scan.

**`QueryResult::cpu_execution_seconds`, not `gpu_execution_seconds`, times
this backend.** The two fields are kept separate (both `std::optional`,
mutually exclusive in practice) rather than overloading one name, since the
two backends measure genuinely different work (one `DeclarationToTable()`
call vs. an operator pull-loop plus a separate device-to-host transfer).
`parquet_decoding_seconds`/`device_to_host_seconds`/
`peak_gpu_memory_bytes` stay `nullopt` for this backend: Acero's
`Declaration` tree runs as one opaque call with no per-node instrumentation
hook wired up (unlike the GPU path's `MetricsRegistry`-wrapped operator
tree), and there is no GPU memory to report.

Cross-backend correctness is verified directly: `tests/gpu/
query_engine_execute_test.cpp` runs the same SQL against the same Parquet
fixture through both `execute()` (GPU) and a `backend: "cpu"`-configured
`QueryEngine::execute()` (CPU) and asserts matching values -- comparing
column *values* only, not full `RecordBatch`/schema equality, since cudf's
aggregate kernels happen not to allocate a null-mask buffer when a
particular result contains no nulls (making the GPU path's Arrow schema
incidentally report a field as "not null"), while Arrow Acero's hash
aggregate kernels always allocate one; neither backend actually promises a
specific nullability for aggregate outputs, so requiring schema equality
would fail on an implementation detail unrelated to correctness.

### LIKE/IN/CASE/CAST implementation notes

- **`cudf::ast::compute_column` cannot produce a STRING (or other
  non-fixed-width) output column at all, even from a pure literal
  expression.** A string literal is only valid as an *intermediate* AST
  node (e.g. one side of a comparison); it can never be the compiled tree's
  root/output. This surfaces most visibly with `CASE` branches that are
  string literals (e.g. `CASE WHEN ... THEN 'high' ELSE 'low' END`):
  routing that through the AST compiler aborts with cudf's "Invalid,
  non-fixed-width type" error. `ProjectionOperator` and
  `HashAggregateOperator` both detect a plain `LiteralExpression` and
  materialize it directly via `cudf::make_column_from_scalar`
  (`cudf_adapter.hpp`'s `literal_to_scalar()`), bypassing the AST compiler
  entirely, mirroring the existing plain-column-reference fast path
  described above.
- **`CASE` is implemented by folding branches from last to first** with
  `cudf::copy_if_else(branch_result, accumulated_result, condition, ...)`,
  where `accumulated_result` starts as the `ELSE` value (or an all-NULL
  column of the result type, if there is no `ELSE`).
- **`LIKE`/`NOT LIKE` cannot go through `cudf::ast` at all** --
  `ast_operator` has no LIKE-equivalent. `FilterOperator` splits its
  predicate into top-level AND-conjuncts, evaluates any `LIKE`/`NOT LIKE`
  conjuncts separately via `cudf::strings::like()` (negated with
  `cudf::unary_operation(..., unary_operator::NOT, ...)`), evaluates the
  remaining AST-expressible conjuncts as one compiled expression, and
  combines the two boolean masks with `cudf::binary_operation(...,
  binary_operator::LOGICAL_AND, ...)`.
- **`CAST(DOUBLE AS BIGINT)` truncates, matching `cudf::ast::CAST_TO_INT64`'s
  direct `static_cast<int64_t>`.** This is a real, deliberate semantic
  difference from DuckDB, which rounds to the nearest integer on the same
  cast (confirmed by cross-validating identical queries against both
  engines: e.g. `996.604` casts to `996` here vs. `997` in DuckDB). There is
  no rounding step before the cast; this is documented rather than "fixed"
  because truncation is what `cudf::ast`'s cast operator natively does, and
  changing it would mean adding a rounding pass that most callers casting a
  DOUBLE to an integer type don't actually want.
- `IN (a, b, c)` is desugared entirely at bind time into `(x = a) OR (x = b)
  OR (x = c)` (`NOT IN` into the equivalent `AND` chain of `<>`), so it
  needs no new GPU-execution-layer support at all -- it reuses the existing
  `OR`/comparison AST compilation path.

### DECIMAL support

- **cudf's `fixed_point` (DECIMAL32/64/128) types work natively through
  `cudf::ast`**, including comparisons, arithmetic (`ADD`/`SUB`/`MUL`/`DIV`),
  and casting *from* DECIMAL to `INT64`/`UINT64`/`FLOAT64` -- verified
  empirically before committing to this design, since it wasn't obvious
  going in. The one real gap is casting *to* DECIMAL: `cudf::ast::ast_operator`
  has no `CAST_TO_DECIMAL*` (only `CAST_TO_INT64`/`UINT64`/`FLOAT64`), so
  `CAST(expr AS DECIMAL(p, s))` is scoped to the `SELECT` list / `GROUP BY`
  key position (same scope as `CASE`) and materialized directly via
  `cudf::cast()`, bypassing `cudf::ast` entirely -- see
  `ProjectionOperator`/`HashAggregateOperator`'s `decimal_cast` fast path.
- **Width selection**: `precision <= 9` -> DECIMAL32, `<= 18` -> DECIMAL64,
  `<= 38` -> DECIMAL128 (the same tiers Spark/Postgres-family engines use).
  `CAST(... AS DECIMAL)` with no explicit `(p, s)` is rejected (hsql parses
  it as `precision=scale=0`) -- matching how `CAST(... AS VARCHAR)` requires
  an explicit length elsewhere in this grammar.
- **Scale sign convention**: cudf's fixed_point scale is the exponent
  applied to the stored raw integer (`value = raw * 10^scale`), the
  *negative* of the "digits after the decimal point" convention
  `DataType::scale`/Arrow/Parquet use. `to_cudf_type()` negates it; get this
  backwards and comparisons silently evaluate against the wrong magnitude
  rather than failing loudly, so it's covered by a real Parquet-file
  round-trip test (`tests/gpu/decimal_test.cpp`), not just a unit test.
- **Implicit promotion coerces a numeric literal to a DECIMAL column's exact
  type; it does not widen or auto-cast a column or computed expression.**
  `WHERE price > 19.99` retypes the literal `19.99` in place (a compile-time
  constant can be exactly retargeted with no precision loss); `WHERE price >
  some_int_column` is rejected at bind time (`cast_if_needed` in
  binder.cpp), since that would need a genuine runtime CAST to DECIMAL and
  cudf::ast has none. Two DECIMALs with different precision/scale are
  likewise rejected rather than auto-widened -- both are real, honest scope
  limits, not oversights; wrap the non-literal side in an explicit
  `CAST(... AS DECIMAL(p, s))` instead.
- **`SUM`/`MIN`/`MAX` over DECIMAL keep the argument's own precision/scale
  unchanged** (like `MIN`/`MAX` over any type) rather than widening the
  declared result type -- cudf's own `SUM` aggregation evaluates DECIMAL
  columns natively and preserves scale. This is a deliberate, documented
  difference from DuckDB, which widens `SUM(DECIMAL(p, s))`'s result to
  `DECIMAL(38, s)` to rule out overflow across many rows; KernelLake does
  not, so summing enough full-precision (`p` already near 38) DECIMAL
  values could in principle overflow where DuckDB wouldn't. Not a concern
  at realistic precisions, but worth knowing.
- **`AVG` over DECIMAL is not supported** (rejected at bind time) --
  `CAST` the argument to `DOUBLE` first.
- A DECIMAL literal's value only ever has `double` precision to begin with
  (hsql parses `19.99` into a C++ `double`; see `LiteralStorage`), so it
  cannot represent more significant digits than a `double` can -- the same
  class of documented precision caveat as the CAST-truncates-vs-DuckDB-
  rounds difference above, not a bug.
- **Arrow interop**: this Arrow version (25.x) has distinct
  `Decimal32Type`/`Decimal64Type`/`Decimal128Type`/`Decimal256Type` (not
  just `Decimal128Type`), and `cudf::to_arrow_host`/`to_arrow_schema` map a
  cudf DECIMAL32/64/128 column to the matching Arrow width rather than
  always widening to Decimal128 -- a result column's actual Arrow type
  depends on its precision. Code reading a KernelLake DECIMAL result via
  Arrow should use `arrow::Array::GetScalar(...)->ToString()` (as
  `result_formatter.cpp` already does) rather than assuming a specific
  `arrow::Decimal128Array`/etc. subtype.

### Hash joins

- **Scope**: exactly two `read_parquet(...)` sources, both explicitly
  aliased, joined with `INNER JOIN ... ON <a.col = b.col>` -- a single
  equality between one plain column from each side, of identical type.
  Comma-style joins (`FROM a, b WHERE a.k = b.k`), `LEFT`/`RIGHT`/`FULL`/
  `CROSS` JOIN, multi-key/non-equality conditions, and 3+-table joins all
  fail clearly at parse or bind time rather than being silently
  reinterpreted. The right-hand (second) table is always the *build* side
  (materialized in full before probing begins); put the smaller table
  there for performance -- there is no cost-based optimizer to choose this
  automatically.
- **Combined-index design**: the binder resolves every column reference in
  a JOIN query (qualified or not) to a `ColumnExpression` whose index is
  into the *combined* `[left_schema fields..., right_schema fields...]`
  row -- exactly what `HashJoinOperator` actually produces (left columns
  gathered first, then right). This is what lets almost the entire rest of
  the pipeline (the GPU expression compiler, the optimizer's column
  collection, every operator except the join itself) treat a joined query
  no differently from a single-table one above the join; only the physical
  planner's `LogicalJoin` -> `HashJoinNode` conversion and the join
  operator itself need to know two tables are involved at all. An
  unqualified reference that exists on both sides is rejected as ambiguous
  at bind time, same as SQL generally requires.
- **Implicit promotion does not extend across a JOIN condition**: the two
  key columns must already be the same type. Mixing e.g. `INT32` and
  `INT64` join keys produces an implicit `CastExpression` around one side
  during binding (the same promotion `WHERE` comparisons get), which is no
  longer a bare `ColumnExpression` -- `extract_equi_join_keys()` in
  binder.cpp rejects this with a clear `BindingError` rather than trying to
  compile a cast into the join key extraction. Make both sides the same
  type instead.
- **No predicate pushdown across a join**: `LogicalScan::pushable_predicates()`
  stays empty for both sides of a JOIN query (the optimizer's
  `annotate_scan()` splits *required columns* by side using each
  `ColumnExpression`'s combined index, but always discards
  `pushable_predicates` collected above a `LogicalJoin` rather than routing
  them to one side -- `PushablePredicate`'s bare column name has no way to
  say which side it came from once two schemas are in play). Row-group
  pruning still runs per-side in the physical planner; a JOIN query's WHERE
  clause (beyond the ON condition) just never narrows it. Column pruning
  (only reading columns actually referenced, on each side, including the
  join key) still works normally.
- **Known limitation**: remapping a column reference that sits *directly
  above* the join (in a `Filter`/`Projection`/`Aggregate`/`Sort` whose
  child is the join itself) matches by column *name* against the join's
  combined output schema, the same way remapping above a plain scan always
  has. If both JOIN sides happen to have a same-named column, an
  unqualified reference to it *after* the join could resolve to the wrong
  side -- not a concern for the join condition itself (bound with each
  side's own index) or for any qualified reference (which the binder
  resolves to a definite side up front), only for a hypothetical unqualified
  post-join reference, which the binder's own ambiguity check already
  rejects before this could matter in practice. `SELECT *` on a JOIN with
  colliding column names is likewise rejected (the ordinary "duplicate
  output column name" check).
- **cudf::hash_join mechanics**: unlike every streaming operator elsewhere
  in this codebase, `HashJoinOperator`'s *build* side must be fully
  materialized before any output can be produced -- `cudf::hash_join`
  builds its hash table once, up front, from a single `cudf::table_view`
  (the same "consume child to exhaustion, concatenate into one table"
  shape `SortOperator` uses, for an analogous reason). The *probe* side
  streams through normally, one `inner_join()` call per incoming batch,
  gathering matching rows from both the probe batch and the persistent
  build table. A build side with zero rows short-circuits without ever
  constructing a `cudf::hash_join` (an INNER JOIN against no rows can never
  match anything); this is a real gap for LEFT JOIN outer rows should that
  ever be added, but is correct for INNER JOIN's current scope.

### GPU dependency vendoring (no conda)

The `gpu-dev` preset (`KERNELLAKE_WITH_CUDA=ON`) needs libcudf and RMM.
Rather than requiring conda/mamba, `cmake/ThirdPartyRapids.cmake` vendors
RAPIDS's self-contained "lib*" wheels from PyPI via `FetchContent` --
pinned by URL + SHA-256, the same pattern `ThirdPartySqlParser.cmake` uses
for `hyrise/sql-parser`. Each wheel is a plain zip containing C++ headers,
a compiled shared library, and a full CMake package config
(`lib64/cmake/<name>/<name>-config.cmake`), with no Python runtime
dependency (tagged `py3-none`). This has been verified end-to-end: a
standalone smoke-test CMake project linking `cudf::cudf` successfully
allocated and inspected a real GPU-resident `cudf::column`.

Packages vendored this way: `rapids-logger`, `librmm-cu12`,
`libkvikio-cu12`, `libcudf-cu12`, and `nvidia-libnvcomp-cu12` (the last
supplies `libnvcomp.so.5`, an undeclared transitive dependency of
`libcudf.so` not expressed in its own CMake config -- `nvidia-nvcomp-cu12`,
the similarly-named package, is a trap: it only ships Python extension
modules, not the plain `.so`). libcudf's own `cudf-config.cmake` requires
CMake >= 3.30.4, newer than Ubuntu 24.04's apt package (3.28.3).

### Ubuntu 26.04 baseline (`docker/Dockerfile`)

`docker/Dockerfile`'s published images (`dev`/`runtime`) build on plain
`ubuntu:26.04`, not Ubuntu 24.04 or NVIDIA's official `nvidia/cuda` devel/
runtime images. This was originally a deliberate, empirically-verified
departure from this project's own non-container development environment,
which at the time stayed on Ubuntu 24.04 -- the two were independent, with
no requirement that they match. That has since changed on its own: a later
development session's own sandbox moved to Ubuntu 26.04 too (`lsb_release`:
`resolute`), which is what let `kernellake-server`'s `server-dev` preset
(see "Arrow Flight SQL server" below) be built and tested directly there,
no Docker required. The independence point still holds either way -- there
remains no requirement that the two match, they just currently happen to.

**Why**: two real, distro-level blockers, both confirmed by an actual
`docker build`/`docker run --gpus all` in this project's own development
session (not inferred from package metadata alone):

- **Arrow Flight SQL doesn't link on Ubuntu 24.04.** Arrow 25.0.0's apt
  package for `libarrow-flight-sql-dev` fails at final link with undefined
  references like `absl::lts_20250127::synchronization_internal::
  KernelTimeout::MakeAbsTimespec()`. Root cause, confirmed via `nm` on the
  actual `.deb` contents: Ubuntu 24.04's own system Abseil (a 2022-06-23
  snapshot) is too old for Arrow 25's minimum requirement, so Arrow's
  24.04 package statically bundles a private, newer Abseil copy inside
  `libarrow_bundled_dependencies.a` -- and that bundle is incomplete (the
  translation unit defining `KernelTimeout::MakeAbsTimespec()` wasn't
  included at Apache's own build time). On Ubuntu 26.04, `libarrow2500`
  instead *dynamically depends on* a real system package,
  `libabsl20260107`, because 26.04's system Abseil is new enough that
  Arrow doesn't need to bundle its own -- the entire broken code path
  doesn't exist. Verified for real: an empty `FlightSqlServerBase`
  subclass builds, links, and calls `Init()` successfully in an
  `ubuntu:26.04` container (needs `$<LINK_GROUP:RESCAN,...>` around
  `ArrowFlightSql::arrow_flight_sql_static`/`ArrowFlight::arrow_flight_static`/
  `Arrow::arrow_static`/`gRPC::grpc++` either way, since Arrow's own CMake
  target doesn't declare `gRPC::grpc++` as a dependency).
- **otel-cpp has no Ubuntu 24.04 apt package at all** (`FetchContent`
  vendoring was the only option there). Ubuntu 26.04 ships
  `opentelemetry-cpp-dev` (1.23.0) directly, with both OTLP/HTTP and
  OTLP/gRPC exporter CMake targets pre-built -- both verified for real: a
  minimal program creating and ending a span links and runs against each
  exporter (the connection-refused errors in that test are expected --
  no collector was listening -- not build failures).

**CUDA: apt's `nvidia-cuda-toolkit`, not NVIDIA's `nvidia/cuda` image.**
This was the original motivation for considering Ubuntu 26.04 at all
(installing CUDA as a normal apt package rather than pinning a specific
`nvidia/cuda:<version>-<devel|runtime>-ubuntu<version>` Docker tag).
Two apt-level facts made this concrete rather than aspirational:

- `nvidia-cuda-toolkit` on Ubuntu 26.04 is version **12.4.1** -- a *minor*
  step down from the previously-pinned `nvidia/cuda:12.6.3`, not a major
  version change. `cmake/ThirdPartyRapids.cmake`'s pinned RAPIDS wheels
  stay exactly as they are (`-cu12`, unchanged) -- no re-vendoring needed.
  (NVIDIA's own `nvidia/cuda` image, by contrast, only publishes
  `ubuntu26.04` tags starting at CUDA 13.3 -- taking that path instead
  would have forced a major CUDA version bump and a `-cu13` RAPIDS
  re-vendor, a materially bigger and differently-risky change that was
  considered and rejected in favor of the apt-toolkit path.)
- `nvidia-cuda-toolkit` does **not** pull in `libcufile-dev` (GPUDirect
  Storage), which `kvikio` (a libcudf dependency) requires --
  `kvikio-config.cmake` fails configure outright ("Compiled with cuFile
  support but cuFile not found") without it. `libcufile-dev` is a separate
  apt package, but from the same repo at the matching `12.4.1-8` build
  revision -- not a NVIDIA-repo dependency, still fully apt-native.

nvcc lands at `/usr/bin/nvcc` under this packaging (Debian convention),
not `/usr/local/cuda/bin/nvcc` (NVIDIA installer/Docker-image convention).
`CMakePresets.json`'s `gpu-dev` preset's `CMAKE_CUDA_COMPILER` default
still points at the NVIDIA-installer path -- a stale assumption from when
this project's own non-container environment used that convention; now
that this project's own sandbox is on Ubuntu 26.04 too (see "Ubuntu 26.04
baseline" above) with apt's `nvidia-cuda-toolkit`, invoking the `gpu-dev`
preset here also needs an explicit `-DCMAKE_CUDA_COMPILER=/usr/bin/nvcc`
override, same as `docker/Dockerfile` already does in its own `cmake
--preset gpu-dev` invocation. Updating the preset's own default to match is
a reasonable follow-up, not done here since both current environments work
fine with the explicit override and neither depends on the preset default
being correct.

`CMAKE_CUDA_ARCHITECTURES` cannot be left at the top-level `CMakeLists.txt`
default of `native` (which probes an actual device) inside
`docker/Dockerfile`, since no GPU is visible during `docker build` (unlike
`docker run --gpus all`). It's pinned explicitly instead:
`"70-real;75-real;80-real;86-real;89"` -- real compiled code for
Volta through Ada, with Hopper's entry left without the `-real` suffix so
its PTX is embedded too, letting the driver JIT-compile for newer
architectures with no native code in the binary at all. This is exactly
what let an RTX 5060 Ti (Blackwell/sm_120 -- newer than anything CUDA
12.4's nvcc can target directly) run the full test suite successfully
against this image.

**Verified for real, not just configured**: `docker build --target dev`
completes (RAPIDS/libcudf/RMM/kvikio FetchContent vendoring, 103/103
targets built); `docker build --target runtime` produces a 2.17 GB image
(down from `dev`'s 14.1 GB) containing 22 shared libraries and the
compiled binary only -- no compiler, no `nvcc`, no CUDA headers, no static
libraries (`which nvcc gcc g++ cmake` all resolve to nothing inside it);
`docker run --gpus all` against a real GPU (RTX 5060 Ti) ran **all 214
tests successfully** in the `dev` image, and a real `GROUP BY` query
against real generated Parquet data through the `runtime` image alone
produced correct, GPU-executed results (`gpu_execution_seconds` and
`peak_gpu_memory_bytes` populated, matching values from an equivalent
local run).

**`runtime` stage's shared-library closure.** `runtime-libs`'s `ldd`-based
closure (used to avoid hard-coding vendored library names/paths that would
go stale on a version bump) excludes only a short, fixed list of
glibc/libgcc/libstdc++ basenames -- those are guaranteed present, at an
identical version, on the `runtime` stage's `ubuntu:26.04` base, since
`dev` builds `FROM` that exact same base. This differs from the
NVIDIA-image-based design it replaced, which additionally excluded
everything under `/usr/lib/*` on the assumption that the CUDA *runtime*
Docker image (`nvidia/cuda:*-runtime-*`) pre-supplied those paths --  an
assumption that no longer applies now that `runtime` is plain Ubuntu with
no CUDA preinstalled, so CUDA's own shared libraries (`libcudart.so.12`,
etc.) are now correctly included in the copied closure instead of silently
assumed-present.

### Arrow Flight SQL server (`kernellake-server`)

Phase 1 of the Flight SQL/otel-cpp/Helm-chart epic (see `docs/ROADMAP.md`).
Built behind `KERNELLAKE_BUILD_SERVER` (default `OFF`; a `server-dev` CMake
preset turns it on), so it adds no new required dependency for anyone not
using it. `KernelLakeFlightSqlServer`
(`include/kernellake/server/flight_sql_server.hpp`,
`src/server/flight_sql_server.cpp`) subclasses
`arrow::flight::sql::FlightSqlServerBase` and implements just
`GetFlightInfoStatement`/`DoGetStatement` -- Arrow 25.0.0 declares the rest
of that base class's RPCs virtual with default `NotImplemented` bodies, not
pure virtual, so a minimal override compiles and serves real queries
without touching prepared statements, catalog/schema/table listing, or
`SqlInfo`. The two implemented RPCs deliberately execute the query
*eagerly* inside `GetFlightInfoStatement` (the first RPC a client makes)
and buffer the `QueryResult` in an in-process handle-keyed registry that
`DoGetStatement` (the second RPC) streams from and then erases -- avoiding
the need to keep a live cursor open across two separate gRPC calls that may
not even land on the same connection, at the cost of buffering the whole
result in host memory between the two calls. Every `KernelLakeError`
subclass thrown by `QueryEngine` is translated to a matching generic
`arrow::Status` code (`Invalid` for `SqlError`/`BindingError`/
`PlanningError`/`OptimizationError`, `IOError` for `StorageError`,
`ExecutionError`/`OutOfMemory` for the GPU-side errors) before it can cross
the gRPC boundary as a raw C++ exception -- verified with a real ADBC
Python client seeing a clean `INVALID_ARGUMENT` for bad SQL rather than a
dropped connection.

Respects `engine.backend: gpu|cpu` (new `ServerSection` in `EngineConfig`
adds `server.host`/`server.port`) exactly like the CLI's `query --backend`
flag -- not hardcoded to GPU. For `backend: gpu`, a long-lived server can't
use `QueryEngine::execute(sql)`'s one-shot convenience overload (it builds
and tears down its own `RmmEnvironment` per call -- a real
use-after-free race under concurrent gRPC handler threads, per the
Concurrency notes above); instead `GpuExecutionCoordinator`
(`include/kernellake/server/gpu_execution_coordinator.hpp`) owns one
`RmmEnvironment` for the server's whole lifetime and serializes GPU
`execute()` calls behind a single mutex (single-flight -- sufficient given
this project's existing "one query executes on the GPU at a time" MVP
simplification; a real request queue is a reasonable future refinement).
Split into `gpu_execution_coordinator_{gpu,stub}.cpp`, selected by
`KERNELLAKE_WITH_CUDA` exactly like `query_engine_execute_{gpu,stub}.cpp`,
so `server-dev` (CPU-only) needs no CUDA/RMM at all; requesting `backend:
gpu` against such a build fails fast at server startup with a
`ConfigurationError`, not silently at first query.

**A real CMake linking gotcha, worth documenting precisely** since the
earlier note above (end of "Ubuntu 26.04 baseline") turned out to be an
incomplete fix once tried against this project's actual (non-minimal)
dependency tree: wrapping `ArrowFlightSql::arrow_flight_sql_static`/
`ArrowFlight::arrow_flight_static`/`Arrow::arrow_static`/`gRPC::grpc++` in
CMake's `$<LINK_GROUP:RESCAN,...>` genex does **not** actually fix the
undefined-reference failures in a project this size, for two independent
reasons, both confirmed by inspecting the generated `ninja` link command
directly rather than guessing from documentation:

1. **`LINK_GROUP` only wraps the *explicitly listed* members in
   `--start-group`/`--end-group`** -- it does not pull each member's own
   transitively-required libraries inside the group boundary too.
   `gRPC::grpc++`'s own dependency closure (`libgrpc.so`, every `libabsl_*`
   library, `libgpr.so`, `libcares.so`, ...) still landed *after*
   `--end-group` in the generated link line, textually separated from the
   archives that actually reference symbols in them.
2. **Including `Arrow::arrow_static` in the group is what the minimal
   verified reproduction needed, but it creates a genuine CMake-level
   dependency cycle in this project's real tree**: `Arrow::arrow_static` is
   linked by nearly every other `kernellake_*` module too, so CMake's own
   `LINK_GROUP` cycle detector sees "the group depends on
   `kernellake_api`'s other libraries, which depend on `Arrow::arrow_static`,
   which is itself inside the group" and refuses to generate the build at
   configure time.

The actual fix, in `src/server/CMakeLists.txt` and (for the test binary
that also links `kernellake_flight_sql_server`) `tests/unit/CMakeLists.txt`:
raw `-Wl,--start-group ... -Wl,--end-group` strings passed directly to
`target_link_libraries`, not the `LINK_GROUP` genex. Being plain strings
rather than CMake target names, they aren't analyzed by CMake's
dependency-cycle detection at all, sidestepping problem (2); and while they
turned out *not* to solve problem (1) via any special interleaving
behavior either (transitively-required libraries still land after the
literal `--end-group` marker, confirmed by inspecting the link line again
after switching), the fix works anyway once applied consistently to
*every* target that links `kernellake_flight_sql_server` directly (the
`kernellake-server` executable itself, and separately the
`kernellake_unit_tests` binary) -- the actual root cause was simply that
`libarrow_flight_sql.a`/`libarrow_flight.a` and `libgrpc++.so` had never
been placed adjacent to each other on the link line at all before this
group existed anywhere in the command, not a subtler ordering problem
requiring the full transitive closure to be inside the bracket.

Verified for real: `server-dev` (147/147 tests, including a real
`arrow::flight::sql::FlightSqlClient` round-trip test and an invalid-SQL
error-path test) and a manual smoke test against a running server using an
independent Python `adbc_driver_flightsql` client -- a real grouped
aggregate query against `generate-data`-produced Parquet returned correct
rows. `dev` (145/145) and `gpu-dev` (214/214) both reconfirmed unaffected.

## Future architecture (interfaces only, not yet implemented)

These are named as forward-declared types or documented models so later
work has a clean seam to build against; none of them have implementations
yet, and none are exposed as supported CLI features.

**Future physical operators**: `Exchange`, `Spill`, `Repartition`,
`MergeAggregate`, `Broadcast`.

**Future distributed model**:
```
Coordinator -> parse and optimize query
            -> create physical fragments
            -> assign file and row-group partitions
Workers     -> execute local GPU pipelines
            -> produce partial aggregates
            -> exchange/merge partial results
            -> final result returned to client
```
Large worker-to-worker data is never planned to route through the
coordinator.

**Future catalog model**: `Catalog`, `Table`, `TableSnapshot`, `DataFile`,
`PartitionSpec`, with eventual support for Apache Iceberg REST catalogs,
Apache Polaris, Nessie, and Hadoop-compatible catalogs. KernelLake will not
implement a proprietary table format.

**Future storage backend**: an `S3ObjectStore` implementing the same
`ObjectStore` interface `LocalObjectStore` does today, plus documentation
for MinIO/Ceph compatibility.
