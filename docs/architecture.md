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
`FROM` clause only accepts table names, joins, and subqueries -- it has no
table-valued-function syntax. `kernellake::sql::parse_sql()` therefore
recognizes the specific pattern `FROM read_parquet('path' [, 'path2', ...])`
with a regex, extracts the path arguments, substitutes a plain placeholder
identifier, and hands the rewritten text to the real parser. Any `FROM`
clause that doesn't match this pattern (a bare table name, a join, a
subquery) fails with a clear `SqlError` rather than being silently
reinterpreted. This is a narrow, one-time syntax adapter, not general
SQL-string rewriting -- optimizer rules always operate on the structured
plan/expression trees, never on SQL text.

## Supported SQL grammar (current)

```
SELECT <items> FROM read_parquet('path' [, 'path2', ...])
  [WHERE <expr>] [GROUP BY <cols>] [ORDER BY <cols>] [LIMIT <n>]
```

- Column references, aliases, `*`
- Numeric, string, boolean, date (`DATE 'YYYY-MM-DD'`), and `NULL` literals
- Arithmetic (`+ - * /`), comparisons, `AND`/`OR`/`NOT`, `BETWEEN`,
  `IS [NOT] NULL`
- Aggregates: `SUM`, `COUNT`, `COUNT(*)`, `MIN`, `MAX`, `AVG`

Not yet supported (fails clearly rather than being silently reinterpreted):
`DISTINCT`, `HAVING`, set operations (`UNION`/etc.), `WITH`/CTEs, joins,
subqueries, `OFFSET`, `LIKE`/`IN`/`CASE`/`CAST` expressions, window
functions, and any function other than the five aggregates above.

`ORDER BY` after `GROUP BY` is parsed and bound but rejected at physical
planning time (`PlanningError`): sorting the grouped output would require
binding `ORDER BY` against the post-aggregation schema (including SELECT
aliases), which is not yet implemented. `ORDER BY` on a non-aggregate query
works normally.

## Logical plan and the optimizer

Logical plan nodes: `LogicalScan`, `LogicalFilter`, `LogicalProjection`,
`LogicalAggregate`, `LogicalSort`, `LogicalLimit`. `LogicalScan` always
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
a `GROUP BY`, or `ScalarAggregate` when it does not), always wrapping the
result in `ArrowResultNode`. It throws `PlanningError` for a `LogicalSort`
node, since no physical sort operator exists yet.

## CPU/GPU build split

`KERNELLAKE_WITH_CUDA` (off by default) gates everything that needs CUDA,
libcudf, and RMM: `DeviceBatch` (the GPU-resident batch wrapping
`cudf::table`), `ExecutionContext`, the `PhysicalOperator` streaming
interface, and the concrete GPU operators (`kernellake_execution`,
`kernellake_memory`; see `tests/gpu/`). `QueryEngine::execute()` is built
from one of two mutually exclusive translation units selected by a CMake
generator expression on `KERNELLAKE_WITH_CUDA`
(`src/api/query_engine_execute_gpu.cpp` vs. `_stub.cpp`): the GPU build runs
the real operator pipeline (`kernellake/execution/operator_builder.hpp`
turns a `PhysicalPlanPtr` into a `PhysicalOperator` tree, pulled to
exhaustion inside `RmmEnvironment::track_query()` for per-query memory
accounting, with each resulting `DeviceBatch` converted to an Arrow
`RecordBatch` via `kernellake/execution/arrow_bridge.hpp`); the CPU-only
`dev` preset's stub throws a clear `ExecutionError` instead -- KernelLake
never substitutes a CPU implementation for GPU execution without saying so
explicitly. The `kernellake query` CLI command (`src/cli/query_command.cpp`)
is unconditionally built and calls `QueryEngine::execute()`, so its
behavior likewise depends on which preset built it.

Everything else in this document (parsing through physical planning and
pruning), plus `kernellake generate-data`, is CPU-only and builds/tests
with the `dev` CMake preset alone.

### GPU operators

| Operator | Notes |
| --- | --- |
| `ParquetScanOperator` | `cudf::io::chunked_parquet_reader`, bounded by `pass_read_limit_bytes` (not row count) |
| `FilterOperator` | `cudf::compute_column` + `cudf::apply_boolean_mask` over a compiled AST predicate |
| `ProjectionOperator` | Compiled AST per computed item; a bare column reference is copied directly instead (see below) |
| `ScalarAggregateOperator` | No `GROUP BY`: `cudf::reduce` with its `init` parameter folds each batch into a running scalar (SUM/MIN/MAX/AVG numerator); COUNT/AVG's denominator is a host-side counter. Empty input produces NULL, not zero, except `COUNT(*)`/`COUNT(x)` which produce 0. |
| `HashAggregateOperator` | `GROUP BY`: `cudf::groupby::streaming_groupby` accumulates partial groups across batches within `max_distinct_keys` (default 10M), `finalize()`d on last `next()` |
| `LimitOperator` | `cudf::slice` + the `cudf::table` copy constructor to truncate the final batch |
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

## Future architecture (interfaces only, not yet implemented)

These are named as forward-declared types or documented models so later
work has a clean seam to build against; none of them have implementations
yet, and none are exposed as supported CLI features.

**Future physical operators**: `HashJoin`, `Sort`, `Exchange`, `Spill`,
`Repartition`, `MergeAggregate`, `Broadcast`.

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
