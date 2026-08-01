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
  -> GPU execution           (kernellake::PhysicalOperator / DeviceBatch -- pending libcudf/RMM)
  -> Arrow result
```

Every stage above physical planning is implemented and covered by unit
tests (see `tests/unit/`). GPU execution is not: see "CPU/GPU build split"
below.

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
interface, and the GPU filter/projection/aggregation operator
implementations. `QueryEngine::execute()` and the `kernellake query` CLI
command throw a clear `ExecutionError` until that layer exists --
KernelLake never substitutes a CPU implementation for GPU execution without
saying so explicitly.

Everything else in this document (parsing through physical planning and
pruning) is CPU-only and builds/tests with the `dev` CMake preset.

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
