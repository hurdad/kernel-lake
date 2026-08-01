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

Not yet supported (fails clearly rather than being silently reinterpreted):
`DISTINCT`, `HAVING`, set operations (`UNION`/etc.), `WITH`/CTEs, joins,
subqueries, `OFFSET`, window functions, `LIKE` outside a top-level `WHERE`
AND-conjunct, `CASE` in `WHERE`/aggregate arguments, and any function other
than the five aggregates above.

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
a `GROUP BY`, or `ScalarAggregate` when it does not; `LogicalSort` becomes
`SortNode`, see `SortOperator` below), always wrapping the result in
`ArrowResultNode`.

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
| `SortOperator` | `ORDER BY`: **blocking**, unlike every operator above -- consumes `child` to exhaustion, concatenates every batch (`cudf::concatenate`) into one table, then `cudf::stable_sorted_order` + `cudf::gather`. Memory footprint is the whole result set, not bounded like the streaming operators. |
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

**Future physical operators**: `HashJoin`, `Exchange`, `Spill`,
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
