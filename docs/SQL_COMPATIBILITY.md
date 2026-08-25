# SQL compatibility

A single, scannable reference for what SQL KernelLake accepts today. This
is a compatibility matrix, not an implementation guide -- for *why* a
given scope boundary exists, see `docs/ARCHITECTURE.md`'s "Supported SQL
grammar (current)" section and its sub-sections (`HAVING`/scalar
subqueries, `IN (SELECT ...)` subqueries, Hash joins, DECIMAL support,
LIKE/IN/CASE/CAST implementation notes, Derived tables). For which TPC-H
queries this adds up to, see `docs/TPCH.md` and `docs/ROADMAP.md`'s
"Done"/"Not yet started" sections. Everything below reflects the
codebase as of the TPC-H Q15 addition (16 of 22 TPC-H queries supported).

KernelLake vendors `hyrise/sql-parser` (hsql) for grammar parsing, then
applies its own, much narrower binder/logical-planner scope on top --
hsql itself accepts a large fraction of standard SQL, but KernelLake
rejects anything outside the scope below with a clear parse- or bind-time
error rather than silently reinterpreting it.

## Query shape

```
SELECT <items> FROM read_parquet('path' [, 'path2', ...])
  [WHERE <expr>] [GROUP BY <cols>] [HAVING <bool expr>]
  [ORDER BY <cols>] [LIMIT <n>]

SELECT <items> FROM read_parquet(...) AS a
  [INNER | LEFT [OUTER]] JOIN read_parquet(...) AS b
    ON <a.col = b.col> [AND <predicate over b's own columns only>]
  [JOIN read_parquet(...) AS c ON <... = c.col>] ...
  [WHERE <expr>] [GROUP BY <cols>] [HAVING <bool expr>]
  [ORDER BY <cols>] [LIMIT <n>]

SELECT <items> FROM (SELECT ...) AS alias
  [WHERE <expr>] [GROUP BY <cols>] [HAVING <bool expr>]
  [ORDER BY <cols>] [LIMIT <n>]
```

`read_parquet(...)` may be replaced by `read_iceberg('catalog.namespace.table')`,
`read_delta('table_uri')`, or `read_unity_catalog('instance.catalog.schema.table')`
in either of the first two shapes, mixed freely across sources within one
join chain. A single, top-level derived table (`FROM (SELECT ...) AS
alias`, third shape) is also accepted -- but only as a query's *entire*
FROM clause: a derived table nested inside a JOIN, or a JOIN inside a
derived table's own FROM, is rejected at parse time, as is a comma-style
join list.

| Clause | Supported | Notes |
| --- | --- | --- |
| `SELECT` list | Yes | Columns, aliases, `*`, expressions, aggregates |
| `FROM` | Single source, JOIN chain, or single derived table | `read_parquet`/`read_iceberg`/`read_delta`/`read_unity_catalog`, or `(SELECT ...) AS alias` |
| `JOIN ... ON` | Yes, N-way | `INNER` or `LEFT [OUTER]`, one equality key per step plus an optional right-side-only predicate -- see "Joins" below |
| `WHERE` | Yes | See "Expressions" below; also accepts one subquery form, `IN (SELECT ...)` -- see "Subqueries" |
| `GROUP BY` | Yes | By base column, computed expression, or `SELECT`-list alias |
| `HAVING` | Yes, narrow | Aggregates/`GROUP BY` keys only, plus one scalar subquery form -- see "Subqueries" |
| `ORDER BY` | Yes | Multiple keys, `ASC`/`DESC` |
| `LIMIT` | Yes | |
| `OFFSET` | No | |
| `DISTINCT` | No | |
| `UNION`/`INTERSECT`/`EXCEPT` | No | |
| `WITH` (CTEs) | No | Rejected unconditionally at parse time |
| Window functions | No | |
| Subquery in `FROM` (derived table) | Yes, narrow | Single, top-level only -- not itself joined/joinable, not correlated -- see "Joins" below |

## Joins

- **Type**: `INNER` or `LEFT [OUTER] JOIN`. `RIGHT`/`FULL`/`CROSS` JOIN
  and comma-style joins (`FROM a, b WHERE a.k = b.k`) are rejected at
  parse time.
- **Condition**: exactly one equality (`a.col = b.col`) between a plain
  column already in scope (`a`, the left/probe side) and a plain column
  from the newly joined source (`b`, the right/build side), same type,
  required. The `ON` clause may additionally combine that key with
  `AND`-conjuncts, but only ones that reference *exclusively* `b`'s own
  columns (TPC-H Q13's own `o_comment NOT LIKE '%special%requests%'`) --
  these are pushed down as a pre-filter on `b` before the join runs,
  exact for both `INNER` and `LEFT OUTER`. A conjunct referencing `a` at
  all (alone, or mixed with `b`'s columns) is rejected: unlike the
  right-side-only case, a `LEFT OUTER JOIN`'s own left-side `ON`
  conjunct has different semantics (a left row failing it must still
  appear once, null-extended, not be dropped like a pre-filter would)
  that isn't implemented. Any other multi-key or non-equality condition
  is rejected at bind time -- a composite key must be split, with one
  key as the `JOIN ... ON` condition and the rest moved into `WHERE`
  (valid for `INNER JOIN` only; see TPC-H Q5/Q9 for real examples).
- **Chain length**: up to 12 sources (`kMaxJoinSources`), each explicitly
  aliased. Verified in production shape up to 6-way (TPC-H Q5/Q7/Q9).
- Same-named columns from different join sides resolve correctly by
  qualified name (`a.x`/`b.x`), including a table joined to itself under
  two aliases (TPC-H Q7's `nation AS n1`/`AS n2`).
- A single, top-level derived table as a query's entire `FROM` clause
  (`FROM (SELECT ...) AS alias`, TPC-H Q13's own outer-query shape) is
  supported -- the inner query is bound and planned first, and its
  output schema/logical plan become what the outer query binds/builds
  against, exactly as if it were a real source. Not itself joined or
  joinable, and not correlated (the inner query cannot reference the
  outer query's columns).

## Expressions and operators

| Feature | Supported | Backend notes |
| --- | --- | --- |
| Arithmetic (`+ - * /`) | Yes | Both |
| Comparisons (`= <> < <= > >=`) | Yes | Both |
| `AND` / `OR` / `NOT` | Yes | Both |
| `BETWEEN` | Yes | Both |
| `IS [NOT] NULL` | Yes | Both |
| `LIKE` / `NOT LIKE` | Yes | Both, everywhere (`WHERE`, `SELECT`, `CASE` branch, aggregate argument) |
| `IN (literal, ...)` / `NOT IN (...)` | Yes | Desugared at bind time into an OR/AND chain of equalities; no dedicated execution support needed |
| `IN (SELECT ...)` / `NOT IN (SELECT ...)` | Yes, narrow | Non-correlated, any row count, one column -- resolved to a literal list (same OR/AND desugar) before binding; see "Subqueries" |
| `EXISTS` / `NOT EXISTS` | Yes, narrow | Correlated, single equality key -- rewritten into a `LEFT SEMI`/`LEFT ANTI` join step before binding; see "Subqueries" |
| Correlated *scalar* subqueries | No | |
| `CASE WHEN ... THEN ... [ELSE ...] END` | Yes, backend-scoped | CPU: everywhere. GPU: `SELECT` list, `GROUP BY` keys, aggregate arguments -- **not yet in `WHERE`** |
| `CAST(expr AS type)` | Yes | `INT`/`BIGINT`/`DOUBLE`/`VARCHAR(n)`/`DECIMAL(p, s)`; numeric-to-integer casts truncate (not round -- differs from DuckDB) |
| `EXTRACT(field FROM expr)` | Yes, backend-scoped | `field` is `YEAR`/`MONTH`/`DAY` only; `expr` must be `DATE`/`TIMESTAMP`. Both backends everywhere `CASE` can appear except `WHERE` |
| Aggregate-combining `SELECT` items (e.g. `100.0 * SUM(a) / SUM(b)`) | Yes | Both -- resolved in shared logical-plan construction |

## Subqueries

Three forms are supported. The first two are **non-correlated** (bound
independently against their own `FROM`/`JOIN` schema, with no access to
the outer query's tables or aliases) and both executed always on the CPU
(Acero) backend regardless of the outer query's own `--backend`, as a
side effect of planning (including inside `explain`):

1. **A scalar subquery as an operand inside `HAVING`'s own boolean
   expression** (`HAVING SUM(x) > (SELECT ...)`, TPC-H Q11's shape).
   Must return exactly one row, one column (`DOUBLE`/`INT64`/`STRING`).
   The subquery's own `FROM` may be a single source, a JOIN chain, or a
   derived table -- including a derived table containing its own `GROUP
   BY` feeding an outer `MAX(...)`/`MIN(...)` (TPC-H Q15's shape,
   `HAVING total_revenue = (SELECT MAX(total_revenue) FROM (SELECT ...
   GROUP BY ...) AS r2)`), needed for any "aggregate of a grouped
   aggregate" the subquery itself can't express in one level. An exact
   `=`/`<>` comparison against a `HAVING`/`IN` subquery's own result is
   unreliable on the GPU backend specifically -- see "CPU vs. GPU
   backend differences" below.
2. **`value IN (SELECT ...)` / `NOT IN (SELECT ...)` in `WHERE`**
   (`o_orderkey IN (SELECT l_orderkey FROM ... HAVING SUM(...) > 300)`,
   TPC-H Q18's shape). May return any number of rows, one column
   (`DOUBLE`/`INT64`/`STRING`); resolved into a literal list -- the same
   OR-chain desugar a literal `IN (1, 2, 3)` already gets -- before
   binding. An empty result is standard SQL semantics, not an error:
   `IN ()` is always false, `NOT IN ()` is always true. No size cap on
   the returned row count -- a narrow mechanism for a subquery expected
   to return a modest number of rows (e.g. a tight `HAVING` filter), not
   a general-purpose semi-join.
3. **`EXISTS (SELECT ...)` / `NOT EXISTS (SELECT ...)` as a top-level
   `WHERE` `AND`-conjunct** (`WHERE ... AND EXISTS (SELECT * FROM l
   WHERE l.k = o.k AND <predicate over l only>)`, TPC-H Q4's shape) --
   the one **correlated** form supported. Exactly one equality key
   correlating the subquery to a column already in scope, plus
   optionally one `AND`-conjunct referencing only the subquery's own
   source; no `JOIN`/derived-table `FROM`, `GROUP BY`, `HAVING`, `ORDER
   BY`, or `LIMIT` inside the subquery, and the outer `FROM` must
   already be aliased or joined. Rewritten into a `LEFT SEMI`
   (`EXISTS`)/`LEFT ANTI` (`NOT EXISTS`) join step *before* binding --
   runs on whichever backend the outer query does, unlike the two
   non-correlated forms above, since it becomes a real join step in the
   same physical plan rather than a separate planning-time execution.
   See `docs/ARCHITECTURE.md`'s "Correlated subqueries" section for the
   full scope and rewrite mechanics.

A subquery-as-value-expression anywhere else -- bare in `WHERE` outside
an `EXISTS`/`IN` wrapper, `SELECT`, `GROUP BY`, join `ON` -- is rejected
with a clear `BindingError`. Correlated *scalar* subqueries, `EXISTS`
outside the scope above (e.g. mixed with `OR` rather than `AND`, or
wrapping a subquery with its own `JOIN`), and a `HAVING` subquery
returning more than one row or more than one column are all unsupported.
A derived table in `FROM` (`FROM (SELECT ...) AS alias`, see "Joins"
above) is a different mechanism -- a *relation*, not a scalar/list
value -- and is supported, narrowly, as described there.

## Data types

| SQL type | `TypeId` | Notes |
| --- | --- | --- |
| `BOOLEAN` | `Boolean` | |
| `INT` | `Int32` | |
| `BIGINT` | `Int64` | |
| `DOUBLE` | `Float64` | |
| `VARCHAR(n)` | `String` | |
| `DATE` | `Date32` | Literal syntax: `DATE 'YYYY-MM-DD'` |
| `TIMESTAMP` | `Timestamp` | Column type only -- no `TIMESTAMP` literal syntax |
| `DECIMAL(p, s)` | `Decimal` | See caveats below |
| `UInt32`/`UInt64` | -- | Internal-only (e.g. generator output); no SQL literal/cast syntax produces these |

Nullability is read from the real source schema (Parquet
REQUIRED/OPTIONAL, or explicit for partition columns), not assumed --
`nullable=false` is a genuine, trusted signal.

### `DECIMAL` caveats

- Native arithmetic/comparison/cast-from-DECIMAL work on both backends.
  `CAST(expr AS DECIMAL(p, s))` is scoped to the `SELECT` list / `GROUP
  BY` key position only (same scope as `CASE`).
- Implicit promotion only retargets a numeric *literal* to a DECIMAL
  column's exact type -- it does not widen or cast a column/computed
  expression. Two DECIMALs with different precision/scale are rejected
  rather than auto-widened; wrap the non-literal side in an explicit
  `CAST`.
- `SUM`/`MIN`/`MAX` over DECIMAL keep the argument's own precision/scale
  (no DuckDB-style widening to rule out overflow).
- `AVG` over DECIMAL is rejected at bind time -- `CAST` to `DOUBLE` first.
- A DECIMAL literal only ever has `double` precision (hsql parses it that
  way), so it can't represent more significant digits than a `double`
  can.

## Aggregates

`SUM`, `COUNT`, `COUNT(*)`, `MIN`, `MAX`, `AVG` -- both backends. `AVG`
does not support a `DECIMAL` argument. No other function beyond these
five aggregates and `EXTRACT` is recognized.

## CPU vs. GPU backend differences

Both backends accept the same SQL and are validated to produce identical
results (`tools/validate_tpch.py`) wherever both support a query shape.
Three known gaps:

- `CASE`/`EXTRACT` in `WHERE` (`FilterOperator`'s own gap, GPU-only; not
  needed by any TPC-H query added so far).
- Subquery execution (`HAVING` or `IN`) always uses the CPU backend
  regardless of `--backend` (a deliberate design choice, not a
  capability gap -- see "Subqueries" above).
- A direct consequence of the point above: an exact `=`/`<>` comparison
  between a `HAVING`/`IN` subquery's own result and an outer aggregate
  (TPC-H Q15's own shape) is unreliable specifically when the *outer*
  query runs on the GPU backend -- it ends up comparing a GPU-computed
  floating-point sum against a CPU-computed one, and this project's
  monetary columns are `DOUBLE`, not `DECIMAL`, so the two essentially
  never round to the exact same last bit (empirically: 0/20 CPU-backend
  runs mismatched, 14/20 GPU-backend runs did). A real fix needs an
  externally-owned, safely-shared `RmmEnvironment` threaded into
  planning (subqueries can't just build their own GPU `RmmEnvironment`
  unconditionally -- the Flight SQL server can plan multiple concurrent
  requests against one shared `QueryEngine`, and each `RmmEnvironment`
  installs itself as *its device's* current CUDA device memory resource,
  shared across every concurrent query dispatched to that device --
  see `docs/GPU_OPTIMIZATIONS.md`'s "Multi-GPU Tier 1 implemented"
  section for the current one-`RmmEnvironment`-per-device model) --
  see `docs/ARCHITECTURE.md`'s "`HAVING` and scalar subqueries" section
  for the full investigation. `q15.sql`'s own header
  documents this; only validate/benchmark it with `--backend cpu` until
  a real fix exists.

## Not supported

`DISTINCT`, set operations (`UNION`/`INTERSECT`/`EXCEPT`), `WITH`/CTEs,
`OFFSET`, window functions, a derived table used *as a JOIN source*
(not to be confused with a JOIN *inside* a derived table's own `FROM`,
which TPC-H Q8 already proves works), correlated *scalar*
subqueries or `EXISTS` outside its narrow correlated scope (see
"Subqueries" above), `RIGHT`/`FULL`/`CROSS` JOIN, comma-style joins,
multi-key or non-equality join conditions, a JOIN `ON`-clause predicate
referencing the already-joined (left) side, `CASE`/`EXTRACT` in `WHERE`
on the GPU backend, and any function beyond the five aggregates and
`EXTRACT`. Each fails clearly at parse or bind time with a specific
error rather than being silently reinterpreted.

## TPC-H query coverage

16 of 22 TPC-H queries: **Q1, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12,
Q13, Q14, Q15, Q18, Q19** (Q15 is CPU-backend-only for now -- see "CPU
vs. GPU backend differences" above). See `docs/TPCH.md` for the
generate/query/validate/benchmark workflow and `docs/ROADMAP.md`'s
"Done" section for what each addition needed. Every remaining query is
blocked on a feature in the "Not supported" list above (most commonly
`DISTINCT`, set operations, `WITH`/CTEs, window functions, or a
correlated scalar subquery) -- Q8 needed no new SQL feature (just a real
execution-layer bug fix, see `docs/ARCHITECTURE.md`'s "Derived tables"
section), but every other remaining query is genuinely feature-blocked.
