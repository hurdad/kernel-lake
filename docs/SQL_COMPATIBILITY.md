# SQL compatibility

A single, scannable reference for what SQL KernelLake accepts today. This
is a compatibility matrix, not an implementation guide -- for *why* a
given scope boundary exists, see `docs/ARCHITECTURE.md`'s "Supported SQL
grammar (current)" section and its sub-sections (`HAVING`/scalar
subqueries, `IN (SELECT ...)` subqueries, Hash joins, DECIMAL support,
LIKE/IN/CASE/CAST implementation notes). For which TPC-H queries this
adds up to, see `docs/TPCH.md` and `docs/ROADMAP.md`'s "Done"/"Not yet
started" sections. Everything below reflects the codebase as of the
TPC-H Q18 addition (12 of 22 TPC-H queries supported).

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
  JOIN read_parquet(...) AS b ON <a.col = b.col>
  [JOIN read_parquet(...) AS c ON <... = c.col>] ...
  [WHERE <expr>] [GROUP BY <cols>] [HAVING <bool expr>]
  [ORDER BY <cols>] [LIMIT <n>]
```

`read_parquet(...)` may be replaced by `read_iceberg('catalog.namespace.table')`,
`read_delta('table_uri')`, or `read_unity_catalog('instance.catalog.schema.table')`
in either shape, mixed freely across sources within one join chain.
`FROM`/`JOIN` accept only these four table-valued-function forms -- a real
table name, a subquery/derived table, or a comma-style join list are all
rejected at parse time.

| Clause | Supported | Notes |
| --- | --- | --- |
| `SELECT` list | Yes | Columns, aliases, `*`, expressions, aggregates |
| `FROM` | Single source only | `read_parquet`/`read_iceberg`/`read_delta`/`read_unity_catalog` |
| `JOIN ... ON` | Yes, N-way | `INNER` only, one equality key per step -- see "Joins" below |
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
| Subquery in `FROM` (derived table) | No | |

## Joins

- **Type**: `INNER JOIN` only. `LEFT`/`RIGHT`/`FULL`/`CROSS` JOIN and
  comma-style joins (`FROM a, b WHERE a.k = b.k`) are rejected at parse
  time.
- **Condition**: exactly one equality (`a.col = b.col`) between a plain
  column already in scope and a plain column from the newly joined
  source, same type. Multi-key and non-equality conditions are rejected
  at bind time -- a composite key must be split, with one key as the
  `JOIN ... ON` condition and the rest moved into `WHERE` (valid for
  `INNER JOIN`; see TPC-H Q5/Q9 for real examples).
- **Chain length**: up to 12 sources (`kMaxJoinSources`), each explicitly
  aliased. Verified in production shape up to 6-way (TPC-H Q5/Q7/Q9).
- Same-named columns from different join sides resolve correctly by
  qualified name (`a.x`/`b.x`), including a table joined to itself under
  two aliases (TPC-H Q7's `nation AS n1`/`AS n2`).

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
| `EXISTS` / correlated subqueries | No | |
| `CASE WHEN ... THEN ... [ELSE ...] END` | Yes, backend-scoped | CPU: everywhere. GPU: `SELECT` list, `GROUP BY` keys, aggregate arguments -- **not yet in `WHERE`** |
| `CAST(expr AS type)` | Yes | `INT`/`BIGINT`/`DOUBLE`/`VARCHAR(n)`/`DECIMAL(p, s)`; numeric-to-integer casts truncate (not round -- differs from DuckDB) |
| `EXTRACT(field FROM expr)` | Yes, backend-scoped | `field` is `YEAR`/`MONTH`/`DAY` only; `expr` must be `DATE`/`TIMESTAMP`. Both backends everywhere `CASE` can appear except `WHERE` |
| Aggregate-combining `SELECT` items (e.g. `100.0 * SUM(a) / SUM(b)`) | Yes | Both -- resolved in shared logical-plan construction |

## Subqueries

Exactly two forms are supported, both **non-correlated** (bound
independently against their own `FROM`/`JOIN` schema, with no access to
the outer query's tables or aliases) and both executed always on the CPU
(Acero) backend regardless of the outer query's own `--backend`, as a
side effect of planning (including inside `explain`):

1. **A scalar subquery as an operand inside `HAVING`'s own boolean
   expression** (`HAVING SUM(x) > (SELECT ...)`, TPC-H Q11's shape).
   Must return exactly one row, one column (`DOUBLE`/`INT64`/`STRING`).
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

A subquery anywhere else -- bare in `WHERE` (not inside `IN`), `SELECT`,
`FROM`, `GROUP BY`, join `ON` -- is rejected with a clear `BindingError`.
`EXISTS`/`NOT EXISTS`, correlated subqueries, and a `HAVING` subquery
returning more than one row or more than one column are all unsupported.

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
The only two known gaps, both GPU-only:

- `CASE`/`EXTRACT` in `WHERE` (`FilterOperator`'s own gap; not needed by
  any TPC-H query added so far).
- Subquery execution (`HAVING` or `IN`) always uses the CPU backend
  regardless of `--backend` (a deliberate design choice, not a
  capability gap -- see "Subqueries" above).

## Not supported

`DISTINCT`, set operations (`UNION`/`INTERSECT`/`EXCEPT`), `WITH`/CTEs,
`OFFSET`, window functions, subqueries in `FROM` (derived tables),
`EXISTS`/correlated subqueries, `LEFT`/`RIGHT`/`FULL`/`CROSS` JOIN,
comma-style joins, multi-key or non-equality join conditions,
`CASE`/`EXTRACT` in `WHERE` on the GPU backend, and any function beyond
the five aggregates and `EXTRACT`. Each fails clearly at parse or bind
time with a specific error rather than being silently reinterpreted.

## TPC-H query coverage

12 of 22 TPC-H queries: **Q1, Q3, Q5, Q6, Q7, Q9, Q10, Q11, Q12, Q14,
Q18, Q19**. See `docs/TPCH.md` for the generate/query/validate/benchmark
workflow and `docs/ROADMAP.md`'s "Done" section for what each addition
needed. Every remaining query is blocked on a feature in the "Not
supported" list above (most commonly an outer join, a derived table, or
`EXISTS`/a correlated subquery) -- none of the "cheap," no-new-SQL-
feature queries remain.
