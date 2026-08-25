-- TPC-H Q15 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- CPU BACKEND ONLY -- see "GPU backend caveat" below before running this
-- on GPU or trusting a GPU result.
--
-- The first query in this project needing a non-correlated scalar
-- subquery as a bare `HAVING` comparison whose own `FROM` is itself a
-- derived table -- `HAVING total_revenue = (SELECT MAX(total_revenue)
-- FROM (SELECT ... GROUP BY ...) AS r2)` below. Unlike Q11's own HAVING
-- subquery (a flat aggregate over a real JOIN chain), this one needs a
-- genuine two-level aggregation -- "the max of each supplier's own
-- grouped revenue sum" can't be expressed without an inner GROUP BY
-- feeding an outer MAX(...), hence the derived table. A real,
-- previously-undiscovered gap was found and fixed while adding this
-- query: `QueryEngine::run_subquery()` (which HAVING/`IN` subqueries
-- both go through) had no case at all for a subquery whose own FROM is
-- a derived table -- it silently fell through to its single-table
-- branch with an empty path list ("no data source given" at execution
-- time). Fixed by having run_subquery() delegate straight to the same
-- recursive `plan_logical_unoptimized()` a real top-level query already
-- uses, instead of reimplementing a narrower join-or-single-table
-- version of it. See docs/ARCHITECTURE.md's "`HAVING` and scalar
-- subqueries" section.
--
-- Deviations from canonical TPC-H Q15:
--   1. Canonical Q15 factors its per-supplier revenue computation into a
--      named `CREATE VIEW revenue0 (...) AS SELECT ...`, referenced
--      twice (once joined to `supplier`, once inside the `MAX(...)`
--      subquery) -- KernelLake has no `CREATE VIEW`/`WITH` (CTE)
--      support, so `revenue0`'s own definition is inlined at both call
--      sites instead: once as a real `supplier JOIN lineitem` (avoiding
--      a derived table as a JOIN source, itself still unsupported --
--      see docs/ARCHITECTURE.md's "Derived tables" section), and once
--      more, identically, inside the `HAVING` subquery's own derived
--      table. Both copies are written with the *exact same* JOIN and
--      `GROUP BY` shape (not simplified to a bare `lineitem`-only
--      `GROUP BY l_suppkey`, which the underlying data's own DOUBLE
--      (not DECIMAL) money columns make a real, observable difference
--      for -- floating-point `SUM` isn't associative, so summing the
--      same rows in the different physical order a `lineitem`-only
--      scan-and-group visits them versus a `supplier JOIN lineitem`
--      visits them can round the last one or two significant digits
--      differently. Canonical Q15's own shared `revenue0` view never
--      hits this, since it computes each supplier's revenue *once* and
--      reuses that exact value both places; this project's inlined
--      workaround has no such view to share, so keeping both copies
--      physically identical -- same JOIN, same `GROUP BY` columns -- is
--      what keeps the `=` comparison exact instead of spuriously
--      returning zero rows).
--   2. `date '1996-01-01' + interval '3' month` is written as the
--      literal `DATE '1996-04-01'` -- KernelLake's expression grammar
--      has no `INTERVAL` arithmetic yet (same accommodation as q01.sql).
--   3. `total_revenue` (canonical Q15's `revenue0`-provided column name)
--      is spelled out as its own `SUM(...)` expression in both the outer
--      query's `GROUP BY`/`HAVING` and the subquery's own derived table,
--      rather than referenced as a plain column -- a direct consequence
--      of deviation 1 above, not a separate change. {data}/
--      {supplier_data} are substituted by the caller with globs over the
--      lineitem/supplier Parquet files generate_tpch.py produced -- each
--      appears *twice* (the outer JOIN and the HAVING subquery's own
--      re-JOIN), both get the same substitution.
--
-- GPU backend caveat (real, investigated, not yet fixed): every `HAVING`/
-- `IN (SELECT ...)` subquery always executes on the CPU (Acero) backend,
-- deliberately, regardless of the outer query's own `--backend` -- see
-- `run_subquery()`'s own doc comment (query_engine.hpp) for why (nesting
-- a second GPU `RmmEnvironment` lifecycle inside planning is unsafe for
-- `QueryEngine`'s server-side callers, which can plan multiple requests
-- concurrently against one shared instance -- see docs/ARCHITECTURE.md's
-- "Concurrency" section). This query's own `=` comparison is therefore
-- always checking a GPU-computed `SUM` (the outer `HAVING`, when
-- `--backend gpu`) against a CPU-computed one (the subquery, always) --
-- and GPU and CPU floating-point summation (this project's monetary
-- columns are DOUBLE, not DECIMAL -- see generate_tpch.py's own
-- docstring) essentially never round to the exact same last bit. Verified
-- empirically: CPU backend is reliable (0/20 repeated runs mismatched);
-- GPU backend is not (14/20 mismatched, sometimes returning zero rows
-- instead of the one true match -- GPU hash-based multi-group aggregation
-- also has its own run-to-run non-determinism on top of the cross-backend
-- gap). Only validate/benchmark this specific query with `--backend cpu`
-- until a real fix exists; see docs/ARCHITECTURE.md's "`HAVING` and
-- scalar subqueries" section for the full investigation.
SELECT s.s_suppkey,
       s.s_name,
       s.s_address,
       s.s_phone,
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
FROM read_parquet('{supplier_data}') AS s
JOIN read_parquet('{data}') AS l ON s.s_suppkey = l.l_suppkey
WHERE l.l_shipdate >= DATE '1996-01-01' AND l.l_shipdate < DATE '1996-04-01'
GROUP BY s.s_suppkey, s.s_name, s.s_address, s.s_phone
HAVING SUM(l.l_extendedprice * (1 - l.l_discount)) = (
    SELECT MAX(total_revenue)
    FROM (
        SELECT s2.s_suppkey AS supplier_no,
               SUM(l2.l_extendedprice * (1 - l2.l_discount)) AS total_revenue
        FROM read_parquet('{supplier_data}') AS s2
        JOIN read_parquet('{data}') AS l2 ON s2.s_suppkey = l2.l_suppkey
        WHERE l2.l_shipdate >= DATE '1996-01-01' AND l2.l_shipdate < DATE '1996-04-01'
        GROUP BY s2.s_suppkey, s2.s_name, s2.s_address, s2.s_phone
    ) AS r2
)
ORDER BY s.s_suppkey
