-- TPC-H Q20 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query nesting a correlated scalar subquery *inside* an
-- `IN (SELECT ...)` subquery's own body (`ps.ps_availqty > (SELECT 0.5 *
-- SUM(l_quantity) FROM lineitem WHERE l_partkey = ps_partkey AND
-- l_suppkey = ps_suppkey AND ...)`, itself nested inside `s_suppkey IN
-- (SELECT ps_suppkey FROM partsupp WHERE ...)`), and the first needing a
-- *two-column* correlation key (`l_partkey = ps_partkey AND l_suppkey =
-- ps_suppkey`). Handled by the exact same machinery as Q17/Q2 with no
-- extra wiring: `sql::rewrite_correlated_scalar_subqueries()` runs at the
-- top of every `QueryEngine::plan_logical_unoptimized()` call, including
-- the ones `run_subquery()` makes to plan an `IN`-subquery's own body
-- independently -- so it fires again, correctly, on that inner body's own
-- `WHERE` clause. The two-column correlation is split the same way TPC-H
-- Q9's own two-column join condition already is in this project: the
-- first column (`l_partkey = ps_partkey`) becomes the synthesized join
-- step's own `ON`-clause key, the second (`l_suppkey = ps_suppkey`) an
-- extra `WHERE` conjunct evaluated after that join. See
-- docs/ARCHITECTURE.md's "Correlated scalar subqueries" section for the
-- full scope.
--
-- Deviations from canonical TPC-H Q20:
--   1. `FROM supplier, nation WHERE ...` -> `JOIN ... ON` (comma-style
--      joins aren't supported). {supplier_data}, {nation_data},
--      {partsupp_data}, {part_data}, and {data} (lineitem) are
--      substituted by the caller with globs over the corresponding
--      Parquet files generate_tpch.py produced.
--   2. The nested `part`/`partsupp` `IN (SELECT ...)` subqueries'
--      single-source `FROM`s use *unqualified* column references
--      (`p_name`, not `p.p_name`) -- this project's single-table Binder
--      mode has no qualified-reference support at all (see Q16/Q18's own
--      identical convention for their own `NOT IN`/`IN` subqueries).
--   3. `l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR` -> the upper
--      bound is written as the literal `DATE '1995-01-01'` -- no
--      `INTERVAL` arithmetic (same accommodation as q01.sql/q04.sql).
--   4. `[NATION]` substitution parameter resolved to the representative
--      literal `CANADA` -- same fixed-literal simplification every other
--      parameterized query in this project's suite already makes.
-- No other semantic change: the two-level `IN`-subquery nesting, the
-- correlated `0.5 * SUM(l_quantity)` threshold, and the final
-- `ORDER BY s_name` all work exactly as canonical TPC-H writes them.
SELECT s.s_name, s.s_address
FROM read_parquet('{supplier_data}') AS s
JOIN read_parquet('{nation_data}') AS n ON s.s_nationkey = n.n_nationkey
WHERE n.n_name = 'CANADA'
  AND s.s_suppkey IN (
    SELECT ps.ps_suppkey
    FROM read_parquet('{partsupp_data}') AS ps
    WHERE ps.ps_partkey IN (
        SELECT p_partkey FROM read_parquet('{part_data}') WHERE p_name LIKE 'forest%'
    )
    AND ps.ps_availqty > (
        SELECT 0.5 * SUM(l.l_quantity)
        FROM read_parquet('{data}') AS l
        WHERE l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
          AND l.l_shipdate >= DATE '1994-01-01'
          AND l.l_shipdate < DATE '1995-01-01'
    )
  )
ORDER BY s.s_name
