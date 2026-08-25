-- TPC-H Q17 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query needing a *correlated scalar subquery* in `WHERE`
-- (unlike Q22's non-correlated one): `l_quantity < (SELECT 0.2 *
-- AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)` -- an
-- inner aggregate whose own value depends on the outer row's own
-- `p_partkey`. Decorrelated internally by
-- `sql::rewrite_correlated_scalar_subqueries()` into a `JOIN` against a
-- synthesized, `GROUP BY l_partkey`-aggregated derived table (`SELECT
-- l_partkey, 0.2 * AVG(l_quantity) AS avg_qty FROM lineitem GROUP BY
-- l_partkey`), with the original comparison rewritten to compare against
-- that derived table's own output column instead of the subquery -- see
-- docs/ARCHITECTURE.md's "Correlated scalar subqueries" section for the
-- full scope and why this reuses the exact same GROUP BY/aggregate/JOIN
-- machinery any real query already goes through.
--
-- Deviations from canonical TPC-H Q17:
--   1. `FROM lineitem, part WHERE p_partkey = l_partkey` -> `JOIN ... ON`
--      (comma-style joins aren't supported). {data}/{part_data} are
--      substituted by the caller with globs over the lineitem/part
--      Parquet files generate_tpch.py produced ({data} appears *twice*
--      -- once in the outer join as `l`, once in the correlated
--      subquery as `l2` -- both get the same substitution).
--   2. `[BRAND]`/`[CONTAINER]` substitution parameters resolved to the
--      representative literal values `Brand#23`/`MED BOX` -- same
--      fixed-literal simplification every other parameterized query in
--      this project's suite already makes.
-- No other semantic change: the correlated subquery's own `0.2 *
-- AVG(l_quantity)` arithmetic-over-an-aggregate and the final
-- `SUM(l_extendedprice) / 7.0` scalar result both work exactly as
-- canonical TPC-H writes them.
SELECT SUM(l.l_extendedprice) / 7.0 AS avg_yearly
FROM read_parquet('{part_data}') AS p
JOIN read_parquet('{data}') AS l ON p.p_partkey = l.l_partkey
WHERE p.p_brand = 'Brand#23'
  AND p.p_container = 'MED BOX'
  AND l.l_quantity < (
    SELECT 0.2 * AVG(l2.l_quantity)
    FROM read_parquet('{data}') AS l2
    WHERE l2.l_partkey = p.p_partkey
  )
