-- TPC-H Q2 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query needing a correlated scalar subquery whose own inner
-- FROM is a *multi-way JOIN chain* (`partsupp`/`supplier`/`nation`/
-- `region`), not a single source (Q17's own shape) -- decorrelated
-- internally by `sql::rewrite_correlated_scalar_subqueries()` into a
-- JOIN against a synthesized derived table wrapping that *whole* inner
-- join chain, `GROUP BY`-aggregated on the one correlation column
-- (`ps_partkey`); the subquery's own additional, non-correlated
-- `r_name = 'EUROPE'` filter stays on the derived table's own `WHERE`
-- clause unchanged. See docs/ARCHITECTURE.md's "Correlated scalar
-- subqueries" section for the full scope.
--
-- Deviations from canonical TPC-H Q2:
--   1. `FROM part, supplier, partsupp, nation, region WHERE ...` ->
--      `JOIN ... ON` throughout, both in the outer query and the
--      correlated subquery (comma-style joins aren't supported).
--      {part_data}/{supplier_data}/{partsupp_data}/{nation_data}/
--      {region_data} are substituted by the caller with globs over the
--      part/supplier/partsupp/nation/region Parquet files
--      generate_tpch.py produced ({supplier_data}/{nation_data}/
--      {region_data} each appear *twice* -- once in the outer join,
--      once again in the correlated subquery's own re-join -- all get
--      the same substitution).
--   2. `[SIZE]`/`[TYPE]`/`[REGION]` substitution parameters resolved to
--      the representative literal values `15`/`%BRASS`/`EUROPE` -- same
--      fixed-literal simplification every other parameterized query in
--      this project's suite already makes.
-- No other semantic change: the outer 5-way join, the correlated
-- `MIN(ps_supplycost)` subquery's own 4-way join, and the final
-- `ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100` all work
-- exactly as canonical TPC-H writes them.
SELECT s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr, s.s_address, s.s_phone, s.s_comment
FROM read_parquet('{part_data}') AS p
JOIN read_parquet('{partsupp_data}') AS ps ON p.p_partkey = ps.ps_partkey
JOIN read_parquet('{supplier_data}') AS s ON s.s_suppkey = ps.ps_suppkey
JOIN read_parquet('{nation_data}') AS n ON s.s_nationkey = n.n_nationkey
JOIN read_parquet('{region_data}') AS r ON n.n_regionkey = r.r_regionkey
WHERE p.p_size = 15
  AND p.p_type LIKE '%BRASS'
  AND r.r_name = 'EUROPE'
  AND ps.ps_supplycost = (
    SELECT MIN(ps2.ps_supplycost)
    FROM read_parquet('{partsupp_data}') AS ps2
    JOIN read_parquet('{supplier_data}') AS s2 ON s2.s_suppkey = ps2.ps_suppkey
    JOIN read_parquet('{nation_data}') AS n2 ON s2.s_nationkey = n2.n_nationkey
    JOIN read_parquet('{region_data}') AS r2 ON n2.n_regionkey = r2.r_regionkey
    WHERE ps2.ps_partkey = p.p_partkey AND r2.r_name = 'EUROPE'
  )
ORDER BY s.s_acctbal DESC, n.n_name, s.s_name, p.p_partkey
LIMIT 100
