-- TPC-H Q11 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query in this project needing HAVING or a subquery -- both
-- now genuinely supported (not flattened/rewritten away the way Q5/Q7/Q9
-- avoided needing them), scoped narrowly: HAVING may reference aggregates/
-- GROUP BY keys same as SELECT-list items already could; a subquery is
-- only legal as an operand inside HAVING's own boolean expression, must
-- be non-correlated, and must return exactly one row/one column (a true
-- scalar) -- see docs/ARCHITECTURE.md's HAVING section for the full
-- scope and how it's implemented (a real pre-bind pass runs the subquery
-- as its own complete, independent query -- always on the CPU backend,
-- regardless of this query's own --backend -- and splices its one
-- result value into the HAVING predicate as a literal before the outer
-- query is ever bound).
--
-- Deviations from canonical TPC-H Q11:
--   1. `FROM partsupp, supplier, nation WHERE ps_suppkey = s_suppkey AND
--      s_nationkey = n_nationkey AND n_name = 'GERMANY'` -> a 3-way
--      `JOIN ... ON` chain, in both the outer query and the (separately
--      scoped, independently aliased) subquery -- comma-style joins
--      aren't supported. {partsupp_data}/{supplier_data}/{nation_data}
--      are substituted by the caller with globs over the partsupp/
--      supplier/nation Parquet files generate_tpch.py produced; each
--      placeholder is substituted into *two* places (once per query),
--      the same `{nation_data}`-appears-twice trick Q7 already uses for
--      its own two nation JOIN steps.
--   2. This is the one query so far that doesn't reference `lineitem` at
--      all, so `{data}` isn't used anywhere in its text -- `--data`
--      itself is still unconditionally required by this project's own
--      tooling regardless of query, so callers still need to pass
--      *something* for it (any real lineitem glob; it's simply never
--      substituted into this query). `{partsupp_data}` (added for Q9)
--      is what actually supplies partsupp here, keeping `{data}`'s own
--      meaning ("lineitem") consistent across every query rather than
--      overloading it per-query.
-- No other semantic change: the scalar subquery's own `SUM(...) *
-- 0.0001000000` threshold, the `GROUP BY`/`HAVING` pair, and
-- `ORDER BY value DESC` all work exactly as canonical TPC-H writes them.
SELECT ps_partkey,
       SUM(ps_supplycost * ps_availqty) AS value
FROM read_parquet('{partsupp_data}') AS ps
JOIN read_parquet('{supplier_data}') AS s ON ps.ps_suppkey = s.s_suppkey
JOIN read_parquet('{nation_data}') AS n ON s.s_nationkey = n.n_nationkey
WHERE n_name = 'GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost * ps_availqty) > (
    SELECT SUM(ps_supplycost * ps_availqty) * 0.0001000000
    FROM read_parquet('{partsupp_data}') AS ps
    JOIN read_parquet('{supplier_data}') AS s ON ps.ps_suppkey = s.s_suppkey
    JOIN read_parquet('{nation_data}') AS n ON s.s_nationkey = n.n_nationkey
    WHERE n_name = 'GERMANY'
)
ORDER BY value DESC
