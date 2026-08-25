-- TPC-H Q21 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query with two chained EXISTS/NOT EXISTS steps correlated to
-- the *same* outer alias (`l1`) plus, unlike q04.sql's single EXISTS, a
-- residual (non-equality, cross-side) predicate inside each subquery's
-- own correlation (`l2.l_suppkey <> l1.l_suppkey`) -- rewritten internally
-- into two chained LEFT SEMI / LEFT ANTI joins, each carrying that
-- residual predicate (see docs/ARCHITECTURE.md's "Correlated subqueries"
-- section for the exact rewrite and the physical-plan machinery that
-- evaluates a residual predicate across both join sides at once via
-- cudf::mixed_left_semi_join()/mixed_left_anti_join() on the GPU backend
-- and Acero's own HashJoinNodeOptions::filter on the CPU backend).
--
-- Deviations from canonical TPC-H Q21:
--   1. `FROM supplier, lineitem l1, orders WHERE ...` -> `JOIN ... ON`
--      (comma-style joins aren't supported). {data}/{supplier_data}/
--      {orders_data}/{nation_data} are substituted by the caller with
--      globs over the lineitem/supplier/orders/nation Parquet files
--      generate_tpch.py produced ({data}=lineitem, reused for l1/l2/l3 --
--      same "one glob, multiple aliases" convention q09.sql's own
--      lineitem/partsupp reuse already establishes).
--   2. `[NATION]` substitution parameter resolved to the representative
--      literal `SAUDI ARABIA` -- same fixed-literal simplification every
--      other parameterized query in this project's suite already makes.
-- No other semantic change: the two correlated EXISTS/NOT EXISTS
-- subqueries, the 4-way JOIN, the GROUP BY, and the
-- `ORDER BY numwait DESC, s_name` multi-key sort all work exactly as
-- canonical TPC-H writes them.
SELECT s_name,
       COUNT(*) AS numwait
FROM read_parquet('{supplier_data}') AS s
JOIN read_parquet('{data}') AS l1 ON s.s_suppkey = l1.l_suppkey
JOIN read_parquet('{orders_data}') AS o ON o.o_orderkey = l1.l_orderkey
JOIN read_parquet('{nation_data}') AS n ON s.s_nationkey = n.n_nationkey
WHERE o.o_orderstatus = 'F'
  AND l1.l_receiptdate > l1.l_commitdate
  AND n.n_name = 'SAUDI ARABIA'
  AND EXISTS (
    SELECT *
    FROM read_parquet('{data}') AS l2
    WHERE l2.l_orderkey = l1.l_orderkey
      AND l2.l_suppkey <> l1.l_suppkey
  )
  AND NOT EXISTS (
    SELECT *
    FROM read_parquet('{data}') AS l3
    WHERE l3.l_orderkey = l1.l_orderkey
      AND l3.l_suppkey <> l1.l_suppkey
      AND l3.l_receiptdate > l3.l_commitdate
  )
GROUP BY s_name
ORDER BY numwait DESC, s_name
LIMIT 100
