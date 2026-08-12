-- TPC-H Q18 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query in this project needing `IN (SELECT ...)` -- a
-- non-correlated subquery used as the source of a WHERE-clause IN list,
-- genuinely supported (not flattened/rewritten away), narrowly scoped:
-- the subquery must be non-correlated and return a single column (any
-- number of rows); it's resolved into a literal list -- the same
-- OR-chain of equality comparisons a literal `IN (1, 2, 3)` already
-- desugars into at bind time -- before the outer query is ever bound.
-- See docs/ARCHITECTURE.md's "HAVING and scalar subqueries" section for
-- the full mechanism and its scale caveat (a narrow mechanism for a
-- subquery expected to return a modest number of rows, not a
-- general-purpose semi-join).
--
-- Deviations from canonical TPC-H Q18:
--   1. `FROM customer, orders, lineitem WHERE ... AND c_custkey =
--      o_custkey AND o_orderkey = l_orderkey` -> a 3-way `JOIN ... ON`
--      chain (comma-style joins aren't supported).
--      {customer_data}/{orders_data}/{data} are substituted by the
--      caller with globs over the customer/orders/lineitem Parquet
--      files generate_tpch.py produced; {data} is substituted into two
--      places (the outer join and the subquery's own FROM), the same
--      `{nation_data}`-appears-twice trick Q7 already uses.
-- No other semantic change: the subquery's own `GROUP BY`/`HAVING`
-- threshold, the outer 5-column `GROUP BY`, and the two-key
-- `ORDER BY revenue DESC, o_orderdate` all work exactly as canonical
-- TPC-H writes them.
SELECT c_name,
       c_custkey,
       o_orderkey,
       o_orderdate,
       o_totalprice,
       SUM(l_quantity) AS quantity_sum
FROM read_parquet('{customer_data}') AS c
JOIN read_parquet('{orders_data}') AS o ON c.c_custkey = o.o_custkey
JOIN read_parquet('{data}') AS l ON o.o_orderkey = l.l_orderkey
WHERE o_orderkey IN (
    SELECT l_orderkey
    FROM read_parquet('{data}')
    GROUP BY l_orderkey
    HAVING SUM(l_quantity) > 300
)
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate
LIMIT 100
