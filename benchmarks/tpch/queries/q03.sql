-- TPC-H Q3 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviation from canonical TPC-H Q3: `FROM customer, orders, lineitem
-- WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey` -> a 3-way
-- `JOIN ... ON` chain (comma-style joins are not supported -- see
-- docs/ARCHITECTURE.md's "read_parquet(...) adapter" section -- and
-- KernelLake has no catalog/table-name resolution yet).
-- {data}/{orders_data}/{customer_data} are substituted by the caller with
-- globs over the lineitem/orders/customer Parquet files generate_tpch.py
-- produced. No other semantic change: `ORDER BY revenue DESC, o_orderdate`
-- (a multi-key sort after a 3-way JOIN) and `LIMIT 10` both work exactly
-- as canonical TPC-H writes them.
SELECT l_orderkey,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       o_orderdate,
       o_shippriority
FROM read_parquet('{customer_data}') AS c
JOIN read_parquet('{orders_data}') AS o ON c.c_custkey = o.o_custkey
JOIN read_parquet('{data}') AS l ON o.o_orderkey = l.l_orderkey
WHERE c_mktsegment = 'BUILDING'
  AND o_orderdate < DATE '1995-03-15'
  AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10
