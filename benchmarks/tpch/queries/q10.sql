-- TPC-H Q10 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q10:
--   1. `FROM customer, orders, lineitem, nation WHERE c_custkey = o_custkey
--      AND l_orderkey = o_orderkey AND c_nationkey = n_nationkey` -> a
--      4-way `JOIN ... ON` chain (comma-style joins are not supported --
--      see docs/ARCHITECTURE.md's "read_parquet(...) adapter" section --
--      and KernelLake has no catalog/table-name resolution yet).
--      {data}/{orders_data}/{customer_data}/{nation_data} are substituted
--      by the caller with globs over the lineitem/orders/customer/nation
--      Parquet files generate_tpch.py produced.
--   2. `o_orderdate >= DATE ':1' AND o_orderdate < DATE ':1' + INTERVAL '3'
--      MONTH` -> two concrete DATE literals (no INTERVAL arithmetic; same
--      accommodation as q03.sql/q12.sql).
-- No other semantic change: the 7-column GROUP BY, `ORDER BY revenue DESC`,
-- and `LIMIT 20` all work exactly as canonical TPC-H writes them.
SELECT c_custkey,
       c_name,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       c_acctbal,
       n_name,
       c_address,
       c_phone,
       c_comment
FROM read_parquet('{customer_data}') AS c
JOIN read_parquet('{orders_data}') AS o ON c.c_custkey = o.o_custkey
JOIN read_parquet('{data}') AS l ON o.o_orderkey = l.l_orderkey
JOIN read_parquet('{nation_data}') AS n ON c.c_nationkey = n.n_nationkey
WHERE o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R'
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
ORDER BY revenue DESC
LIMIT 20
