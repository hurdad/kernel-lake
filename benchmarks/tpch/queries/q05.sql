-- TPC-H Q5 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q5:
--   1. `FROM customer, orders, lineitem, supplier, nation, region WHERE
--      c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey =
--      s_suppkey AND s_nationkey = n_nationkey AND n_regionkey =
--      r_regionkey` -> a 6-way `JOIN ... ON` chain (comma-style joins are
--      not supported -- see docs/ARCHITECTURE.md's "read_parquet(...)
--      adapter" section -- and KernelLake has no catalog/table-name
--      resolution yet). {data}/{orders_data}/{customer_data}/
--      {supplier_data}/{nation_data}/{region_data} are substituted by the
--      caller with globs over the lineitem/orders/customer/supplier/
--      nation/region Parquet files generate_tpch.py produced.
--   2. `c_nationkey = s_nationkey` (the "local supplier" requirement --
--      the customer and the line's supplier are in the same nation) is a
--      *second* condition, not expressible as part of a single JOIN's ON
--      clause (each JOIN step here is already a single equality against
--      the newly-joined table -- see docs/ARCHITECTURE.md's own "Hash
--      joins" scope note on multi-key conditions). Moved into WHERE
--      instead, an ordinary filter over the already-joined row rather
--      than a join mechanism of its own -- the same shape Q10's WHERE
--      clause already uses for predicates spanning more than one joined
--      source.
-- No other semantic change: the single-column GROUP BY and
-- `ORDER BY revenue DESC` both work exactly as canonical TPC-H writes
-- them.
SELECT n_name,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM read_parquet('{customer_data}') AS c
JOIN read_parquet('{orders_data}') AS o ON c.c_custkey = o.o_custkey
JOIN read_parquet('{data}') AS l ON o.o_orderkey = l.l_orderkey
JOIN read_parquet('{supplier_data}') AS s ON l.l_suppkey = s.s_suppkey
JOIN read_parquet('{nation_data}') AS n ON s.s_nationkey = n.n_nationkey
JOIN read_parquet('{region_data}') AS r ON n.n_regionkey = r.r_regionkey
WHERE r_name = 'ASIA'
  AND c_nationkey = s_nationkey
  AND o_orderdate >= DATE '1994-01-01'
  AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name
ORDER BY revenue DESC
