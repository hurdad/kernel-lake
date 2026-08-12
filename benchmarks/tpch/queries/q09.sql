-- TPC-H Q9 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q9:
--   1. The canonical query wraps its 6-way join in a derived table
--      purely to compute `o_year`/`amount` once before grouping -- same
--      reasoning as Q7's own deviation #1: nothing here needs a real
--      subquery, so it flattens losslessly into one query.
--   2. `FROM part, supplier, lineitem, partsupp, orders, nation WHERE
--      s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey =
--      l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey
--      AND s_nationkey = n_nationkey AND p_name like '%green%'` -> a
--      6-way `JOIN ... ON` chain (comma-style joins are not supported).
--      {data}/{orders_data}/{supplier_data}/{nation_data}/{part_data}/
--      {partsupp_data} are substituted by the caller with globs over the
--      lineitem/orders/supplier/nation/part/partsupp Parquet files
--      generate_tpch.py produced.
--   3. `partsupp`'s join to `lineitem` is on *two* columns together
--      (`ps_partkey = l_partkey AND ps_suppkey = l_suppkey` -- partsupp's
--      own primary key is the pair), which a single `JOIN ... ON` step
--      here can't express (KernelLake's own join scope: one equality per
--      step -- see docs/ARCHITECTURE.md's "Hash joins"). Split the same
--      way Q5 already splits its own two-column `WHERE`-only condition:
--      `ps_partkey = l_partkey` becomes the `JOIN ... ON` key, and
--      `ps_suppkey = l_suppkey` moves into `WHERE` as an ordinary filter
--      over the already-joined row -- correct because `WHERE` applies
--      before `GROUP BY`/aggregation, so the intermediate
--      one-partkey-to-many-suppliers fan-out this JOIN step alone would
--      produce is narrowed back down to exactly the rows a true
--      composite-key join would produce, before any row is summed.
-- No other semantic change: `p_name LIKE '%green%'` (generate_tpch.py's
-- p_name generation was extended with real TPC-H dbgen color words --
-- see that script's own comment -- specifically so this filter matches
-- real rows instead of trivially returning nothing), the 2-column
-- `GROUP BY`, and `ORDER BY nation, o_year DESC` (mixed-direction
-- multi-key sort) all work exactly as canonical TPC-H writes them.
SELECT n_name AS nation,
       EXTRACT(YEAR FROM o_orderdate) AS o_year,
       SUM(l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity) AS sum_profit
FROM read_parquet('{part_data}') AS p
JOIN read_parquet('{data}') AS l ON p.p_partkey = l.l_partkey
JOIN read_parquet('{supplier_data}') AS s ON l.l_suppkey = s.s_suppkey
JOIN read_parquet('{partsupp_data}') AS ps ON ps.ps_partkey = l.l_partkey
JOIN read_parquet('{orders_data}') AS o ON o.o_orderkey = l.l_orderkey
JOIN read_parquet('{nation_data}') AS n ON s.s_nationkey = n.n_nationkey
WHERE p_name LIKE '%green%'
  AND ps_suppkey = l_suppkey
GROUP BY nation, o_year
ORDER BY nation, o_year DESC
