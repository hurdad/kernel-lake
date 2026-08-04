-- TPC-H Q12 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q12:
--   1. `FROM orders, lineitem WHERE o_orderkey = l_orderkey` -> `FROM
--      read_parquet('{orders_data}') AS o JOIN read_parquet('{data}') AS l
--      ON o.o_orderkey = l.l_orderkey` (comma-style joins are not
--      supported -- see docs/ARCHITECTURE.md's "read_parquet(...) adapter"
--      section -- and KernelLake has no catalog/table-name resolution
--      yet). {data}/{orders_data} are substituted by the caller with globs
--      over the lineitem/orders Parquet files generate_tpch.py produced.
--   2. `ORDER BY l_shipmode` is omitted -- KernelLake has no physical sort
--      operator for this shape yet (same accommodation as q01.sql; see
--      that file's own comment).
SELECT l_shipmode,
       SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END)
           AS high_line_count,
       SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END)
           AS low_line_count
FROM read_parquet('{orders_data}') AS o JOIN read_parquet('{data}') AS l ON o.o_orderkey = l.l_orderkey
WHERE l_shipmode IN ('MAIL', 'SHIP')
  AND l_commitdate < l_receiptdate
  AND l_shipdate < l_commitdate
  AND l_receiptdate >= DATE '1994-01-01'
  AND l_receiptdate < DATE '1995-01-01'
GROUP BY l_shipmode
