-- TPC-H Q14 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q14:
--   1. `FROM lineitem, part WHERE l_partkey = p_partkey` -> `FROM
--      read_parquet('{data}') AS l JOIN read_parquet('{part_data}') AS p
--      ON l.l_partkey = p.p_partkey` (comma-style joins are not supported
--      -- see docs/ARCHITECTURE.md's "read_parquet(...) adapter" section --
--      and KernelLake has no catalog/table-name resolution yet).
--      {data}/{part_data} are substituted by the caller with globs over
--      the lineitem/part Parquet files generate_tpch.py produced.
--   2. No other semantic change: the SELECT list's `100.00 *
--      SUM(CASE WHEN p_type LIKE 'PROMO%' THEN ... ELSE 0 END) /
--      SUM(...)` shape (two aggregates combined arithmetically, one of
--      them a CASE with a LIKE condition) is exactly canonical TPC-H Q14 --
--      the two engine gaps this query needed (LIKE inside a CASE branch,
--      and a SELECT item that combines multiple aggregates arithmetically
--      rather than being a single bare aggregate call) are both now fixed
--      on both execution backends -- see docs/ARCHITECTURE.md.
SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
           / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM read_parquet('{data}') AS l JOIN read_parquet('{part_data}') AS p ON l.l_partkey = p.p_partkey
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
