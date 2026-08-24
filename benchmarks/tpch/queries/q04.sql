-- TPC-H Q4 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query using EXISTS/NOT EXISTS -- rewritten internally into a
-- LEFT SEMI JOIN (see docs/ARCHITECTURE.md's "Correlated subqueries"
-- section for the exact rewrite scope: a single equality correlation key
-- plus an inner-only auxiliary predicate, `l_commitdate < l_receiptdate`
-- here, same ON-clause-auxiliary-predicate machinery q13.sql's LEFT JOIN
-- already exercises).
--
-- Deviations from canonical TPC-H Q4:
--   1. `FROM orders WHERE ... EXISTS (SELECT * FROM lineitem WHERE
--      l_orderkey = o_orderkey AND ...)` -> both sources explicitly
--      aliased (`AS o` / `AS l`) -- required by the EXISTS rewrite (see
--      subquery_resolver.hpp's own doc comment on its scope). {data}/
--      {orders_data} are substituted by the caller with globs over the
--      lineitem/orders Parquet files generate_tpch.py produced (same
--      {data}=lineitem, {orders_data}=orders convention as q12.sql).
--   2. `o_orderdate >= DATE '1993-07-01' AND o_orderdate < DATE
--      '1993-07-01' + INTERVAL '3' MONTH` -> the upper bound is written as
--      the literal `DATE '1993-10-01'` -- KernelLake's expression grammar
--      has no INTERVAL arithmetic yet (same accommodation as q01.sql).
SELECT o.o_orderpriority,
       COUNT(*) AS order_count
FROM read_parquet('{orders_data}') AS o
WHERE o.o_orderdate >= DATE '1993-07-01'
  AND o.o_orderdate < DATE '1993-10-01'
  AND EXISTS (
    SELECT *
    FROM read_parquet('{data}') AS l
    WHERE l.l_orderkey = o.o_orderkey
      AND l.l_commitdate < l.l_receiptdate
  )
GROUP BY o.o_orderpriority
ORDER BY o.o_orderpriority
